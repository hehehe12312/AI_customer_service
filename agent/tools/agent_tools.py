from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest

import random, os
import threading
import requests
from dotenv import load_dotenv
from utils.config_handler import agent_conf
from utils.path_tools import get_abs_path
from utils.logger_handler import logger

load_dotenv()

vector_store = None
rag = None
profile_store = None
vector_memory = None
_memory_users: set[str] = set()
_memory_lock = threading.Lock()


def init_knowledge() -> None:
    """打开已有向量库索引并接上资料检索。不扫描资料文件。"""
    global vector_store, rag
    if rag is not None:
        return
    from rag.vector_store import VectorStoreService
    from rag.rag_service import RagSummarizeService

    vector_store = VectorStoreService()
    rag = RagSummarizeService(vector_store)


def _summarize_new_file(path: str, text: str) -> str:
    """只概括这一份新资料。失败时抛出，由同步任务记下并继续。"""
    from model.factory import get_chat_model

    clipped = (text or "").strip()
    if len(clipped) > 6000:
        clipped = clipped[:6000]
    name = os.path.basename(path)
    result = get_chat_model().invoke(
        "请概括下面这份店铺资料，保留尺码、参数、时效和规则，用简体中文，不要编造。\n"
        f"文件名：{name}\n{clipped}"
    )
    content = getattr(result, "content", "")
    if isinstance(content, list):
        content = "".join(str(part) for part in content)
    return str(content).strip()


def _read_md5_lines() -> list[str]:
    from utils.config_handler import chroma_conf

    path = get_abs_path(chroma_conf["md5_hex_store"])
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _texts_for_new_files(new_md5: set[str]) -> list[tuple[str, str]]:
    from utils.config_handler import chroma_conf
    from utils.file_handler import (
        csv_loader,
        get_file_md5_hex,
        listdir_with_allowed_type,
        pdf_loader,
        txt_loader,
    )

    paths = listdir_with_allowed_type(
        get_abs_path(chroma_conf["data_path"]),
        tuple(chroma_conf["allow_knowledge_file_type"]),
    )
    found = []
    for path in paths:
        digest = get_file_md5_hex(path)
        if digest not in new_md5:
            continue
        if path.endswith("txt"):
            docs = txt_loader(path)
        elif path.endswith("pdf"):
            docs = pdf_loader(path)
        elif path.endswith("csv"):
            docs = csv_loader(path)
        else:
            continue
        text = "\n".join(doc.page_content for doc in docs if getattr(doc, "page_content", ""))
        if text.strip():
            found.append((path, text))
    return found


def sync_documents() -> None:
    """扫描资料文件，没变的跳过，新的写入向量库，并只为新文件补摘要。不挡住对话。"""
    if vector_store is None:
        return
    before = set(_read_md5_lines())
    logger.info("开始同步资料文件")
    vector_store.load_document()
    new_md5 = {line for line in _read_md5_lines() if line not in before}
    added = _texts_for_new_files(new_md5) if new_md5 else []
    for path, text in added:
        try:
            summary = _summarize_new_file(path, text)
        except Exception as exc:
            logger.error(f"资料摘要失败 {path}: {exc}")
            continue
        if not summary:
            logger.warning(f"资料摘要为空，跳过 {path}")
            continue
        from langchain_core.documents import Document

        vector_store.vector_store.add_documents(
            [Document(page_content=summary, metadata={"source": path, "kind": "摘要"})]
        )
        logger.info(f"已写入资料摘要: {path}")
    logger.info("资料文件同步结束")


def ensure_user_memory(user_id: str) -> None:
    """这个用户第一次说话时再打开画像和用户记忆。"""
    global profile_store, vector_memory
    user_id = str(user_id or "").strip() or "default"
    with _memory_lock:
        if user_id in _memory_users:
            return
        if profile_store is None:
            from memory.profile_store import ProfileStore

            profile_store = ProfileStore()
        if vector_memory is None:
            from memory.vector_memory import VectorMemory

            vector_memory = VectorMemory()
        _memory_users.add(user_id)

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}

_QWEATHER_KEY = os.getenv("QWEATHER_API_KEY", "")
# 优先用独立 API Host 拼路径；未配置时回退到完整 URL（旧公共域名对新 Key 会 Invalid Host）
_QWEATHER_HOST = (os.getenv("QWEATHER_API_HOST") or "").strip().removeprefix("https://").removeprefix("http://").rstrip("/")
_GEO_URL = (
    f"https://{_QWEATHER_HOST}/geo/v2/city/lookup"
    if _QWEATHER_HOST
    else os.getenv("QWEATHER_GEO_URL", "https://geoapi.qweather.com/v2/city/lookup")
)
_WEATHER_URL = (
    f"https://{_QWEATHER_HOST}/v7/weather/now"
    if _QWEATHER_HOST
    else os.getenv("QWEATHER_WEATHER_URL", "https://devapi.qweather.com/v7/weather/now")
)
_INDICES_URL = (
    f"https://{_QWEATHER_HOST}/v7/indices/1d"
    if _QWEATHER_HOST
    else os.getenv("QWEATHER_INDICES_URL", "https://devapi.qweather.com/v7/indices/1d")
)


def _qweather_headers() -> dict:
    return {
        "X-QW-Api-Key": _QWEATHER_KEY,
        "Accept-Encoding": "gzip",
        "User-Agent": "zst-agent/1.0",
    }


def _lookup_city_id(city: str) -> tuple[str, str] | tuple[None, None]:
    """通过和风地理接口把城市名解析为 Location ID。"""
    resp = requests.get(
        _GEO_URL,
        params={"location": city, "number": 1},
        headers=_qweather_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if str(data.get("code")) != "200" or not data.get("location"):
        return None, None
    loc = data["location"][0]
    name = loc.get("name") or city
    adm = loc.get("adm1") or ""
    display = f"{name}" if not adm or adm == name else f"{adm}{name}"
    return loc["id"], display


def _fetch_weather_now(location_id: str) -> dict | None:
    """查询和风实时天气。"""
    resp = requests.get(
        _WEATHER_URL,
        params={"location": location_id},
        headers=_qweather_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if str(data.get("code")) != "200" or not data.get("now"):
        return None
    return data["now"]


def _fetch_uv_index(location_id: str) -> str:
    """查询和风紫外线生活指数（type=5），失败时返回「暂无」。"""
    try:
        resp = requests.get(
            _INDICES_URL,
            params={"location": location_id, "type": "5"},
            headers=_qweather_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if str(data.get("code")) != "200" or not data.get("daily"):
            return "暂无"
        item = data["daily"][0]
        category = item.get("category") or ""
        level = item.get("level") or ""
        text = item.get("text") or ""
        if category and level:
            detail = f"{category}（等级{level}）"
        elif category:
            detail = category
        else:
            detail = "暂无"
        if text:
            return f"{detail}，说明：{text}"
        return detail
    except Exception as e:
        logger.warning(f"[get_weather]紫外线指数获取失败: {e}")
        return "暂无"


@tool(description="从向量存储中检索冲锋衣/防晒衣参数、尺码表、洗护保养与售后规则等参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="从和风天气获取指定城市的气温、降雨/降水与紫外线信息，以消息字符形式返回")
def get_weather(city: str) -> str:
    if not _QWEATHER_KEY:
        return f"未能获取城市{city}的天气：未配置和风天气 API Key"
    try:
        location_id, display_name = _lookup_city_id(city)
        if not location_id:
            return f"未能识别城市「{city}」，请换一个更具体的城市名后再试"
        now = _fetch_weather_now(location_id)
        if not now:
            return f"已定位到{display_name}，但未能获取实时天气，请稍后重试"
        text = now.get("text", "未知")
        temp = now.get("temp", "--")
        precip = now.get("precip", "0")
        # 旧版长文案字段（湿度/能见度等）保留注释，便于回滚
        # humidity = now.get("humidity", "--")
        # wind_dir = now.get("windDir", "")
        # wind_scale = now.get("windScale", "--")
        # feels_like = now.get("feelsLike", "--")
        # vis = now.get("vis", "--")
        uv = _fetch_uv_index(location_id)
        # 旧返回：
        # return (
        #     f"城市{display_name}天气为{text}，气温{temp}摄氏度，体感温度{feels_like}摄氏度，"
        #     f"空气湿度{humidity}%，{wind_dir}{wind_scale}级，能见度{vis}公里，降水量{precip}毫米"
        # )
        return (
            f"城市{display_name}天气为{text}，气温{temp}摄氏度，"
            f"降水量{precip}毫米，紫外线{uv}"
        )
    except requests.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:300] if e.response is not None else ""
        except Exception:
            pass
        logger.warning(f"[get_weather]和风天气 HTTP 错误: {e}; body={body}")
        if e.response is not None and e.response.status_code == 403:
            if "Invalid Host" in body:
                return (
                    f"获取城市{city}天气失败：和风天气 API Host 无效。"
                    f"请到控制台设置复制 API Host，填入 .env 的 QWEATHER_API_HOST 后重启"
                )
            if "Security Restriction" in body or "security restriction" in body:
                return (
                    f"获取城市{city}天气失败：和风凭据触发了安全限制（Security Restriction）。"
                    f"请到控制台 → 项目 → 该 API KEY，检查「应用限制/API 限制」："
                    f"开发阶段建议先清空 IP/网站白名单，或把本机公网 IP 加入白名单；"
                    f"API 限制需包含城市查询、实时天气与生活指数"
                )
        return f"获取城市{city}天气失败，请稍后重试"
    except requests.RequestException as e:
        logger.warning(f"[get_weather]和风天气请求失败: {e}")
        return f"获取城市{city}天气失败，请稍后重试"
    except Exception as e:
        logger.warning(f"[get_weather]解析天气数据失败: {e}")
        return f"获取城市{city}天气失败，请稍后重试"


# ---- 以下为扫地机旧工具，保留但不注册到户外店铺客服 Agent ----
@tool(description="获取用户所在城市名称，以纯字符形式返回")
def get_user_location() -> str:
    return random.choice(["深圳", "合肥", "杭州"])


@tool(description="获取用户ID，以纯字符形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data():
    if not external_data:
        if "external_data_path" not in agent_conf:
            raise KeyError("配置中缺少 external_data_path 字段")

        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件不存在: {external_data_path}")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr = line.strip().split(",")

                user_id = arr[0].replace('"', "")
                feature = arr[1].replace('"', "")
                efficiency = arr[2].replace('"', "")
                consumables = arr[3].replace('"', "")
                comparison = arr[4].replace('"', "")
                time = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }



@tool(description="检索指定用户在指定月份的扫地/扫拖机器人完整使用记录，以纯字符形式返回，如未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warn(f"[fetch_external_data]未能检索到用户:{user_id}在{month}的数据。")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成场景动态注入上下文信息，为后续提示词切换提供上下文支撑")
def fill_context_for_report():
    return "fill_context_for_report已调用"


@tool(description="将用户长期信息写入画像或向量记忆。key 常用 city/robot_model/preferences/note；key=note 时同时写入向量记忆")
def save_user_memory(user_id: str, key: str, value: str) -> str:
    user_id = (user_id or "").strip()
    key = (key or "").strip()
    value = (value or "").strip()
    if not user_id or not key or not value:
        return "保存失败：user_id、key、value 均不能为空"
    if key == "note":
        profile_store.update(user_id, {"notes": value})
        vector_memory.add(user_id, value)
        return f"已为用户{user_id}保存备注记忆: {value}"
    profile_store.update(user_id, {key: value})
    # 结构化字段也写入一条自然语言，便于语义召回
    vector_memory.add(user_id, f"{key}: {value}")
    return f"已为用户{user_id}保存画像字段 {key}={value}"


@tool(description="召回指定用户的长期画像与相关事实记忆，入参为 user_id 与检索词 query")
def recall_user_memory(user_id: str, query: str) -> str:
    user_id = (user_id or "").strip()
    query = (query or "").strip() or "用户偏好与基本信息"
    if not user_id:
        return "召回失败：user_id 不能为空"
    profile_text = profile_store.summary(user_id)
    facts = vector_memory.search(user_id, query)
    if facts:
        fact_text = "\n".join(f"- {item}" for item in facts)
    else:
        fact_text = "- 暂无相关向量记忆"
    return f"【用户画像】\n{profile_text}\n\n【相关记忆】\n{fact_text}"


if __name__ == "__main__":
    # 单独跑本文件时，默认测一下和风天气
    print(get_weather.invoke({"city": "深圳"}))