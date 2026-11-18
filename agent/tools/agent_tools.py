from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest

from rag.vector_store import VectorStoreService
from rag.rag_service import RagSummarizeService
import random, os
import requests
from dotenv import load_dotenv
from utils.config_handler import agent_conf
from utils.path_tools import get_abs_path
from utils.logger_handler import logger

load_dotenv()

vector_store = VectorStoreService()
rag = RagSummarizeService(vector_store)

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


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="从和风天气获取指定城市的实时天气，以消息字符形式返回")
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
        humidity = now.get("humidity", "--")
        wind_dir = now.get("windDir", "")
        wind_scale = now.get("windScale", "--")
        feels_like = now.get("feelsLike", "--")
        vis = now.get("vis", "--")
        precip = now.get("precip", "0")
        return (
            f"城市{display_name}天气为{text}，气温{temp}摄氏度，体感温度{feels_like}摄氏度，"
            f"空气湿度{humidity}%，{wind_dir}{wind_scale}级，能见度{vis}公里，降水量{precip}毫米"
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
                    f"API 限制需包含城市查询与实时天气"
                )
        return f"获取城市{city}天气失败，请稍后重试"
    except requests.RequestException as e:
        logger.warning(f"[get_weather]和风天气请求失败: {e}")
        return f"获取城市{city}天气失败，请稍后重试"
    except Exception as e:
        logger.warning(f"[get_weather]解析天气数据失败: {e}")
        return f"获取城市{city}天气失败，请稍后重试"


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

if __name__ == "__main__":
    # 单独跑本文件时，默认测一下和风天气
    print(get_weather.invoke({"city": "深圳"}))