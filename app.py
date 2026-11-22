import streamlit as st
from dotenv import load_dotenv
from utils.secrets_loader import apply_streamlit_secrets, missing_required_secrets

load_dotenv()
loaded_keys = apply_streamlit_secrets()
missing = missing_required_secrets()

st.title("户外运动店铺智能客服")
st.divider()

if missing:
    st.error("Streamlit Secrets 缺少必要配置，应用无法启动。")
    st.markdown("请到 **Manage app → Settings → Secrets**，粘贴下面 TOML（把值换成你的真实密钥）：")
    st.code(
        '''DEEPSEEK_API_KEY = "你的DeepSeek密钥"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

VOYAGE_API_KEY = "你的Voyage密钥"

QWEATHER_API_KEY = "你的和风密钥"
QWEATHER_API_HOST = "你的host.re.qweatherapi.com"''',
        language="toml",
    )
    st.write("当前缺失的 key：", ", ".join(missing))
    st.write("已读到的 Secrets key：", ", ".join(loaded_keys) if loaded_keys else "（无）")
    st.stop()

from memory.conversation_store import ConversationStore

DEFAULT_GREETING = {
    "role": "assistant",
    "content": "你好，欢迎光临本店，我是户外运动店铺客服。发我城市和身高体重，我可以按天气推荐一款服饰并给尺码建议；也可以直接问选购场景、穿着搭配、发货时效、商品参数、洗护保养或售后退换。",
}
PROFILE_USER_ID = "default"

conversation_store = ConversationStore()

# 主题/知识库变更后 bump 此版本，避免 session 里缓存旧 Agent（含失效的 Chroma collection）
AGENT_BUILD = "outdoor-cs-v5"


def get_agent():
    from agent.loader import get_agent as load_agent

    return load_agent(AGENT_BUILD)


def _awaiting_reply(messages: list) -> bool:
    """最后一条有效消息是用户说的，说明回复还没写上。"""
    for item in reversed(messages):
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return item.get("role") == "user"
    return False


def start_new_session() -> None:
    created = conversation_store.create_session(DEFAULT_GREETING.copy())
    st.session_state["session_id"] = created["session_id"]
    st.session_state["message"] = list(created["messages"])


def open_session(session_id: str) -> None:
    st.session_state["session_id"] = session_id
    loaded = conversation_store.load_session(session_id)
    st.session_state["message"] = loaded if loaded else [DEFAULT_GREETING.copy()]


def delete_session_and_switch(session_id: str) -> None:
    conversation_store.delete_session(session_id)
    if st.session_state.get("session_id") != session_id:
        return
    remaining = conversation_store.list_sessions()
    if remaining:
        open_session(remaining[0]["session_id"])
    else:
        start_new_session()


# 启动：有最近会话则打开最新；否则新建
if "session_id" not in st.session_state or "message" not in st.session_state:
    recent = conversation_store.list_sessions()
    if recent:
        open_session(recent[0]["session_id"])
    else:
        start_new_session()


with st.sidebar:
    st.subheader("**控制面板**")
    if st.button("新建会话", use_container_width=True, key="new_chat_btn"):
        start_new_session()
        st.rerun()

    st.markdown("**最近**")
    recent_sessions = conversation_store.list_sessions()
    current_sid = st.session_state.get("session_id")
    # 新建会话尚无用户消息时 list_sessions 会过滤掉，侧栏仍展示当前「新对话」
    if current_sid and all(s["session_id"] != current_sid for s in recent_sessions):
        recent_sessions = [{"session_id": current_sid, "title": "新对话"}] + recent_sessions
    if not recent_sessions:
        st.caption("暂无聊天记录")
    else:
        for item in recent_sessions:
            sid = item["session_id"]
            title = item["title"]
            is_current = sid == st.session_state.get("session_id")
            btn_type = "secondary" if is_current else "tertiary"
            col_open, col_del = st.columns([6, 1], vertical_alignment="center")
            with col_open:
                if st.button(
                    title,
                    key=f"session_{sid}",
                    use_container_width=True,
                    type=btn_type,
                    icon="💬",
                ):
                    if not is_current:
                        open_session(sid)
                        st.rerun()
            with col_del:
                if st.button(
                    "",
                    icon="🗑️",
                    key=f"del_{sid}",
                    use_container_width=True,
                    help="删除该会话",
                    type="tertiary",
                ):
                    delete_session_and_switch(sid)
                    st.rerun()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input(placeholder="请输入")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})
    # 先落盘。页面刚打开就发送时，这一轮可能被打断，下一轮仍能接着回复。
    conversation_store.save_session(
        st.session_state["session_id"],
        st.session_state["message"],
    )

if _awaiting_reply(st.session_state["message"]):
    with st.spinner("客服思考中..."):
        res_stream = get_agent().execute_stream(
            st.session_state["message"],
            user_id=PROFILE_USER_ID,
        )
        with st.chat_message("assistant"):
            assistant_text = st.write_stream(res_stream)
    if not isinstance(assistant_text, str):
        assistant_text = ""
    assistant_text = assistant_text.strip()
    if not assistant_text:
        assistant_text = "抱歉，这次没有生成出回复，请再发一次。"
    st.session_state["message"].append({"role": "assistant", "content": assistant_text})
    conversation_store.save_session(
        st.session_state["session_id"],
        st.session_state["message"],
    )
    st.rerun()

# 页面元素先登记完，脚本结束页面才会显示；知识库和模型图在这之后再加载。
from agent.loader import start_agent_load

start_agent_load(AGENT_BUILD)


