import base64
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.secrets_loader import apply_streamlit_secrets, missing_required_secrets

_DELETE_ICON_PATH = Path(__file__).resolve().parent / "assets" / "delete_trash.png"
_DELETE_ICON_B64 = base64.b64encode(_DELETE_ICON_PATH.read_bytes()).decode("ascii")

load_dotenv()
loaded_keys = apply_streamlit_secrets()
missing = missing_required_secrets()

st.title("户外运动店铺客服")
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

from agent.react_agent import ReactAgent
from memory.conversation_store import ConversationStore

DEFAULT_GREETING = {
    "role": "assistant",
    "content": "你好，欢迎光临本店，我是户外运动店铺客服。告诉我城市和身高体重，我可以根据天气为您推荐一款冲锋衣或防晒衣并给尺码建议；也可以咨询商品参数、洗护保养或售后退换规则。",
}
PROFILE_USER_ID = "default"

conversation_store = ConversationStore()

# 主题/知识库变更后 bump 此版本，避免 session 里缓存旧 Agent（含失效的 Chroma collection）
AGENT_BUILD = "outdoor-cs-v3"
if st.session_state.get("agent_build") != AGENT_BUILD or "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()
    st.session_state["agent_build"] = AGENT_BUILD


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

st.markdown(
    f"""
    <style>
    /* 侧栏会话行右侧：垃圾桶图标按钮（无文字） */
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:last-child button {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.2rem !important;
        min-height: 2rem !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:last-child button p,
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:last-child button span {{
        font-size: 0 !important;
        color: transparent !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:last-child button::before {{
        content: "";
        display: block;
        width: 20px;
        height: 20px;
        margin: 0 auto;
        background: url("data:image/png;base64,{_DELETE_ICON_B64}") center / contain no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    if st.button("新对话", use_container_width=True, key="new_chat_btn"):
        start_new_session()
        st.rerun()

    st.markdown("**最近**")
    recent_sessions = conversation_store.list_sessions()
    if not recent_sessions:
        st.caption("暂无聊天记录")
    else:
        for item in recent_sessions:
            sid = item["session_id"]
            title = item["title"]
            is_current = sid == st.session_state.get("session_id")
            label = f"› {title}" if is_current else title
            col_open, col_del = st.columns([5, 1])
            with col_open:
                if st.button(label, key=f"session_{sid}", use_container_width=True):
                    if not is_current:
                        open_session(sid)
                        st.rerun()
            with col_del:
                if st.button("-", key=f"del_{sid}", use_container_width=True, help="删除该会话"):
                    delete_session_and_switch(sid)
                    st.rerun()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(
            st.session_state["message"],
            user_id=PROFILE_USER_ID,
        )

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        assistant_text = response_messages[-1] if response_messages else ""
        st.session_state["message"].append({"role": "assistant", "content": assistant_text})
        conversation_store.save_session(
            st.session_state["session_id"],
            st.session_state["message"],
        )
        st.rerun()
