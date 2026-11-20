import time
from urllib.parse import quote

import streamlit as st
from dotenv import load_dotenv

from utils.secrets_loader import apply_streamlit_secrets, missing_required_secrets

# 红色线性垃圾桶图标（贴近参考图：圆角描边、无填充）
_DELETE_ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">
  <path d="M9 3.75h6" stroke="#EF4444" stroke-width="1.75" stroke-linecap="round"/>
  <path d="M4.5 6.5h15" stroke="#EF4444" stroke-width="1.75" stroke-linecap="round"/>
  <path d="M6.4 6.5V18a2.2 2.2 0 0 0 2.2 2.2h6.8A2.2 2.2 0 0 0 17.6 18V6.5"
        stroke="#EF4444" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M10 10.2v6.2M14 10.2v6.2" stroke="#EF4444" stroke-width="1.75" stroke-linecap="round"/>
</svg>"""
_DELETE_ICON_URI = "data:image/svg+xml;charset=utf-8," + quote(_DELETE_ICON_SVG)

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

from agent.react_agent import ReactAgent
from memory.conversation_store import ConversationStore

DEFAULT_GREETING = {
    "role": "assistant",
    "content": "你好，欢迎光临本店，我是户外运动店铺客服。发我城市和身高体重，我可以按天气推荐一款服饰并给尺码建议；也可以直接问选购场景、穿着搭配、发货时效、商品参数、洗护保养或售后退换。",
}
PROFILE_USER_ID = "default"

conversation_store = ConversationStore()

# 主题/知识库变更后 bump 此版本，避免 session 里缓存旧 Agent（含失效的 Chroma collection）
AGENT_BUILD = "outdoor-cs-v5"
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
    /* 侧栏会话行：标题与删除按钮垂直居中对齐 */
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {{
        align-items: center !important;
        gap: 0.15rem !important;
    }}
    /* 删除按钮：透明底 + 红色垃圾桶图标 */
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:last-child div[data-testid="stButton"] {{
        display: flex !important;
        justify-content: center !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:last-child button {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        width: 2rem !important;
        height: 2rem !important;
        min-height: 2rem !important;
        border-radius: 0.45rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: background-color 0.15s ease, transform 0.15s ease !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:last-child button:hover {{
        background: rgba(239, 68, 68, 0.12) !important;
        transform: scale(1.06);
    }}
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:last-child button:focus {{
        box-shadow: none !important;
        outline: none !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:last-child button p,
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:last-child button span {{
        font-size: 0 !important;
        line-height: 0 !important;
        color: transparent !important;
        margin: 0 !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
    div[data-testid="column"]:last-child button::before {{
        content: "";
        display: block;
        width: 1.15rem;
        height: 1.15rem;
        background: url("{_DELETE_ICON_URI}") center / contain no-repeat;
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
            col_open, col_del = st.columns([6, 1], vertical_alignment="center")
            with col_open:
                if st.button(label, key=f"session_{sid}", use_container_width=True):
                    if not is_current:
                        open_session(sid)
                        st.rerun()
            with col_del:
                if st.button(
                    "删除",
                    key=f"del_{sid}",
                    use_container_width=True,
                    help="删除该会话",
                    type="tertiary",
                ):
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
                    time.sleep(0.001)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        assistant_text = response_messages[-1] if response_messages else ""
        st.session_state["message"].append({"role": "assistant", "content": assistant_text})
        conversation_store.save_session(
            st.session_state["session_id"],
            st.session_state["message"],
        )
        st.rerun()


