import time

import streamlit as st
from dotenv import load_dotenv

from utils.secrets_loader import apply_streamlit_secrets, missing_required_secrets

load_dotenv()
loaded_keys = apply_streamlit_secrets()
missing = missing_required_secrets()

st.title("智能客服")
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
from memory.profile_store import ProfileStore

DEFAULT_GREETING = {
    "role": "assistant",
    "content": "你好，我是扫地机器人智能客服，请问有什么可以帮助你？",
}

conversation_store = ConversationStore()
profile_store = ProfileStore()


def known_session_ids() -> list[str]:
    """只保留已有对话或画像记忆的会话 ID。"""
    ids = set(conversation_store.list_user_ids()) | set(profile_store.list_user_ids())
    return sorted(ids)


if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

memory_ids = known_session_ids()
if "user_id" not in st.session_state:
    st.session_state["user_id"] = memory_ids[0] if memory_ids else ""

with st.sidebar:
    if st.button("新对话", use_container_width=True, help="打开新的聊天窗口"):
        st.session_state["message"] = [DEFAULT_GREETING.copy()]
        st.session_state["fresh_chat"] = True
        st.rerun()

    st.subheader("会话记忆")
    if memory_ids:
        current = st.session_state["user_id"]
        index = memory_ids.index(current) if current in memory_ids else 0
        selected_session = st.selectbox(
            "选择会话 ID",
            memory_ids,
            index=index,
            help="仅显示已有对话或画像记忆的会话",
        )
        if selected_session != st.session_state["user_id"]:
            st.session_state["user_id"] = selected_session
            st.session_state["fresh_chat"] = False
            loaded = conversation_store.load(selected_session)
            st.session_state["message"] = loaded if loaded else [DEFAULT_GREETING.copy()]
            st.rerun()
    else:
        st.caption("暂无已保存记忆的会话")

    new_session = st.text_input("新建会话 ID", placeholder="例如 1003")
    if st.button("切换到新会话") and new_session.strip():
        st.session_state["user_id"] = new_session.strip()
        st.session_state["fresh_chat"] = False
        loaded = conversation_store.load(st.session_state["user_id"])
        st.session_state["message"] = loaded if loaded else [DEFAULT_GREETING.copy()]
        st.rerun()

if not st.session_state.get("user_id"):
    st.info("请先在左侧输入会话 ID 并点击「切换到新会话」。")
    st.stop()

if "message" not in st.session_state:
    if st.session_state.get("fresh_chat"):
        st.session_state["message"] = [DEFAULT_GREETING.copy()]
    else:
        loaded = conversation_store.load(st.session_state["user_id"])
        st.session_state["message"] = loaded if loaded else [DEFAULT_GREETING.copy()]

st.caption(f"当前会话 ID：{st.session_state['user_id']}（对话与长期画像按此 ID 持久化）")

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(
            st.session_state["message"],
            user_id=st.session_state["user_id"],
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
        conversation_store.save(st.session_state["user_id"], st.session_state["message"])
        st.session_state["fresh_chat"] = False
        st.rerun()
