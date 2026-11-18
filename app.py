import os
import time

import streamlit as st

# Streamlit Cloud：把 Secrets 注入环境变量，供 factory / 和风天气读取
try:
    for key, value in st.secrets.items():
        if isinstance(value, str) and value and key not in os.environ:
            os.environ[key] = value
except Exception:
    pass

from agent.react_agent import ReactAgent
from memory.conversation_store import ConversationStore

DEFAULT_GREETING = {
    "role": "assistant",
    "content": "你好，我是智扫通机器人智能客服，请问有什么可以帮助你？",
}
USER_IDS = [str(i) for i in range(1001, 1011)]

st.title("智能客服")
st.divider()

conversation_store = ConversationStore()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "user_id" not in st.session_state:
    st.session_state["user_id"] = USER_IDS[0]

with st.sidebar:
    st.subheader("用户记忆")
    selected_user = st.selectbox(
        "选择用户 ID",
        USER_IDS,
        index=USER_IDS.index(st.session_state["user_id"]),
        help="同一用户的对话与画像会跨刷新保留",
    )
    if selected_user != st.session_state["user_id"]:
        st.session_state["user_id"] = selected_user
        loaded = conversation_store.load(selected_user)
        st.session_state["message"] = loaded if loaded else [DEFAULT_GREETING.copy()]
        st.rerun()

if "message" not in st.session_state:
    loaded = conversation_store.load(st.session_state["user_id"])
    st.session_state["message"] = loaded if loaded else [DEFAULT_GREETING.copy()]

st.caption(f"当前用户：{st.session_state['user_id']}（对话与长期画像按此 ID 持久化）")

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
        st.rerun()
