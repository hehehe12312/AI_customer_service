import os


def apply_streamlit_secrets() -> None:
    """把 Streamlit Secrets 写入环境变量，供 os.getenv 使用。"""
    try:
        import streamlit as st

        secrets = st.secrets
    except Exception:
        return

    def _set(key: str, value: object) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text:
            os.environ[key] = text

    def _walk(obj) -> None:
        try:
            keys = list(obj.keys())
        except Exception:
            return
        for key in keys:
            try:
                value = obj[key]
            except Exception:
                continue
            key_str = str(key)
            # 嵌套表（TOML section）继续展开
            if hasattr(value, "keys") and not isinstance(value, (str, bytes)):
                _walk(value)
            else:
                _set(key_str, value)

    try:
        _walk(secrets)
    except Exception:
        return

    # 兼容部分 langchain-openai 回退读取 OPENAI_API_KEY
    if os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
