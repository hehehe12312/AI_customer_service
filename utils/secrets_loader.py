import os


REQUIRED_SECRET_KEYS = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "VOYAGE_API_KEY",
)


def apply_streamlit_secrets() -> list[str]:
    """把 Streamlit Secrets 写入环境变量，返回已读到的 key 名列表。"""
    loaded: list[str] = []
    try:
        import streamlit as st

        secrets = st.secrets
    except Exception:
        return loaded

    def _set(key: str, value: object) -> None:
        if value is None:
            return
        text = str(value).strip()
        if not text:
            return
        os.environ[key] = text
        loaded.append(key)

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
            if hasattr(value, "keys") and not isinstance(value, (str, bytes)):
                _walk(value)
            else:
                _set(key_str, value)

    try:
        _walk(secrets)
    except Exception:
        return loaded

    if os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]

    return loaded


def missing_required_secrets() -> list[str]:
    apply_streamlit_secrets()
    return [key for key in REQUIRED_SECRET_KEYS if not os.getenv(key)]
