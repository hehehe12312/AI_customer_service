import os


def apply_streamlit_secrets() -> None:
    """把 Streamlit Secrets 写入环境变量，供 dotenv/os.getenv 使用。"""
    try:
        import streamlit as st

        secrets = st.secrets
    except Exception:
        return

    def _walk(obj, prefix: str = "") -> None:
        try:
            items = obj.items()
        except Exception:
            return
        for key, value in items:
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"
            if isinstance(value, str):
                if value:
                    os.environ[str(key)] = value
            elif hasattr(value, "items"):
                _walk(value, str(key))

    _walk(secrets)

    # langchain-openai 在部分版本仍会回退读 OPENAI_API_KEY
    if os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
    if os.getenv("SILICONFLOW_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        # 仅当完全没有 chat key 时不覆盖；上面已处理 DeepSeek
        pass
