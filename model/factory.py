import os
from abc import ABC, abstractmethod
from typing import Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.config_handler import rag_conf
from utils.secrets_loader import apply_streamlit_secrets

_chat_model = None
_embed_model = None


def _prepare_env() -> None:
    load_dotenv()
    apply_streamlit_secrets()


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    """聊天模型：DeepSeek OpenAI 兼容接口。"""

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        _prepare_env()
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if not api_key:
            raise ValueError(
                "缺少 DEEPSEEK_API_KEY。"
                "请到 Streamlit Cloud → Manage app → Settings → Secrets，"
                "用 TOML 格式配置，例如：DEEPSEEK_API_KEY = \"你的key\""
            )
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
        )


class EmbeddingsFactory(BaseModelFactory):
    """文档嵌入：硅基流动 OpenAI 兼容接口。"""

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        _prepare_env()
        api_key = os.getenv("SILICONFLOW_API_KEY")
        base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        if not api_key:
            raise ValueError(
                "缺少 SILICONFLOW_API_KEY。"
                "请到 Streamlit Cloud → Manage app → Settings → Secrets 配置。"
            )
        return OpenAIEmbeddings(
            model=rag_conf["embedding_model_name"],
            api_key=api_key,
            base_url=base_url,
        )


def get_chat_model():
    global _chat_model
    if _chat_model is None:
        _chat_model = ChatModelFactory().generator()
    return _chat_model


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = EmbeddingsFactory().generator()
    return _embed_model


def __getattr__(name: str):
    # 兼容 `from model.factory import chat_model`
    if name == "chat_model":
        return get_chat_model()
    if name == "embed_model":
        return get_embed_model()
    raise AttributeError(f"module 'model.factory' has no attribute {name!r}")
