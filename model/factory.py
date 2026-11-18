import os
import time
from abc import ABC, abstractmethod
from typing import Optional

import voyageai
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
# 硅基流动嵌入（暂不使用，改用 Voyage）
# from langchain_openai import OpenAIEmbeddings

from utils.config_handler import rag_conf
from utils.logger_handler import logger
from utils.secrets_loader import apply_streamlit_secrets

_chat_model = None
_embed_model = None


def _prepare_env() -> None:
    load_dotenv()
    apply_streamlit_secrets()


class VoyageEmbeddings(Embeddings):
    """基于 voyageai 官方 SDK 的嵌入，避免拉取 HuggingFace tokenizer。"""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = voyageai.Client(api_key=api_key)

    def _embed_with_retry(self, texts: list[str], input_type: str):
        """免费档约 3 RPM，遇限流自动等待重试。"""
        last_err = None
        for attempt in range(6):
            try:
                return self.client.embed(texts, model=self.model, input_type=input_type)
            except Exception as e:
                last_err = e
                name = type(e).__name__
                msg = str(e).lower()
                if "ratelimit" in name.lower() or "rate limit" in msg or "rpm" in msg:
                    wait = 22 + attempt * 5
                    logger.warning(f"[Voyage]限流，{wait}s 后重试 ({attempt + 1}/6): {e}")
                    time.sleep(wait)
                    continue
                raise
        raise last_err

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Voyage 单次批量有上限，简单按 64 条切分
        vectors: list[list[float]] = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self._embed_with_retry(batch, input_type="document")
            vectors.extend(result.embeddings)
            # 批次之间稍作间隔，降低免费档触发限流概率
            if i + batch_size < len(texts):
                time.sleep(21)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        result = self._embed_with_retry([text], input_type="query")
        return result.embeddings[0]


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
    """文档嵌入：当前使用 Voyage AI；硅基流动实现保留在注释中。"""

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        _prepare_env()

        # --- 当前：Voyage AI ---
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "缺少 VOYAGE_API_KEY。"
                "请到 Streamlit Cloud → Manage app → Settings → Secrets 配置。"
            )
        return VoyageEmbeddings(
            model=rag_conf["embedding_model_name"],
            api_key=api_key,
        )

        # --- 硅基流动（暂不启用）---
        # api_key = os.getenv("SILICONFLOW_API_KEY")
        # base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        # if not api_key:
        #     raise ValueError(
        #         "缺少 SILICONFLOW_API_KEY。"
        #         "请到 Streamlit Cloud → Manage app → Settings → Secrets 配置。"
        #     )
        # return OpenAIEmbeddings(
        #     model=rag_conf["embedding_model_name"],  # 例如 BAAI/bge-m3
        #     api_key=api_key,
        #     base_url=base_url,
        # )


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
