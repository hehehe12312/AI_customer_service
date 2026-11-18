import os
from abc import ABC, abstractmethod
from typing import Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.config_handler import rag_conf
from utils.secrets_loader import apply_streamlit_secrets

load_dotenv()
apply_streamlit_secrets()


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    """聊天模型：DeepSeek OpenAI 兼容接口。"""

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if not api_key:
            raise ValueError(
                "缺少 DEEPSEEK_API_KEY。本地请配置 .env；"
                "Streamlit Cloud 请到 App settings → Secrets 添加该密钥。"
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
        api_key = os.getenv("SILICONFLOW_API_KEY")
        base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        if not api_key:
            raise ValueError(
                "缺少 SILICONFLOW_API_KEY。本地请配置 .env；"
                "Streamlit Cloud 请到 App settings → Secrets 添加该密钥。"
            )
        return OpenAIEmbeddings(
            model=rag_conf["embedding_model_name"],
            api_key=api_key,
            base_url=base_url,
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
