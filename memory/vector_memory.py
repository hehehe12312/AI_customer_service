import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document

from model.factory import embed_model
from utils.config_handler import memory_conf
from utils.path_tools import get_abs_path


class VectorMemory:
    """用户自由文本长期记忆，独立于知识库 Chroma 集合。"""

    def __init__(self):
        self.vector_store = Chroma(
            collection_name=memory_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(memory_conf["persist_directory"]),
        )
        self.k = int(memory_conf.get("recall_k", 3))

    def add(self, user_id: str, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "记忆内容为空，未写入"
        doc_id = f"{user_id}_{uuid.uuid4().hex[:12]}"
        self.vector_store.add_documents(
            [Document(page_content=text, metadata={"user_id": str(user_id)})],
            ids=[doc_id],
        )
        return f"已写入长期记忆: {text}"

    def search(self, user_id: str, query: str, k: int | None = None) -> list[str]:
        query = (query or "").strip() or "用户偏好"
        top_k = k if k is not None else self.k
        try:
            docs = self.vector_store.similarity_search(
                query,
                k=top_k,
                filter={"user_id": str(user_id)},
            )
        except Exception:
            # 部分 chroma 版本 filter 语法差异时退化为全量检索再过滤
            docs = self.vector_store.similarity_search(query, k=max(top_k * 5, 10))
            docs = [d for d in docs if str(d.metadata.get("user_id")) == str(user_id)][:top_k]
        return [d.page_content for d in docs if d.page_content]