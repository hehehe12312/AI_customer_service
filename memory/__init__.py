# 避免 import memory 时立刻拉起 embedding 模型（Cloud 上密钥尚未注入会失败）
from memory.conversation_store import ConversationStore
from memory.profile_store import ProfileStore

__all__ = ["ConversationStore", "ProfileStore", "VectorMemory"]


def __getattr__(name: str):
    if name == "VectorMemory":
        from memory.vector_memory import VectorMemory

        return VectorMemory
    raise AttributeError(name)
