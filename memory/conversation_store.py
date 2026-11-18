import json
import os
from typing import Any

from utils.config_handler import memory_conf
from utils.path_tools import get_abs_path


class ConversationStore:
    """按 user_id 持久化对话消息（仅 role + content）。"""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = get_abs_path(base_dir or memory_conf["conversation_dir"])
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, user_id: str) -> str:
        safe_id = "".join(c for c in str(user_id) if c.isalnum() or c in ("-", "_"))
        return os.path.join(self.base_dir, f"{safe_id}.json")

    def load(self, user_id: str) -> list[dict[str, str]]:
        path = self._path(user_id)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            messages = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                content = item.get("content")
                if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content})
            return messages
        except (json.JSONDecodeError, OSError):
            return []

    def save(self, user_id: str, messages: list[dict[str, Any]]) -> None:
        cleaned = []
        for item in messages:
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                cleaned.append({"role": role, "content": content})
        path = self._path(user_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)

    def trim(self, messages: list[dict[str, Any]], max_messages: int | None = None) -> list[dict[str, Any]]:
        limit = max_messages if max_messages is not None else int(memory_conf.get("max_history_messages", 20))
        if limit <= 0 or len(messages) <= limit:
            return messages
        return messages[-limit:]
