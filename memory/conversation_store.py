import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from utils.config_handler import memory_conf
from utils.path_tools import get_abs_path

DEFAULT_TITLE = "新对话"


class ConversationStore:
    """按 session_id 持久化对话（含标题与更新时间）。"""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = get_abs_path(base_dir or memory_conf["conversation_dir"])
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, session_id: str) -> str:
        safe_id = "".join(c for c in str(session_id) if c.isalnum() or c in ("-", "_"))
        return os.path.join(self.base_dir, f"{safe_id}.json")

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _title_from_messages(messages: list[dict[str, Any]]) -> str:
        for item in messages:
            if item.get("role") == "user" and isinstance(item.get("content"), str):
                text = item["content"].strip().replace("\n", " ")
                if text:
                    return text[:20] + ("…" if len(text) > 20 else "")
        return DEFAULT_TITLE

    @staticmethod
    def _has_user_message(messages: list[dict[str, Any]]) -> bool:
        return any(
            item.get("role") == "user" and isinstance(item.get("content"), str) and item["content"].strip()
            for item in messages
        )

    def _normalize(self, session_id: str, data: Any) -> dict[str, Any]:
        """兼容旧版纯 messages 数组。"""
        if isinstance(data, list):
            messages = self._clean_messages(data)
            return {
                "session_id": session_id,
                "title": self._title_from_messages(messages),
                "updated_at": self._now_iso(),
                "messages": messages,
            }
        if not isinstance(data, dict):
            return {
                "session_id": session_id,
                "title": DEFAULT_TITLE,
                "updated_at": self._now_iso(),
                "messages": [],
            }
        messages = self._clean_messages(data.get("messages") or [])
        title = data.get("title") or self._title_from_messages(messages)
        return {
            "session_id": data.get("session_id") or session_id,
            "title": title if isinstance(title, str) and title.strip() else DEFAULT_TITLE,
            "updated_at": data.get("updated_at") or self._now_iso(),
            "messages": messages,
        }

    @staticmethod
    def _clean_messages(messages: list[Any]) -> list[dict[str, str]]:
        cleaned = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                cleaned.append({"role": role, "content": content})
        return cleaned

    def _read_raw(self, session_id: str) -> dict[str, Any] | None:
        path = self._path(session_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._normalize(session_id, data)
        except (json.JSONDecodeError, OSError):
            return None

    def _write(self, payload: dict[str, Any]) -> None:
        session_id = payload["session_id"]
        path = self._path(session_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def create_session(self, greeting: dict[str, str] | None = None) -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        messages = []
        if greeting and greeting.get("role") and greeting.get("content"):
            messages = [{"role": greeting["role"], "content": greeting["content"]}]
        payload = {
            "session_id": session_id,
            "title": DEFAULT_TITLE,
            "updated_at": self._now_iso(),
            "messages": messages,
        }
        self._write(payload)
        return payload

    def load_session(self, session_id: str) -> list[dict[str, str]]:
        data = self._read_raw(session_id)
        if not data:
            return []
        return data["messages"]

    def save_session(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        title: str | None = None,
    ) -> dict[str, Any]:
        cleaned = self._clean_messages(messages)
        existing = self._read_raw(session_id)
        auto_title = self._title_from_messages(cleaned)

        if title and title.strip():
            final_title = title.strip()
        elif existing and existing.get("title") and existing["title"] != DEFAULT_TITLE:
            final_title = existing["title"]
        else:
            final_title = auto_title

        payload = {
            "session_id": session_id,
            "title": final_title,
            "updated_at": self._now_iso(),
            "messages": cleaned,
        }
        self._write(payload)
        return payload

    def delete_session(self, session_id: str) -> bool:
        """删除指定会话文件；成功或不存在均视为完成时返回 True/False。"""
        path = self._path(session_id)
        if not os.path.exists(path):
            return False
        try:
            os.remove(path)
            return True
        except OSError:
            return False

    def list_sessions(self) -> list[dict[str, str]]:
        """返回有用户消息的会话，按更新时间倒序。"""
        if not os.path.isdir(self.base_dir):
            return []
        sessions = []
        for name in os.listdir(self.base_dir):
            if not name.endswith(".json"):
                continue
            session_id = name[:-5]
            data = self._read_raw(session_id)
            if not data:
                continue
            if not self._has_user_message(data["messages"]):
                continue
            sessions.append(
                {
                    "session_id": data["session_id"],
                    "title": data.get("title") or DEFAULT_TITLE,
                    "updated_at": data.get("updated_at") or "",
                }
            )
        sessions.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
        return [{"session_id": s["session_id"], "title": s["title"]} for s in sessions]

    # --- 兼容旧接口（如仍有调用）---
    def load(self, user_id: str) -> list[dict[str, str]]:
        return self.load_session(user_id)

    def save(self, user_id: str, messages: list[dict[str, Any]]) -> None:
        self.save_session(user_id, messages)

    def list_user_ids(self) -> list[str]:
        return [s["session_id"] for s in self.list_sessions()]

    def trim(self, messages: list[dict[str, Any]], max_messages: int | None = None) -> list[dict[str, Any]]:
        limit = max_messages if max_messages is not None else int(memory_conf.get("max_history_messages", 20))
        if limit <= 0 or len(messages) <= limit:
            return messages
        return messages[-limit:]
