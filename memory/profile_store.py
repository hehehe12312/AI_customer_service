import json
import os
from typing import Any

from utils.config_handler import memory_conf
from utils.path_tools import get_abs_path


class ProfileStore:
    """按 user_id 存储结构化用户画像（城市、机型、偏好等）。"""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = get_abs_path(base_dir or memory_conf["profile_dir"])
        os.makedirs(self.base_dir, exist_ok=True)

    def _path(self, user_id: str) -> str:
        safe_id = "".join(c for c in str(user_id) if c.isalnum() or c in ("-", "_"))
        return os.path.join(self.base_dir, f"{safe_id}.json")

    def get(self, user_id: str) -> dict[str, Any]:
        path = self._path(user_id)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def update(self, user_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        profile = self.get(user_id)
        for key, value in fields.items():
            if value is None:
                continue
            if key == "notes":
                notes = profile.get("notes")
                if not isinstance(notes, list):
                    notes = []
                notes.append(str(value))
                profile["notes"] = notes[-20:]
            else:
                profile[key] = value
        path = self._path(user_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        return profile

    def summary(self, user_id: str) -> str:
        profile = self.get(user_id)
        if not profile:
            return "暂无该用户的长期画像。"
        lines = [f"用户ID: {user_id}"]
        for key, value in profile.items():
            if key == "notes" and isinstance(value, list):
                if value:
                    lines.append("备注: " + "；".join(str(v) for v in value[-5:]))
                continue
            lines.append(f"{key}: {value}")
        return "\n".join(lines)
