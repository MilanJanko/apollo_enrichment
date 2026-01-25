from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


NOT_FOUND = "__NOT_FOUND__"


class OrgSearchCache:
    """
    Simple file-backed cache for Apollo org lookups.
    Key: domain
    Value: org dict OR NOT_FOUND sentinel

    Stored at: data/cache/org_cache.json
    """

    def __init__(self, path: str = "data/cache/org_cache.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._cache: Dict[str, Any] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f) or {}
            except Exception:
                self._cache = {}

    def get(self, domain: str) -> Optional[dict]:
        v = self._cache.get(domain)
        if v is None:
            return None  # not cached
        if v == NOT_FOUND:
            return {}  # cached as "not found"
        return v

    def set_found(self, domain: str, org: dict) -> None:
        self._cache[domain] = org
        self._dirty = True

    def set_not_found(self, domain: str) -> None:
        self._cache[domain] = NOT_FOUND
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        tmp = f"{self.path}.tmp_{int(time.time())}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False)
        os.replace(tmp, self.path)
        self._dirty = False
