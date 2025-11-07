from __future__ import annotations

import time
from typing import Dict, Optional, Tuple


class IdempotencyStore:
    def __init__(self, ttl_sec: int = 600) -> None:
        self.ttl_sec = ttl_sec
        self._store: Dict[Tuple[str, str], Tuple[float, str]] = {}

    def put(self, api_key: str, idem_key: str, job_id: str) -> None:
        self._store[(api_key, idem_key)] = (time.time(), job_id)

    def get(self, api_key: str, idem_key: str) -> Optional[str]:
        key = (api_key, idem_key)
        item = self._store.get(key)
        if not item:
            return None
        ts, job_id = item
        if time.time() - ts > self.ttl_sec:
            del self._store[key]
            return None
        return job_id


