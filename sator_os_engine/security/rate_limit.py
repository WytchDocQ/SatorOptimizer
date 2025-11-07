from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple


class SimpleRateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._events: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)  # (api_key, ip) -> timestamps

    def allow(self, api_key: str, ip: str) -> bool:
        now = time.time()
        window_start = now - 60.0
        dq = self._events[(api_key, ip)]
        while dq and dq[0] < window_start:
            dq.popleft()
        if len(dq) >= self.per_minute:
            return False
        dq.append(now)
        return True


