from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header, HTTPException, Request
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from ..settings import Settings, get_settings
from ..security.api_keys import get_api_key
from ..security.idempotency import IdempotencyStore
from ..security.rate_limit import SimpleRateLimiter
from ..runtime.jobs import JobStore


_job_store: Optional[JobStore] = None
_idem_store: Optional[IdempotencyStore] = None
_limiter: Optional[SimpleRateLimiter] = None


def get_job_store(settings: Settings = Depends(get_settings)) -> JobStore:
    global _job_store
    if _job_store is None:
        _job_store = JobStore(ttl_sec=settings.job_ttl_sec, timeout_sec=settings.job_timeout_sec)
    return _job_store


def get_idempotency_store(settings: Settings = Depends(get_settings)) -> IdempotencyStore:
    global _idem_store
    if _idem_store is None:
        _idem_store = IdempotencyStore(ttl_sec=settings.job_ttl_sec)
    return _idem_store


def rate_limit(
    request: Request,
    api_key: str = Depends(get_api_key),
    settings: Settings = Depends(get_settings),
) -> str:
    global _limiter
    if _limiter is None:
        _limiter = SimpleRateLimiter(per_minute=settings.rate_limit_per_min)
    ip = request.client.host if request.client else ""
    if not _limiter.allow(api_key, ip):
        raise HTTPException(status_code=HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
    return api_key


def idempotency(
    api_key: str = Depends(get_api_key),
    idem_store: IdempotencyStore = Depends(get_idempotency_store),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> Optional[str]:
    if not idempotency_key:
        return None
    existing = idem_store.get(api_key, idempotency_key)
    return existing


