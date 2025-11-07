from __future__ import annotations

from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.status import HTTP_403_FORBIDDEN

from ..settings import Settings


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else ""


class IPFilterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next: Callable):
        ip = _client_ip(request)
        if self.settings.ip_blacklist and ip in self.settings.ip_blacklist:
            return JSONResponse({"detail": "IP blocked"}, status_code=HTTP_403_FORBIDDEN)
        if self.settings.ip_whitelist and ip not in self.settings.ip_whitelist:
            return JSONResponse({"detail": "IP not allowed"}, status_code=HTTP_403_FORBIDDEN)
        return await call_next(request)


