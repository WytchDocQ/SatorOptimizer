from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Depends, HTTPException, Header
from starlette.status import HTTP_401_UNAUTHORIZED

from ..settings import Settings, get_settings


def _constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def get_api_key(
    settings: Settings = Depends(get_settings),
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> str:
    provided: Optional[str] = None
    if x_api_key:
        provided = x_api_key.strip()
    elif authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()

    # Require configured single API key
    if not settings.api_key:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Missing API key")

    if provided and _constant_time_equals(provided, settings.api_key):
        return provided

    raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid API key")


