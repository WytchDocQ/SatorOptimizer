from __future__ import annotations

import uvicorn

from ..observability.logging import setup_logging
from ..settings import get_settings
from ..api.app import create_app


def run() -> None:
    settings = get_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format, to_file=settings.log_to_file, file_path=settings.log_file_path)
    app = create_app(settings)
    ssl_kwargs = {}
    if getattr(settings, "enable_tls", False) and settings.tls_cert_file and settings.tls_key_file:
        ssl_kwargs = {
            "ssl_certfile": settings.tls_cert_file,
            "ssl_keyfile": settings.tls_key_file,
        }
        if settings.tls_key_password:
            ssl_kwargs["ssl_keyfile_password"] = settings.tls_key_password
        if settings.tls_ca_certs:
            ssl_kwargs["ssl_ca_certs"] = settings.tls_ca_certs
    uvicorn.run(app, host=settings.http_host, port=settings.http_port, log_level=settings.log_level, **ssl_kwargs)


if __name__ == "__main__":
    run()


