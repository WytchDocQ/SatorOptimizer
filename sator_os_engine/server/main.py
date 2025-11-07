from __future__ import annotations

import uvicorn

from ..observability.logging import setup_logging
from ..settings import get_settings
from ..api.app import create_app


def run() -> None:
    settings = get_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format, to_file=settings.log_to_file, file_path=settings.log_file_path)
    app = create_app(settings)
    uvicorn.run(app, host=settings.http_host, port=settings.http_port, log_level=settings.log_level)


if __name__ == "__main__":
    run()


