from __future__ import annotations

import logging
from typing import Optional

import structlog


def setup_logging(level: str = "info", fmt: str = "human", to_file: bool = False, file_path: Optional[str] = None) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)

    processors = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
    ]
    if fmt == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )

    handlers = []
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    handlers.append(stream_handler)

    if to_file and file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    logging.basicConfig(level=log_level, handlers=handlers)


