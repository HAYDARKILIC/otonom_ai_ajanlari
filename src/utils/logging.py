"""İçe aktarma anında bir kez yapılandırılan yapılandırılmış loglama."""

from __future__ import annotations

import logging
import os
import sys

import structlog


def _configure() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    fmt = os.environ.get("LOG_FORMAT", "console")

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level, logging.INFO),
    )

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if fmt == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        # Loglar stderr'e gider: MCP stdio aktarımında stdout yalnızca JSON-RPC
        # çerçevelerine ayrılmıştır; oraya yazılan tek bir log satırı protokolü bozar.
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level, logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


_configure()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Çağıran modüle bağlı bir structlog logger'ı döndürür."""
    return structlog.get_logger(name)
