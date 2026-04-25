"""Structured logging setup using structlog.

Two output formats are supported:
- ``console`` (default for local dev): human-friendly colored output.
- ``json``: machine-parseable; preferred in production / Streamlit Cloud.

Every chat call binds a ``correlation_id`` to its logger so any user complaint
can be traced through every pipeline step. This is the demonstrable Task 3
(consulting standards) artifact.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: str = "INFO", format: str = "console") -> None:
    """Configure structlog processors and the stdlib logging level.

    Args:
        level: standard logging level name (DEBUG/INFO/WARNING/ERROR).
        format: ``"console"`` for dev pretty-print, ``"json"`` for prod.
    """
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to ``name``.

    Args:
        name: usually ``__name__`` of the calling module.

    Returns:
        A structlog BoundLogger ready for ``.bind(...)`` and ``.info(...)``.
    """
    return structlog.get_logger(name)
