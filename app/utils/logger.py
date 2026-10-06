"""Logging setup shared by the pipeline, the agent and the mock CMS."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party libraries are noisy at INFO; keep them at WARNING.
_QUIET_LOGGERS = ("httpx", "httpcore", "urllib3", "asyncio", "uvicorn.access")


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure root logging with console + optional file handler.

    Safe to call multiple times (handlers are replaced, not duplicated).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("app")


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (``app.<name>``)."""
    if name.startswith("app"):
        return logging.getLogger(name)
    return logging.getLogger(f"app.{name}")
