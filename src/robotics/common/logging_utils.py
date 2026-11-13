"""Logging helpers shared by scripts and library code."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str | int = "INFO") -> None:
    """Configure root logging once and silence noisy third-party loggers."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S", force=True)
    # robosuite prints macro-file warnings on import; they are irrelevant for this project.
    for noisy in ("matplotlib", "PIL", "numba"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
    # robosuite logs every controller-config load at INFO; keep only its warnings.
    logging.getLogger("robosuite_logs").setLevel(max(level, logging.WARNING))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
