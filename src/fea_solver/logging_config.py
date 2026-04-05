"""Structured logging configuration for the FEA solver.

Creates per-case log files in the logs/ directory with timestamped records.
Console output is limited to WARNING and above to avoid cluttering solver output.
File output captures DEBUG-level traces for diagnostics.
"""
from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: Path, case_label: str) -> logging.Logger:
    """Configure logging for a solver run.

    Args:
        log_dir: Directory where log files will be written (created if absent).
        case_label: Identifier used as the log filename stem.

    Returns:
        The 'fea_solver' logger, ready to use.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{case_label}.log"

    # Get (or create) the fea_solver namespace logger
    logger = logging.getLogger("fea_solver")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers on repeated calls (e.g. in tests)
    if logger.handlers:
        logger.handlers.clear()

    # File handler — DEBUG and above
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Console handler — WARNING and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Logging initialized for case: %s", case_label)
    return logger
