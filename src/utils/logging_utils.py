"""
Logging utilities — dual output to console + log file.
"""

import logging
import os
import sys

from .config import Config


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger that writes to both console and a log file.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    Config.ensure_dirs()
    log_path = os.path.join(Config.LOGS_DIR, "pipeline.log")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
