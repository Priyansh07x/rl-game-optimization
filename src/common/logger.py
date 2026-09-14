"""Structured logging utilities for the project."""

from pathlib import Path
from typing import Optional, Union
import logging


def get_logger(name: str = "cornerstone", log_dir: Optional[Union[str, Path]] = "logs", level: int = logging.INFO) -> logging.Logger:
    """Get or create a configured logger instance that outputs to stdout and a log file.

    Args:
        name: Logger name identifier.
        log_dir: Directory where log files will be saved. If None, only console output is created.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
