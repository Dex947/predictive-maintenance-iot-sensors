"""
Logging utility for predictive maintenance system.

This module sets up consistent logging across the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logger with console and file handlers.

    Args:
        name: Logger name (typically __name__)
        log_file: Path to log file. If None, only console logging
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_format: Custom log format string

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file provided)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str, config: dict = None) -> logging.Logger:
    """
    Get logger using configuration.

    Args:
        name: Logger name
        config: Configuration dictionary with logging settings

    Returns:
        Configured logger instance
    """
    if config is None:
        return setup_logger(name)

    logging_config = config.get('logging', {})

    level_str = logging_config.get('level', 'INFO')
    level = getattr(logging, level_str.upper(), logging.INFO)

    log_format = logging_config.get('format')
    log_file = logging_config.get('file')

    return setup_logger(name, log_file, level, log_format)


if __name__ == "__main__":
    # Test logger
    test_logger = setup_logger("test", "logs/test.log")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
