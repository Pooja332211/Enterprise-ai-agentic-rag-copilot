import logging
import sys
from typing import Optional
from backend.app.config.settings import get_settings


def setup_logger(name: Optional[str] = "copilot") -> logging.Logger:
    """Configures and returns a structured logger for the application."""
    settings = get_settings()
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if re-called
    if logger.handlers:
        return logger

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Format with ISO timestamp, level, module, function name, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(module)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


logger = setup_logger()
