"""Logging configuration for VLMRun SDK."""

import os
import sys
from loguru import logger

# Configure logger with WARNING level by default
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level=os.getenv("VLMRUN_LOGGING_LEVEL", "WARNING"),
)

__all__ = ["logger"]
