
"""
Centralised Loguru configuration.

Sets up two logging sinks with different verbosity levels:
  - **stderr** (terminal): INFO and above, compact timestamp format, coloured.
  - **File**: DEBUG and above, full timestamps with source location, daily
    rotation with automatic compression and 30-day retention.

Call ``setup_logger()`` once at application startup (before any other
``logger`` usage) to activate both sinks.
"""
import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_dir: str = "logs") -> logger:
    """Configure and return the global Loguru logger.

    Args:
        log_dir: Directory for rotated log files.  Created automatically
                 if it does not exist.

    Returns:
        The configured ``logger`` instance (same singleton used everywhere
        via ``from loguru import logger``).
    """
    # Clear any default handlers so we don't get duplicate output.
    logger.remove()

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Daily log file: e.g. logs/rwa_engine_2026-02-08.log
    log_file = log_path / "rwa_engine_{time:YYYY-MM-DD}.log"

    # Terminal sink — concise, coloured output for interactive use.
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File sink — verbose, includes source location for post-mortem debugging.
    logger.add(
        log_file,
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
        enqueue=True,
        encoding="utf-8",
    )

    return logger