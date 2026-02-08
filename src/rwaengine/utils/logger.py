"""
File: src/rwaengine/utils/logger.py
Description: Centralized logging configuration.
Separates concerns: Concise CLI output vs. Detailed File logging.
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_dir: str = "logs"):
    """
    配置 Loguru：
    1. Terminal: 仅显示 INFO 及以上，格式简洁。
    2. File: 显示 DEBUG 所有细节，包含行号，自动轮转。
    """
    logger.remove()

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 定义日志文件路径: logs/rwa_engine_2026-02-08.log
    log_file = log_path / "rwa_engine_{time:YYYY-MM-DD}.log"

    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True
    )

    logger.add(
        log_file,
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,
        encoding="utf-8"
    )

    return logger