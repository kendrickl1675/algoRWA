"""
File: src/rwaengine/utils/portfolio_loader.py
Description: Utility to load portfolio definitions from JSON.
"""
import json
import os
from pathlib import Path
from typing import List, Dict
from loguru import logger


class PortfolioLoader:
    def __init__(self, portfolio_file: str = "portfolios/portfolios.json"):
        """
        初始化加载器。

        Args:
            portfolio_file: 资产组合文件的路径 (相对于项目根目录)
        """
        # 使用 pathlib 确保跨平台路径兼容
        self.file_path = Path(os.getcwd()) / portfolio_file
        self._cache: Dict[str, List[str]] = {}
        self._load_portfolios()

    def _load_portfolios(self):
        """内部方法：加载并验证 JSON"""
        if not self.file_path.exists():
            logger.critical(f"Portfolio file not found at: {self.file_path}")
            raise FileNotFoundError(f"Missing portfolio definition file: {self.file_path}")

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            logger.info(f"Loaded portfolio definitions from {self.file_path.name}")
        except json.JSONDecodeError as e:
            logger.critical(f"Invalid JSON format in portfolio file: {e}")
            raise ValueError("Corrupted portfolio definition file")

    def get_tickers(self, portfolio_name: str) -> List[str]:
        """
        获取指定名称的资产组合。

        Args:
            portfolio_name: JSON 中的 key (e.g., 'default', 'mag_seven')

        Returns:
            List[str]: 股票代码列表
        """
        if portfolio_name not in self._cache:
            valid_keys = list(self._cache.keys())
            logger.error(f"Portfolio '{portfolio_name}' not found. Available: {valid_keys}")
            raise KeyError(f"Unknown portfolio: {portfolio_name}")

        tickers = self._cache[portfolio_name]
        logger.info(f"Selected Portfolio '{portfolio_name}': {len(tickers)} assets")
        return tickers