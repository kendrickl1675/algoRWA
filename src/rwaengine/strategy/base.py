"""
File: src/rwaengine/strategy/base.py
Description: Interface for View Generators (Strategy Pattern).
"""
from abc import ABC, abstractmethod
from typing import List
import pandas as pd

from src.rwaengine.strategy.types import InvestorView


class ViewGenerator(ABC):
    """
    Abstract Base Class for generating Investor Views (P and Q matrices).
    """

    @abstractmethod
    def generate_views(self, current_prices: pd.Series) -> List[InvestorView]:
        """
        基于当前市场状态生成观点。

        Args:
            current_prices: 最新资产价格 (用于参考或计算隐含市值)

        Returns:
            List[InvestorView]: 标准化的观点列表
        """
        pass