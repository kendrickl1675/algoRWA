"""
File: src/rwaengine/data/base.py
Description: Abstract Base Class for all data providers.
"""
from abc import ABC, abstractmethod
from datetime import date
from typing import List

import pandas as pd
from loguru import logger

# Correct Absolute Import based on your src-layout
from src.rwaengine.data.schemas import MarketData


class MarketDataProvider(ABC):
    """
    Abstract Base Class that enforces a strict interface for data fetching.
    """

    @abstractmethod
    def fetch_history(
            self,
            tickers: List[str],
            start_date: date,
            end_date: date
    ) -> pd.DataFrame:
        """
        Fetch historical market data.

        Returns:
            pd.DataFrame: A standardized DataFrame.
                          Must contain columns: [open, high, low, close, volume]
                          Index: MultiIndex (date, ticker) or simple Index.
        """
        pass


    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Defense-in-Depth:
        Even if the adapter code runs, we verify the output DataFrame structure
        before passing it to the Core Engine.
        """
        required_cols = {
            'open_price',
            'high_price',
            'low_price',
            'close_price',
            'volume'
        }
        # Normalize columns to lowercase for check
        df_cols = set(c.lower() for c in df.columns)

        if not required_cols.issubset(df_cols):
            missing = required_cols - df_cols
            logger.critical(f"Data Schema Violation! Missing columns: {missing}")
            # 打印当前拥有的列，方便调试
            logger.debug(f"Current Columns: {df_cols}")
            raise ValueError(f"DataFrame violates MarketData schema. Missing: {missing}")

        return True