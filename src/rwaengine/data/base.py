
"""
Abstract base class for market data providers.

Every concrete data adapter (e.g. YFinance, Bloomberg, on-chain oracle)
must implement the `fetch_history` interface defined here.  The base class
also provides a shared `validate_schema` method that acts as a final
safety net before data reaches the core engine.
"""
from abc import ABC, abstractmethod
from datetime import date
from typing import List

import pandas as pd
from loguru import logger

from src.rwaengine.data.schemas import MarketData


class MarketDataProvider(ABC):
    """Contract that all market data adapters must satisfy."""

    @abstractmethod
    def fetch_history(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for the given tickers and date range.

        Args:
            tickers: Yahoo-style ticker symbols (e.g. ``["AAPL", "MSFT"]``).
            start_date: First calendar date of the window (inclusive).
            end_date: Last calendar date of the window (exclusive in most providers).

        Returns:
            A standardized DataFrame whose columns include at minimum:
            ``open_price``, ``high_price``, ``low_price``, ``close_price``,
            ``volume``, plus ``trade_date`` and ``ticker`` identifiers.
        """

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Verify that the DataFrame contains every column the engine expects.

        This is a defence-in-depth check: even if the adapter's own
        transformation logic ran without errors, we still confirm the output
        schema before handing data to downstream consumers.

        Args:
            df: The adapter-produced DataFrame to validate.

        Returns:
            ``True`` if validation passes.

        Raises:
            ValueError: If one or more required columns are missing.
        """
        required_cols = {
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
        }

        # Lowercase comparison so mixed-case columns don't slip through.
        df_cols = {c.lower() for c in df.columns}

        if not required_cols.issubset(df_cols):
            missing = required_cols - df_cols
            logger.critical(f"Data schema violation! Missing columns: {missing}")
            logger.debug(f"Columns present: {sorted(df_cols)}")
            raise ValueError(
                f"DataFrame violates MarketData schema. Missing: {missing}"
            )

        return True