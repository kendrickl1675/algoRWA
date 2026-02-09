"""
YFinance Market Data Adapter.

Fetches historical OHLCV data via the yfinance library and normalizes it
into the internal schema expected by the RWA engine.  Handles quirks across
different yfinance versions (column ordering, MultiIndex layouts, renamed
'Adj Close' variants) so downstream consumers always receive a consistent
long-format DataFrame.
"""
from datetime import date
from typing import List, Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from src.rwaengine.data.base import MarketDataProvider


class YFinanceAdapter(MarketDataProvider):
    """Concrete MarketDataProvider backed by Yahoo Finance."""

    def __init__(self, proxy: Optional[str] = None):
        """
        Args:
            proxy: Optional HTTP/SOCKS proxy URL for regions with restricted
                   access to Yahoo Finance (e.g. mainland China, Macau).
        """
        self.proxy = proxy

    def fetch_history(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Download historical prices and return a standardized DataFrame.

        Args:
            tickers: List of Yahoo Finance ticker symbols.
            start_date: First calendar date of the requested window.
            end_date: Last calendar date (exclusive in yfinance).

        Returns:
            A long-format DataFrame with columns defined by the internal
            MarketData schema, or an empty DataFrame when no data is available.

        Raises:
            Exception: Re-raised after logging if the download fails
                       unexpectedly (fail-fast during development).
        """
        logger.info(
            f"Fetching {len(tickers)} assets ({start_date} to {end_date}) "
            f"| Proxy: {self.proxy}"
        )

        try:
            # yfinance treats end_date as exclusive; identical dates yield no rows.
            if start_date == end_date:
                logger.warning(
                    f"Start date equals end date ({start_date}). "
                    "Returning empty DataFrame."
                )
                return pd.DataFrame()

            df = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=False,   # Keep raw Close AND Adj Close (both needed for RWA NAV vs strategy returns)
                actions=False,       # Exclude dividend / split events for now
                progress=False,
                threads=True,
                group_by="ticker",   # Request (Ticker, Price) MultiIndex for multi-asset downloads
            )

            # Guard against empty responses (weekends, holidays, delisted tickers).
            if df.empty:
                logger.warning(
                    f"No data returned for {tickers}. "
                    "Possible holiday range or delisted symbols."
                )
                return pd.DataFrame()

            df_clean = self._standardize_columns(df)

            # Post-cleaning guard: the standardization logic may legitimately
            # discard all rows if critical columns are absent.
            if df_clean.empty:
                logger.warning("Data became empty after column standardization.")
                return pd.DataFrame()

            # Final schema validation defined in the base class.
            self.validate_schema(df_clean)

            logger.success(f"Fetched {len(df_clean)} rows successfully.")
            return df_clean

        except Exception as e:
            logger.error(f"YFinance Critical Failure: {e}")
            raise

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw yfinance output into the internal long-format schema.

        Handles two known MultiIndex layouts that vary across yfinance
        versions:
          - (Ticker, Price) — produced by ``group_by='ticker'``.
          - (Price, Ticker) — legacy default in older versions.

        For single-ticker downloads the DataFrame has a flat column index
        and no ticker information; a placeholder is inserted so the schema
        remains uniform.

        Returns:
            A flat DataFrame with lowercase column names mapped to the
            internal naming convention, or an empty DataFrame if required
            price columns are missing.
        """
        data = df.copy()

        # Protect against zombie DataFrames that have an index but no columns.
        if len(data.columns) == 0:
            return pd.DataFrame()

