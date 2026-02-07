"""
File: src/data/adapters/yfinance_adapter.py
Description: Robust YFinance adapter handling version discrepancies.
"""
from datetime import date
from typing import List, Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from src.rwa_engine.data.base import MarketDataProvider

class YFinanceAdapter(MarketDataProvider):
    def __init__(self, proxy: Optional[str] = None):
        """
        Args:
            proxy: Proxy URL for restricted regions (e.g. Macau/CN).
        """
        self.proxy = proxy

    def fetch_history(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:

        logger.info(f"Fetching {len(tickers)} assets | Proxy: {self.proxy}")

        try:
            # [CRITICAL] auto_adjust=False is mandatory for RWA.
            # We need Raw Close for NAV (Wallet Balance)
            # We need Adj Close for Strategy (Performance)
            df = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                proxy=self.proxy,
                auto_adjust=False,  # FORCE Raw Data + Adj Close
                actions=False,      # We don't need dividends detail yet
                progress=False,
                threads=True,
                group_by='ticker'   # Ensure consistent MultiIndex structure
            )

            if df.empty:
                logger.warning(f"No data returned for {tickers}")
                return pd.DataFrame()

            # Clean and Standardize
            df_clean = self._standardize_columns(df)

            # Validation (Will raise error if critical columns missing)
            self.validate_schema(df_clean)

            logger.success(f"Fetched {len(df_clean)} rows successfully.")
            return df_clean

        except Exception as e:
            logger.error(f"YFinance Critical Failure: {str(e)}")
            raise

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Robust column mapper handling yfinance version chaos.
        """
        data = df.copy()

        # 1. Handle Multi-Ticker format (Ticker as Level 0 or Level 1)
        # yfinance behavior varies: sometimes (Price, Ticker), sometimes (Ticker, Price)
        # We normalize to a flat DataFrame with 'ticker' column.

        if isinstance(data.columns, pd.MultiIndex):
            # Check if 'Ticker' is in columns levels
            # Standard yf.download(group_by='ticker') returns (Ticker, Price)
            # We want to stack the Price level to get long format

            # Let's try to infer structure
            example_col = data.columns[0]
            if example_col[0] in ['Open', 'High', 'Low', 'Close', 'Volume']:
                # Format: (Price, Ticker) -> Old Default
                data = data.stack(level=1, future_stack=True)
            else:
                # Format: (Ticker, Price) -> group_by='ticker' Default
                data = data.stack(level=0, future_stack=True)

            data.index.names = ['trade_date', 'ticker']
            data = data.reset_index()
        else:
            # Single ticker case
            data.index.name = 'trade_date'
            data = data.reset_index()
            # If single ticker, yfinance doesn't return 'ticker' column, we must add it
            # NOTE: Ideally we pass the ticker in, but for now let's assume single ticker
            # or handle logic upstream. For simplicity in this method:
            if 'ticker' not in data.columns:
                # Fallback: We might lose ticker info if not careful.
                # In production, we iterate tickers.
                # For this Phase, we assume single ticker or handle upstream.
                data['ticker'] = "UNKNOWN"

        # 2. Normalize Column Names (Case Insensitive)
        data.columns = [str(c).lower().strip() for c in data.columns]

        # 3. Map to Internal Schema
        # We look for fuzzy matches because 'Adj Close' might be 'adj_close', 'adj close', 'adjclose'
        col_map = {
            'open': 'open_price',
            'high': 'high_price',
            'low': 'low_price',
            'close': 'close_price',
            'volume': 'volume'
        }

        # Find the specific 'adj close' column dynamically
        adj_col_candidate = next((c for c in data.columns if 'adj' in c and 'close' in c), None)

        rename_dict = {}
        for src, target in col_map.items():
            if src in data.columns:
                rename_dict[src] = target

        if adj_col_candidate:
            rename_dict[adj_col_candidate] = 'adj_close'
        else:
            logger.warning("Column 'Adj Close' missing! Strategy returns may be inaccurate.")
            # Fallback: If missing, use Close as Adj Close (Better than crashing, but logged)
            # But ONLY if we confirm Close exists
            if 'close' in data.columns:
                 data['adj_close'] = data['close']

        data = data.rename(columns=rename_dict)

        return data