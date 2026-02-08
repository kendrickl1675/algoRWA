"""
File: src/data/adapters/yfinance_adapter.py
Description: Robust YFinance adapter handling version discrepancies.
"""
from datetime import date
from typing import List, Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from src.rwaengine.data.base import MarketDataProvider

class YFinanceAdapter(MarketDataProvider):
    def __init__(self, proxy: Optional[str] = None):
        """
        Args:
            proxy: Proxy URL for restricted regions (e.g. Macau/CN).
        """
        self.proxy = proxy

    # === 在 fetch_history 方法中修改异常处理部分 ===
    def fetch_history(self, tickers: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        logger.info(f"Fetching {len(tickers)} assets ({start_date} to {end_date}) | Proxy: {self.proxy}")

        try:
            # yfinance 特性：end_date 是 exclusive 的，如果 start==end，它会报错或返回空
            if start_date == end_date:
                logger.warning(f"Start date equals End date ({start_date}). Returns will be empty.")
                return pd.DataFrame()

            df = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=True,
                group_by='ticker'
            )

            # [Fix 1] 早期空值拦截
            if df.empty:
                logger.warning(f"No data returned for {tickers}. Possible holiday or delisting.")
                return pd.DataFrame()

            # Clean and Standardize
            df_clean = self._standardize_columns(df)

            # [Fix 2] 清洗后再次检查，防止清洗逻辑把数据洗没了
            if df_clean.empty:
                logger.warning("Data became empty after standardization.")
                return pd.DataFrame()

            # Validation
            self.validate_schema(df_clean)

            logger.success(f"Fetched {len(df_clean)} rows successfully.")
            return df_clean

        except Exception as e:
            # 捕获所有未知异常，防止整个程序崩溃，但要记录 Stack Trace
            logger.error(f"YFinance Critical Failure: {str(e)}")
            # 在开发阶段，我们还是抛出异常以便调试；在生产环境可能会吞掉
            raise

    # === 完全替换 _standardize_columns ===
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Robust column mapper handling yfinance version chaos.
        """
        data = df.copy()

        # [Safety Check] 如果 yfinance 返回了只有 Index 没有 Column 的僵尸数据
        if len(data.columns) == 0:
            return pd.DataFrame()

        # 1. Handle Multi-Ticker format
        if isinstance(data.columns, pd.MultiIndex):
            # Try to infer structure. Level 0 could be Price or Ticker depending on version
            example_col = data.columns[0]

            # yfinance 默认 group_by='ticker' -> (Ticker, Price)
            # 我们需要把 Ticker 这一层保留，把 Price 这一层 stack 变成 Series
            # 这段逻辑非常 tricky，最好的办法是看 column level names

            try:
                # 尝试标准处理：假设 Level 0 是 Ticker (因为我们设置了 group_by='ticker')
                data = data.stack(level=0, future_stack=True)
                data.index.names = ['trade_date', 'ticker']
                data = data.reset_index()
            except Exception:
                # Fallback: 尝试 stack level 1
                data = data.stack(level=1, future_stack=True)
                data.index.names = ['trade_date', 'ticker']
                data = data.reset_index()

        else:
            # Single ticker case
            data.index.name = 'trade_date'
            data = data.reset_index()
            if 'ticker' not in data.columns:
                data['ticker'] = "UNKNOWN"  # 暂时标记，生产环境通常由上层传入 Ticker Map 修正

        # 2. Normalize Column Names (Case Insensitive)
        data.columns = [str(c).lower().strip() for c in data.columns]

        # 3. Map to Internal Schema
        col_map = {
            'open': 'open_price',
            'high': 'high_price',
            'low': 'low_price',
            'close': 'close_price',
            'volume': 'volume'
        }

        # 动态寻找 Adj Close
        adj_col_candidate = next((c for c in data.columns if 'adj' in c and 'close' in c), None)

        rename_dict = {}
        found_columns = set(data.columns)

        # 构建重命名映射，只映射存在的列
        for src, target in col_map.items():
            if src in found_columns:
                rename_dict[src] = target

        if adj_col_candidate:
            rename_dict[adj_col_candidate] = 'adj_close'

        data = data.rename(columns=rename_dict)

        # [Critical] 如果重命名后，核心字段依然缺失（比如 yfinance 下载到了数据但全是 NaN），
        # 此时返回空 DF 比返回坏数据要好。
        required = {'open_price', 'close_price'}
        if not required.issubset(data.columns):
            logger.warning(f"Data missing critical columns after cleaning. Available: {data.columns}")
            return pd.DataFrame()

        # Fallback for adj_close if missing
        if 'adj_close' not in data.columns and 'close_price' in data.columns:
            data['adj_close'] = data['close_price']

        return data