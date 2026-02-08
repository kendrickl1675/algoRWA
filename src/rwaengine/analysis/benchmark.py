"""
File: src/rwaengine/analysis/benchmark.py
Description: Benchmark calculation utilities (SPY & Equal Weight).
"""
import pandas as pd
from typing import List


class BenchmarkProvider:
    @staticmethod
    def calculate_spy(spy_prices: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
        """
        计算 SPY 的累计收益，并对齐到策略的时间轴。
        """
        rets = spy_prices.pct_change()
        aligned_rets = rets.reindex(target_index).fillna(0.0)
        return (1 + aligned_rets).cumprod()

    @staticmethod
    def calculate_equal_weight(asset_prices: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.Series:
        """
        计算资产池等权重持有的累计收益。
        """
        asset_rets = asset_prices.pct_change()

        eq_rets = asset_rets.mean(axis=1)

        aligned_rets = eq_rets.reindex(target_index).fillna(0.0)

        return (1 + aligned_rets).cumprod()