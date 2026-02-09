"""
Benchmark return calculators.

Provides static methods for computing buy-and-hold cumulative returns of
common benchmarks (SPY, equal-weight basket) aligned to a strategy's
date index so they can be plotted side-by-side.
"""
import pandas as pd


class BenchmarkProvider:
    """Utility class for computing benchmark cumulative-return curves."""

    @staticmethod
    def calculate_spy(
        spy_prices: pd.Series,
        target_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """Compute SPY cumulative returns aligned to *target_index*.

        Args:
            spy_prices: Daily SPY adjusted-close price series.
            target_index: DatetimeIndex of the strategy's return curve.

        Returns:
            Cumulative-return series starting at 1.0.
        """
        rets = spy_prices.pct_change()
        aligned_rets = rets.reindex(target_index).fillna(0.0)
        return (1 + aligned_rets).cumprod()

    @staticmethod
    def calculate_equal_weight(
        asset_prices: pd.DataFrame,
        target_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """Compute the equal-weight basket's cumulative return.

        Each asset receives an equal 1/N weight, rebalanced daily.

        Args:
            asset_prices: Wide-format DataFrame of adjusted-close prices.
            target_index: DatetimeIndex to align the output to.

        Returns:
            Cumulative-return series starting at 1.0.
        """
        daily_rets = asset_prices.pct_change()
        eq_rets = daily_rets.mean(axis=1)
        aligned_rets = eq_rets.reindex(target_index).fillna(0.0)
        return (1 + aligned_rets).cumprod()