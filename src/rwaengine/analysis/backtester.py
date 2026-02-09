"""
Multi-strategy backtester with allocation weight tracking.

Iterates through a price history at a fixed rebalancing frequency, asks
each registered strategy for new portfolio weights, then computes daily
returns weighted by those allocations.  Returns both cumulative return
curves and a full audit trail of historical weights.
"""
import pandas as pd
from datetime import timedelta
from loguru import logger
from typing import Dict, List, Tuple

from src.rwaengine.analysis.strategies import BaseStrategy


class Backtester:
    """Walk-forward backtester that supports an arbitrary list of strategies."""

    def __init__(self, prices: pd.DataFrame, strategies: List[BaseStrategy]):
        """
        Args:
            prices: Wide-format DataFrame — DatetimeIndex × ticker columns,
                    values are adjusted-close prices.
            strategies: Strategy instances to evaluate side-by-side.
        """
        self.prices = prices
        self.strategies = strategies

    def run(
        self,
        start_date: str,
        rebalance_freq_days: int = 20,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Execute the walk-forward simulation.

        Args:
            start_date: ISO-format date string for the first rebalancing point.
            rebalance_freq_days: Calendar-day gap between rebalancing events.

        Returns:
            A tuple of:
              - Cumulative-return DataFrame (one column per strategy).
              - Dict mapping strategy name → DataFrame of weight snapshots.
        """
        logger.info(f"Starting multi-strategy backtest from {start_date}...")

        full_dates = self.prices.index
        start_idx = full_dates.get_indexer(
            [pd.Timestamp(start_date)], method="nearest"
        )[0]
        rebalance_dates = full_dates[start_idx::rebalance_freq_days]

        strategy_returns: Dict[str, list] = {s.name: [] for s in self.strategies}
        weights_history: Dict[str, list] = {s.name: [] for s in self.strategies}

        # Every strategy starts fully in cash (USDC) until its first rebalance.
        current_weights: Dict[str, Dict[str, float]] = {
            s.name: {"USDC": 1.0} for s in self.strategies
        }

        for i in range(len(rebalance_dates) - 1):
            period_start = rebalance_dates[i]
            period_end = rebalance_dates[i + 1]

            # Build a 1-year lookback window for strategy signals.
            lookback_start = period_start - timedelta(days=365)
            hist_data = self.prices.loc[lookback_start:period_start]

            # Only rebalance when enough history is available.
            if len(hist_data) > 100:
                for strat in self.strategies:
                    try:
                        new_weights = strat.rebalance(hist_data)
                        if new_weights:
                            current_weights[strat.name] = new_weights

                            # Snapshot the weights for post-hoc analysis.
                            record = new_weights.copy()
                            record["Date"] = period_start
                            weights_history[strat.name].append(record)

                    except Exception as e:
                        logger.error(
                            f"Strategy {strat.name} failed at "
                            f"{period_start.date()}: {e}"
                        )

            # Compute weighted daily returns for the period.
            period_prices = self.prices.loc[period_start:period_end]
            period_rets = period_prices.pct_change().dropna()

            for strat_name in strategy_returns:
                w = current_weights[strat_name]
                daily = pd.Series(0.0, index=period_rets.index)
                for ticker, weight in w.items():
                    if ticker != "USDC" and ticker in period_rets.columns:
                        daily += period_rets[ticker] * weight
                strategy_returns[strat_name].append(daily)

            if i % 10 == 0:
                logger.info(f"Step {period_start.date()}: rebalanced.")

        # Assemble cumulative return curves.
        result_df = pd.DataFrame()
        for name, ret_list in strategy_returns.items():
            if ret_list:
                full_series = pd.concat(ret_list)
                result_df[name] = (1 + full_series).cumprod()

        # Assemble weight audit trail.
        final_weights: Dict[str, pd.DataFrame] = {}
        for name, records in weights_history.items():
            if records:
                df_w = pd.DataFrame(records)
                if not df_w.empty:
                    df_w.set_index("Date", inplace=True)
                    df_w = df_w.fillna(0.0)
                    final_weights[name] = df_w

        return result_df, final_weights