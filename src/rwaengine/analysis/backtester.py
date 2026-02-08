"""
File: src/rwaengine/analysis/backtester.py
Description: Generalized Backtester with Weight Tracking.
"""
import pandas as pd
from datetime import timedelta
from loguru import logger
from typing import List, Dict, Tuple

from src.rwaengine.analysis.strategies import BaseStrategy

class Backtester:
    def __init__(self, prices: pd.DataFrame, strategies: List[BaseStrategy]):
        self.prices = prices
        self.strategies = strategies

    def run(self, start_date: str, rebalance_freq_days: int = 20) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Returns:
            Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
            1. Cumulative Returns DataFrame
            2. Historical Weights Dictionary {strategy_name: weights_df}
        """
        logger.info(f"⏳ Starting Multi-Strategy Backtest from {start_date}...")

        full_dates = self.prices.index
        start_idx = full_dates.get_indexer([pd.Timestamp(start_date)], method='nearest')[0]
        rebalance_dates = full_dates[start_idx::rebalance_freq_days]

        strategy_returns = {s.name: [] for s in self.strategies}

        weights_history = {s.name: [] for s in self.strategies}

        current_weights = {s.name: {"USDC": 1.0} for s in self.strategies}

        for i in range(len(rebalance_dates) - 1):
            date = rebalance_dates[i]
            next_date = rebalance_dates[i+1]

            lookback_start = date - timedelta(days=365)
            hist_data = self.prices.loc[lookback_start:date]

            if len(hist_data) > 100:
                for strat in self.strategies:
                    try:
                        new_weights = strat.rebalance(hist_data)
                        if new_weights:
                            current_weights[strat.name] = new_weights

                            # [新增] 记录调仓日的权重
                            record = new_weights.copy()
                            record["Date"] = date
                            weights_history[strat.name].append(record)

                    except Exception as e:
                        logger.error(f"Strategy {strat.name} failed at {date.date()}: {e}")

            period_prices = self.prices.loc[date:next_date]
            period_rets = period_prices.pct_change().dropna()

            for strat_name in strategy_returns.keys():
                w = current_weights[strat_name]
                daily = pd.Series(0.0, index=period_rets.index)
                for t, weight in w.items():
                    if t != "USDC" and t in period_rets.columns:
                        daily += period_rets[t] * weight
                strategy_returns[strat_name].append(daily)

            if i % 10 == 0:
                logger.info(f"📅 Step {date.date()}: Rebalanced.")

        result_df = pd.DataFrame()
        for name, ret_list in strategy_returns.items():
            if ret_list:
                full_series = pd.concat(ret_list)
                result_df[name] = (1 + full_series).cumprod()

        final_weights = {}
        for name, records in weights_history.items():
            if records:
                df_w = pd.DataFrame(records)
                if not df_w.empty:
                    df_w.set_index("Date", inplace=True)
                    df_w = df_w.fillna(0.0)
                    final_weights[name] = df_w

        return result_df, final_weights