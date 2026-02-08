"""
File: src/rwaengine/execution/risk_manager.py
Description: The Gatekeeper. Enforces hard constraints and liquidity buffers.
"""
from typing import List
from loguru import logger
import pandas as pd
import numpy as np

# 复用之前定义的数据结构
from src.rwaengine.strategy.types import OptimizationResult


class PortfolioRiskManager:
    def __init__(self, cash_buffer_pct: float = 0.05, max_weight_pct: float = 0.30):
        """
        Args:
            cash_buffer_pct: 现金缓冲比例 (e.g. 0.05 = 5%)
            max_weight_pct: 单一资产最大持仓 (e.g. 0.30 = 30%)
        """
        self.cash_buffer = cash_buffer_pct
        self.max_weight = max_weight_pct

    def apply_guardrails(self, result: OptimizationResult) -> OptimizationResult:
        """
        对优化结果进行风控清洗。

        修正后的逻辑 (V2):
        1. 过滤碎股。
        2. 全局缩放至 (1 - CashBuffer)。
        3. 应用硬顶 (Hard Cap)。
        4. 被硬顶削减的溢出权重，直接回流到 USDC，不再重新分配给风险资产。
        """
        logger.info("🛡️ Applying Audit Guardrails (V2 - Strict Cap)...")

        weights = pd.Series(data=result.weights, index=result.tickers)

        weights[weights < 0.01] = 0.0

        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            logger.warning("All assets filtered out. 100% Cash.")

        target_equity_exposure = 1.0 - self.cash_buffer
        weights = weights * target_equity_exposure

        # 在缩放后检查。如果某资产是 0.95 (Step 4后)，而 Cap 是 0.30
        # 我们将其强制设为 0.30。差额 (0.65) 自然不再属于该资产。
        overweight = weights > self.max_weight
        if overweight.any():
            overweight_tickers = weights[overweight].index.tolist()
            logger.warning(f"Capping concentrated positions: {overweight_tickers}")
            weights[weights > self.max_weight] = self.max_weight

        final_equity_sum = weights.sum()
        usdc_weight = 1.0 - final_equity_sum

        if usdc_weight < 0:
            usdc_weight = 0.0

        weights['USDC'] = usdc_weight

        logger.success(f"Risk Check Passed. Final Liquidity (USDC): {usdc_weight:.2%}")

        return OptimizationResult(
            tickers=weights.index.tolist(),
            weights=weights.values.tolist(),
            expected_return=result.expected_return * final_equity_sum,
            volatility=result.volatility * final_equity_sum,
            sharpe_ratio=result.sharpe_ratio
        )