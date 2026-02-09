
"""
Post-optimization risk guardrails.

Applies hard constraints to raw optimizer output before it reaches the
execution layer:
  1. **Dust filter** — positions below 1 % are zeroed out.
  2. **Cash buffer** — total equity exposure is scaled to
     ``1 − cash_buffer_pct`` so a liquidity reserve is always maintained.
  3. **Concentration cap** — no single asset may exceed ``max_weight_pct``.
  4. **USDC residual** — any weight freed by the cap flows into cash (USDC),
     *not* back into other risk assets.
"""
import numpy as np
import pandas as pd
from loguru import logger

from src.rwaengine.strategy.types import OptimizationResult


class PortfolioRiskManager:
    """Enforces position-level constraints and a mandatory cash buffer."""

    def __init__(
        self,
        cash_buffer_pct: float = 0.05,
        max_weight_pct: float = 0.30,
    ):
        """
        Args:
            cash_buffer_pct: Minimum fraction held in cash / USDC (e.g. 0.05 = 5 %).
            max_weight_pct: Hard cap on any single asset's weight (e.g. 0.30 = 30 %).
        """
        self.cash_buffer = cash_buffer_pct
        self.max_weight = max_weight_pct

    def apply_guardrails(
        self, result: OptimizationResult
    ) -> OptimizationResult:
        """Clean and constrain the optimizer's raw allocation.

        Steps:
          1. Zero out dust positions (< 1 %).
          2. Re-normalize remaining weights to sum to 1.
          3. Scale down to ``(1 − cash_buffer)`` total equity exposure.
          4. Hard-cap each position at ``max_weight``.
          5. Assign the remaining weight to USDC (cash).

        Args:
            result: Unconstrained optimization output.

        Returns:
            A new ``OptimizationResult`` with adjusted weights that satisfy
            all constraints, plus an explicit USDC cash position.
        """
        logger.info("Applying risk guardrails...")

        weights = pd.Series(data=result.weights, index=result.tickers)

        # Remove dust positions.
        weights[weights < 0.01] = 0.0

        # Re-normalise so surviving positions sum to 1.
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            logger.warning("All positions filtered as dust — defaulting to 100 % cash.")

        # Scale to target equity exposure.
        target_equity_exposure = 1.0 - self.cash_buffer
        weights = weights * target_equity_exposure

        # Apply per-asset hard cap; excess flows to cash, not other assets.
        overweight_mask = weights > self.max_weight
        if overweight_mask.any():
            capped_tickers = weights[overweight_mask].index.tolist()
            logger.warning(f"Capping concentrated positions: {capped_tickers}")
            weights[overweight_mask] = self.max_weight

        # Assign remaining capacity to USDC.
        final_equity_sum = weights.sum()
        usdc_weight = max(1.0 - final_equity_sum, 0.0)
        weights["USDC"] = usdc_weight

        logger.success(f"Risk check passed. USDC liquidity: {usdc_weight:.2%}")

        return OptimizationResult(
            tickers=weights.index.tolist(),
            weights=weights.values.tolist(),
            expected_return=result.expected_return * final_equity_sum,
            volatility=result.volatility * final_equity_sum,
            sharpe_ratio=result.sharpe_ratio,
        )