
"""
Black-Litterman optimization engine.

Wraps PyPortfolioOpt to execute the full BL pipeline:
  1. Compute the market-implied prior (equilibrium returns).
  2. Blend the prior with investor views to obtain a posterior.
  3. Run mean-variance optimization (max Sharpe) on the posterior.

If optimization fails (e.g. near-singular matrices or degenerate
returns), the engine falls back to a 100 % cash (USDC) allocation
rather than producing unreliable risk-asset weights.
"""
import numpy as np
import pandas as pd
from typing import Dict, List

from loguru import logger
from pypfopt import black_litterman, efficient_frontier, risk_models

from src.rwaengine.strategy.types import (
    InvestorView,
    OptimizationResult,
    StrategyConfig,
)


class BlackLittermanEngine:
    """End-to-end BL optimizer: prior → posterior → max-Sharpe weights."""

    def __init__(
        self,
        prices: pd.DataFrame,
        config: StrategyConfig = StrategyConfig(),
    ):
        """
        Args:
            prices: Wide-format DataFrame of adjusted-close prices
                    (DatetimeIndex × tickers).
            config: Strategy hyperparameters (risk aversion, tau).
        """
        self.prices = prices
        self.config = config

        # Pre-compute the covariance matrix using the Ledoit-Wolf shrinkage
        # estimator for improved numerical stability.
        logger.info("Computing covariance matrix (Ledoit-Wolf)...")
        self.S = risk_models.CovarianceShrinkage(self.prices).ledoit_wolf()

    def run_optimization(
        self,
        market_caps: Dict[str, float],
        views: List[InvestorView],
    ) -> OptimizationResult:
        """Execute the full BL optimization pipeline.

        Args:
            market_caps: Mapping of ticker → market capitalization.
            views: Investor views to blend with the market prior.  If empty,
                   the optimizer falls back to the equilibrium prior alone.

        Returns:
            An ``OptimizationResult`` containing cleaned weights and
            performance metrics (expected return, volatility, Sharpe).
        """
        tickers = self.prices.columns.tolist()

        logger.info("Calculating market-implied prior returns...")

        mcaps = pd.Series(market_caps).reindex(tickers)

        # Fill missing market caps with the mean so the pipeline doesn't
        # crash.  In production this should raise instead.
        if mcaps.isnull().any():
            missing = mcaps[mcaps.isnull()].index.tolist()
            logger.warning(f"Missing market caps for {missing}. Filling with mean.")
            mcaps = mcaps.fillna(mcaps.mean())

        delta = self.config.risk_aversion
        logger.info(f"Risk aversion (delta): {delta:.4f}")

        market_prior = black_litterman.market_implied_prior_returns(
            market_caps=mcaps,
            risk_aversion=delta,
            cov_matrix=self.S,
            risk_free_rate=0.04,
        )

        # Without views, optimization uses the equilibrium prior directly.
        if not views:
            logger.warning("No views provided — using market prior only.")
            posterior_rets = market_prior
            posterior_cov = self.S
        else:
            logger.info(f"Integrating {len(views)} investor views...")

            Q, P, view_confidences = self._parse_views(views, tickers)

            bl = black_litterman.BlackLittermanModel(
                cov_matrix=self.S,
                pi=market_prior,
                absolute_views=None,
                Q=Q,
                P=P,
                omega="idzorek",
                view_confidences=view_confidences,
                tau=self.config.tau,
                risk_aversion=delta,
            )

            posterior_rets = bl.bl_returns()
            posterior_cov = bl.bl_cov()

        # Mean-variance optimization on the posterior distribution.
        # L2 regularization is intentionally omitted; concentration control
        # is handled downstream by the risk-management module.
        logger.info("Optimizing portfolio weights (max Sharpe)...")

        ef = efficient_frontier.EfficientFrontier(posterior_rets, posterior_cov)

        try:
            ef.max_sharpe(risk_free_rate=0.04)
            cleaned_weights = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=0.04)
        except Exception as e:
            logger.error(f"Optimization failed: {e}. Falling back to cash.")
            return self._fallback_result(tickers)

        return OptimizationResult(
            tickers=list(cleaned_weights.keys()),
            weights=list(cleaned_weights.values()),
            expected_return=perf[0],
            volatility=perf[1],
            sharpe_ratio=perf[2],
        )

    def _parse_views(
        self,
        views: List[InvestorView],
        tickers: List[str],
    ):
        """Convert a list of ``InvestorView`` objects into NumPy matrices.

        Returns:
            A tuple of (Q, P, confidences):
              - ``Q``: K-element expected-return vector.
              - ``P``: K × N picking matrix.
              - ``confidences``: K-element confidence list.
        """
        n_views = len(views)
        n_assets = len(tickers)

        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        confidences = []

        # Build a ticker → column-index lookup for fast P-matrix population.
        mapper = {t: idx for idx, t in enumerate(tickers)}

        for i, view in enumerate(views):
            Q[i] = view.expected_return
            confidences.append(view.confidence)

            for asset, weight in zip(view.assets, view.weights):
                if asset in mapper:
                    P[i, mapper[asset]] = weight
                else:
                    logger.error(
                        f"Asset '{asset}' in view not found in market data."
                    )

        return Q, P, confidences

    def _fallback_result(self, tickers: List[str]) -> OptimizationResult:
        """Return a zero-risk allocation when the optimizer fails.

        Rather than blindly equal-weighting into risk assets on a failed
        optimization (which usually signals degenerate data), this defaults
        to 100 % cash so the risk-management layer can handle it cleanly.
        """
        logger.warning(
            "Optimizer fallback: flight-to-safety (0 % risk assets)."
        )

        return OptimizationResult(
            tickers=tickers,
            weights=[0.0] * len(tickers),
            expected_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
        )