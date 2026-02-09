"""
Portfolio rebalancing strategies.

Provides a pluggable strategy hierarchy:
  - ``BaseStrategy``: abstract interface shared by all strategies.
  - ``MarkowitzStrategy``: classic mean-variance (max Sharpe) optimisation.
  - ``BLStrategy``: Black-Litterman model with dynamic investor-view sources
    (static JSON, ML-based, or LLM-generated).
"""
import os
from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional

import pandas as pd
from loguru import logger
from pypfopt import EfficientFrontier, expected_returns, risk_models

from src.rwaengine.core.engine import BlackLittermanEngine
from src.rwaengine.execution.risk_manager import PortfolioRiskManager
from src.rwaengine.strategy.types import OptimizationResult

ViewSourceType = Literal["json", "ml", "llm"]


class BaseStrategy(ABC):
    """Common interface for all portfolio rebalancing strategies."""

    def __init__(
        self,
        name: str,
        risk_manager: Optional[PortfolioRiskManager] = None,
    ):
        self.name = name
        self.risk_manager = risk_manager

    @abstractmethod
    def rebalance(self, history: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Return target portfolio weights given recent price history."""

    def _apply_risk_or_pass(
        self, raw_result: OptimizationResult
    ) -> Dict[str, float]:
        """Apply risk guardrails when a manager is configured, otherwise
        return the raw optimiser output unchanged."""
        if self.risk_manager:
            adjusted = self.risk_manager.apply_guardrails(raw_result)
            return dict(zip(adjusted.tickers, adjusted.weights))
        return dict(zip(raw_result.tickers, raw_result.weights))


class MarkowitzStrategy(BaseStrategy):
    """Mean-variance optimisation targeting the maximum Sharpe ratio."""

    def rebalance(self, history: pd.DataFrame, **kwargs) -> Dict[str, float]:
        try:
            mu = expected_returns.mean_historical_return(history)
            cov = risk_models.CovarianceShrinkage(history).ledoit_wolf()

            ef = EfficientFrontier(mu, cov)
            ef.max_sharpe(risk_free_rate=0.04)
            cleaned = ef.clean_weights()

            raw_res = OptimizationResult(
                tickers=list(cleaned.keys()),
                weights=list(cleaned.values()),
                expected_return=0,
                volatility=0,
                sharpe_ratio=0,
            )
            return self._apply_risk_or_pass(raw_res)

        except Exception as e:
            logger.warning(f"[{self.name}] Optimization failed: {e}")
            return {}


class BLStrategy(BaseStrategy):
    """Black-Litterman strategy with configurable investor-view sources."""

    def __init__(
        self,
        name: str,
        risk_manager: Optional[PortfolioRiskManager],
        portfolio_name: str,
        view_source: ViewSourceType = "json",
        view_file: str = "portfolios/views_backtest.json",
    ):
        """
        Args:
            name: Human-readable strategy label.
            risk_manager: Optional guardrail manager applied after optimisation.
            portfolio_name: Key used to look up tickers and view definitions.
            view_source: Where investor views come from — ``"json"``, ``"ml"``,
                         or ``"llm"``.
            view_file: Path to a static JSON view file (only used when
                       *view_source* is ``"json"``).
        """
        super().__init__(name, risk_manager)
        self.portfolio_name = portfolio_name
        self.view_source = view_source
        self.view_file = view_file
        self.mock_caps: Optional[Dict[str, float]] = None

        # Only needed for the LLM path; loaded eagerly so we can warn early.
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.view_source == "llm" and not self.api_key:
            logger.warning("LLM mode selected but GEMINI_API_KEY is not set!")

    def _get_generator(self, history: pd.DataFrame):
        """Instantiate the appropriate view generator for the configured source.

        JSON mode bypasses the factory so that a custom *view_file* path can
        be forwarded.  ML and LLM modes use the standard ``StrategyFactory``.
        """
        if self.view_source == "json":
            return JsonViewGenerator(
                portfolio_name=self.portfolio_name,
                view_file=self.view_file,
            )

        return StrategyFactory.get_generator(
            self.view_source,
            portfolio_name=self.portfolio_name,
            api_key=self.api_key,
            history_data=history,
        )

    def rebalance(self, history: pd.DataFrame, **kwargs) -> Dict[str, float]:
        # Lazily initialise equal mock market-caps on first call.
        if self.mock_caps is None:
            self.mock_caps = {t: 1e12 for t in history.columns}

        try:
            generator = self._get_generator(history)
            views = generator.generate_views(history.iloc[-1])

            engine = BlackLittermanEngine(prices=history)
            raw_res = engine.run_optimization(
                market_caps=self.mock_caps, views=views
            )
            return self._apply_risk_or_pass(raw_res)

        except Exception as e:
            logger.warning(f"[{self.name}] Optimization failed: {e}")
            return {}