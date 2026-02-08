"""
File: src/rwaengine/analysis/strategies.py
Description: Strategy definitions with dynamic View Source configuration.
"""
from abc import ABC, abstractmethod
import pandas as pd
import os
from loguru import logger
from typing import Dict, Optional, Literal

from pypfopt import risk_models, expected_returns, EfficientFrontier

from src.rwaengine.core.engine import BlackLittermanEngine
from src.rwaengine.execution.risk_manager import PortfolioRiskManager
from src.rwaengine.strategy.types import OptimizationResult

# [New] 引入工厂和具体生成器
from src.rwaengine.strategy.factory import StrategyFactory
from src.rwaengine.strategy.generators.json_loader import JsonViewGenerator

# 定义支持的视图来源类型
ViewSourceType = Literal["json", "ml", "llm"]

class BaseStrategy(ABC):
    def __init__(self, name: str, risk_manager: Optional[PortfolioRiskManager] = None):
        self.name = name
        self.risk_manager = risk_manager

    @abstractmethod
    def rebalance(self, history: pd.DataFrame, **kwargs) -> Dict[str, float]:
        pass

    def _apply_risk_or_pass(self, raw_result: OptimizationResult) -> Dict[str, float]:
        """Helper: Apply risk guardrails if manager exists, else return raw."""
        if self.risk_manager:
            final_res = self.risk_manager.apply_guardrails(raw_result)
            return dict(zip(final_res.tickers, final_res.weights))
        else:
            return dict(zip(raw_result.tickers, raw_result.weights))


class MarkowitzStrategy(BaseStrategy):
    def rebalance(self, history: pd.DataFrame, **kwargs) -> Dict[str, float]:
        try:
            mu = expected_returns.mean_historical_return(history)
            S = risk_models.CovarianceShrinkage(history).ledoit_wolf()

            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe(risk_free_rate=0.04)
            cleaned = ef.clean_weights()

            raw_res = OptimizationResult(
                tickers=list(cleaned.keys()),
                weights=list(cleaned.values()),
                expected_return=0, volatility=0, sharpe_ratio=0
            )
            return self._apply_risk_or_pass(raw_res)
        except Exception as e:
            logger.warning(f"[{self.name}] Optimization failed: {e}")
            return {}


class BLStrategy(BaseStrategy):
    def __init__(self,
                 name: str,
                 risk_manager: Optional[PortfolioRiskManager],
                 portfolio_name: str,
                 view_source: ViewSourceType = "json",
                 view_file: str = "portfolios/views_backtest.json"):
        """
        Args:
            view_source: 'json' (static), 'llm' (dynamic AI), 'ml' (dynamic Algo)
            view_file: Only used if view_source is 'json'
        """
        super().__init__(name, risk_manager)
        self.portfolio_name = portfolio_name
        self.view_source = view_source
        self.view_file = view_file
        self.mock_caps = None

        # 预加载 API Key (仅当需要时)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.view_source == "llm" and not self.api_key:
            logger.warning("⚠️ LLM mode selected but GEMINI_API_KEY not found!")

    def _get_generator(self, history: pd.DataFrame):
        """
        Helper: 根据配置动态获取观点生成器
        """
        # 1. JSON Mode: 为了支持自定义文件路径，我们直接实例化而不是走 Factory
        # (Factory 默认只会加载 portfolios/views.json)
        if self.view_source == "json":
            return JsonViewGenerator(
                portfolio_name=self.portfolio_name,
                view_file=self.view_file
            )

        # 2. LLM / ML Mode: 使用 Factory
        factory_kwargs = {
            "portfolio_name": self.portfolio_name,
            "api_key": self.api_key,
            "history_data": history # ML 模式需要最新的历史数据进行训练/预测
        }

        return StrategyFactory.get_generator(self.view_source, **factory_kwargs)

    def rebalance(self, history: pd.DataFrame, **kwargs) -> Dict[str, float]:
        if self.mock_caps is None:
             self.mock_caps = {t: 1e12 for t in history.columns}

        try:
            # [Dynamic] 获取生成器实例
            generator = self._get_generator(history)

            # 生成观点
            # 注意: LLM 调用会消耗 Quota 且速度较慢
            views = generator.generate_views(history.iloc[-1])

            # 运行 BL 引擎
            engine = BlackLittermanEngine(prices=history)
            raw_res = engine.run_optimization(market_caps=self.mock_caps, views=views)

            return self._apply_risk_or_pass(raw_res)

        except Exception as e:
            logger.warning(f"[{self.name}] Optimization failed: {e}")
            return {}