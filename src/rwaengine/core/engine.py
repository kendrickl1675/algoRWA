"""
File: src/rwaengine/core/engine.py
Description: The Core Calculation Engine using PyPortfolioOpt.
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from loguru import logger

from pypfopt import black_litterman
from pypfopt import risk_models
from pypfopt import efficient_frontier
from pypfopt import objective_functions

from src.rwaengine.strategy.types import InvestorView, StrategyConfig, OptimizationResult

class BlackLittermanEngine:
    def __init__(self, prices: pd.DataFrame, config: StrategyConfig = StrategyConfig()):
        """
        初始化引擎，预计算风险模型。
        """
        self.prices = prices
        self.config = config

        # 计算协方差矩阵 (使用 Ledoit-Wolf 收缩算法以增强稳定性)
        logger.info("Computing Covariance Matrix (Ledoit-Wolf)...")
        self.S = risk_models.CovarianceShrinkage(self.prices).ledoit_wolf()

    def run_optimization(
        self,
        market_caps: Dict[str, float],
        views: List[InvestorView]
    ) -> OptimizationResult:
        """
        执行 BL 模型全流程：
        1. Market Prior (Pi)
        2. Posterior Estimate (BL Model)
        3. Mean-Variance Optimization
        """
        tickers = self.prices.columns.tolist()

        logger.info("Calculating Market Implied Returns (Prior)...")

        mcaps = pd.Series(market_caps).reindex(tickers)

        # [Risk Control] 填充缺失市值。生产环境应报错，这里用均值填充防止 Crash
        if mcaps.isnull().any():
            logger.warning(f"Missing market caps for {mcaps[mcaps.isnull()].index.tolist()}. Using mean.")
            mcaps = mcaps.fillna(mcaps.mean())

        delta = self.config.risk_aversion
        logger.info(f"Market Implied Risk Aversion (Delta): {delta:.4f}")

        market_prior = black_litterman.market_implied_prior_returns(
            market_caps=mcaps,
            risk_aversion=delta,
            cov_matrix=self.S,
            risk_free_rate=0.04
        )

        if not views:
            logger.warning("No views provided. Fallback to Market Prior.")
            posterior_rets = market_prior
            posterior_cov = self.S
        else:
            logger.info(f"Integrating {len(views)} Investor Views...")

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
                risk_aversion=delta
            )

            posterior_rets = bl.bl_returns()
            posterior_cov = bl.bl_cov()

        logger.info("Optimizing Portfolio Weights (Max Sharpe)...")

        ef = efficient_frontier.EfficientFrontier(
            posterior_rets,
            posterior_cov
        )

        # 不使用l2正则, 通过后续的风控模块控制集中度
        # ef.add_objective(objective_functions.L2_reg, gamma=0.1)

        # risk_free_rate=0.04 (对应当前美债收益率)
        try:
            raw_weights = ef.max_sharpe(risk_free_rate=0.04)
            cleaned_weights = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=0.04)
        except Exception as e:
            logger.error(f"Optimization failed: {e}. Fallback to Equal Weights.")
            return self._fallback_result(tickers)

        return OptimizationResult(
            tickers=list(cleaned_weights.keys()),
            weights=list(cleaned_weights.values()),
            expected_return=perf[0],
            volatility=perf[1],
            sharpe_ratio=perf[2]
        )

    def _parse_views(self, views: List[InvestorView], tickers: List[str]):
        """
        Helper: 将 InvestorView 列表转换为 Numpy 矩阵 (P, Q) 和置信度列表。
        """
        n_views = len(views)
        n_assets = len(tickers)

        P = np.zeros((n_views, n_assets)) # 观点矩阵
        Q = np.zeros(n_views)             # 预期收益向量
        confidences = []

        # 建立 Ticker -> Index 的映射表
        mapper = {t: i for i, t in enumerate(tickers)}

        for i, view in enumerate(views):
            Q[i] = view.expected_return
            confidences.append(view.confidence)

            # 填充 P 矩阵的一行
            for asset, weight in zip(view.assets, view.weights):
                if asset in mapper:
                    col_idx = mapper[asset]
                    P[i, col_idx] = weight
                else:
                    logger.error(f"Asset {asset} in view not found in market data.")

        return Q, P, confidences

    def _fallback_result(self, tickers: List[str]) -> OptimizationResult:
        """
        当优化器崩溃时的兜底方案。

        如果优化失败（通常是因为收益率太低或矩阵奇异），
        与其盲目买入风险资产 (Equal Weights)，不如持有现金 (100% USDC)。
        """
        logger.warning("Optimization failed. Fallback Strategy: FLIGHT TO SAFETY (100% USDC).")


        n = len(tickers)
        safe_weights = [0.0] * n

        return OptimizationResult(
            tickers=tickers,
            weights=safe_weights,
            expected_return=0.0,  # 假设持有现金超额收益为0
            volatility=0.0,
            sharpe_ratio=0.0
        )