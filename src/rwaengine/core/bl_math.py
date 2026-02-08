"""
File: src/rwaengine/core/bl_math.py
Description: Pure Numpy implementation of Black-Litterman formulas.
Reference: "The Intuition Behind Black-Litterman Model Portfolios", He & Litterman (1999)
"""
import numpy as np
import pandas as pd
from typing import Tuple
from loguru import logger


def compute_market_implied_prior(
        cov_matrix: pd.DataFrame,
        market_caps: pd.Series,
        risk_aversion: float,
        risk_free_rate: float
) -> pd.Series:
    """
    Step 1: 计算市场隐含均衡收益 (Pi)
    Formula: Pi = delta * Sigma * w_mkt
    """
    # 归一化市场权重
    w_mkt = market_caps / market_caps.sum()

    # 确保 w_mkt 的顺序与 cov_matrix 一致
    w_mkt = w_mkt.reindex(cov_matrix.index).fillna(0)

    # Pi = delta * Sigma @ w_mkt
    prior = risk_aversion * cov_matrix.dot(w_mkt)

    return prior + risk_free_rate


def compute_posterior(
        cov_matrix: pd.DataFrame,
        prior_returns: pd.Series,
        P: np.ndarray,
        Q: np.ndarray,
        tau: float,
        confidences: list[float] = None
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Step 2: 计算后验收益分布 (Master Formula)

    Args:
        cov_matrix (Sigma): N x N 协方差矩阵
        prior_returns (Pi): N x 1 市场隐含收益
        P: K x N 观点矩阵 (K个观点, N个资产)
        Q: K x 1 观点收益向量
        tau: 不确定性系数 (Scalar)
        confidences: K x 1 观点置信度 (0.0 - 1.0)

    Returns:
        posterior_mu: 后验预期收益
        posterior_sigma: 后验协方差矩阵
    """
    Sigma = cov_matrix.values
    Pi = prior_returns.values.reshape(-1, 1)

    Q = Q.reshape(-1, 1)

    tau_Sigma = tau * Sigma
    P_tau_Sigma_PT = P @ tau_Sigma @ P.T

    view_variances = np.diag(P_tau_Sigma_PT)

    if confidences:
        # 这里采用: Omega_ii = Variance / Confidence
        # 当 Conf=100%, Omega=Variance; Conf->0, Omega->Inf
        adj_factors = np.array([1.0 / c if c > 0 else 1e6 for c in confidences])
        Omega = np.diag(view_variances * adj_factors)
    else:
        Omega = np.diag(view_variances)

    # E[R] = [(tau*Sigma)^-1 + P.T * Omega^-1 * P]^-1 * [(tau*Sigma)^-1 * Pi + P.T * Omega^-1 * Q]

    try:
        inv_tau_Sigma = np.linalg.inv(tau_Sigma)
        inv_Omega = np.linalg.inv(Omega)

        # M = [(tau*Sigma)^-1 + P.T * Omega^-1 * P]
        M = inv_tau_Sigma + P.T @ inv_Omega @ P

        # Term 2 = [(tau*Sigma)^-1 * Pi + P.T * Omega^-1 * Q]
        Term2 = inv_tau_Sigma @ Pi + P.T @ inv_Omega @ Q

        # Result
        posterior_mu = np.linalg.inv(M) @ Term2

        # Sigma_post = Sigma + [(tau*Sigma)^-1 + P.T * Omega^-1 * P]^-1
        posterior_sigma = Sigma + np.linalg.inv(M)

    except np.linalg.LinAlgError as e:
        logger.error(f"Matrix Inversion Failed: {e}")
        raise ValueError("Singular matrix encountered in BL calculation")

    res_series = pd.Series(posterior_mu.flatten(), index=cov_matrix.index)
    res_df = pd.DataFrame(posterior_sigma, index=cov_matrix.index, columns=cov_matrix.columns)

    return res_series, res_df