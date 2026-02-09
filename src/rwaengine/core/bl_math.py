
"""
Pure NumPy implementation of the Black-Litterman formulas.

Provides two standalone functions that mirror the canonical BL derivation
from He & Litterman (1999):
  1. ``compute_market_implied_prior`` — equilibrium returns (Pi).
  2. ``compute_posterior`` — posterior distribution after incorporating
     investor views via the master formula.

These are kept separate from the PyPortfolioOpt-based engine so they can
be used for unit testing, research notebooks, or alternative optimization
pipelines.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from loguru import logger


def compute_market_implied_prior(
    cov_matrix: pd.DataFrame,
    market_caps: pd.Series,
    risk_aversion: float,
    risk_free_rate: float,
) -> pd.Series:
    """Compute market-implied equilibrium returns (Pi).

    Formula: ``Pi = delta * Sigma @ w_mkt + r_f``

    Args:
        cov_matrix: N × N asset covariance matrix (Sigma).
        market_caps: Market capitalizations for each asset.
        risk_aversion: Risk-aversion coefficient (delta).
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Series of implied excess returns indexed by ticker.
    """
    # Normalize market caps to portfolio weights.
    w_mkt = market_caps / market_caps.sum()

    # Align weight ordering with the covariance matrix.
    w_mkt = w_mkt.reindex(cov_matrix.index).fillna(0)

    prior = risk_aversion * cov_matrix.dot(w_mkt)
    return prior + risk_free_rate


def compute_posterior(
    cov_matrix: pd.DataFrame,
    prior_returns: pd.Series,
    P: np.ndarray,
    Q: np.ndarray,
    tau: float,
    confidences: Optional[List[float]] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Compute the posterior return distribution (BL master formula).

    Combines the market prior with K investor views to produce updated
    expected returns and a posterior covariance matrix.

    Args:
        cov_matrix: N × N covariance matrix (Sigma).
        prior_returns: N × 1 market-implied returns (Pi).
        P: K × N view-picking matrix (each row selects assets for one view).
        Q: K × 1 vector of expected view returns.
        tau: Scalar uncertainty parameter on the prior covariance.
        confidences: K × 1 view confidences in [0, 1].  When provided,
                     each view's variance in Omega is scaled by ``1 / c``
                     so that higher confidence => tighter uncertainty.

    Returns:
        A tuple of (posterior_mu, posterior_sigma):
          - ``posterior_mu``: Series of posterior expected returns.
          - ``posterior_sigma``: DataFrame of the posterior covariance.

    Raises:
        ValueError: If a singular matrix is encountered during inversion.
    """
    Sigma = cov_matrix.values
    Pi = prior_returns.values.reshape(-1, 1)
    Q = Q.reshape(-1, 1)

    tau_Sigma = tau * Sigma

    # View-implied variance: diag(P @ tau*Sigma @ P^T).
    P_tau_Sigma_PT = P @ tau_Sigma @ P.T
    view_variances = np.diag(P_tau_Sigma_PT)

    # Build Omega (view uncertainty matrix).
    # Omega_ii = view_variance / confidence — high confidence shrinks
    # uncertainty, low confidence inflates it toward infinity.
    if confidences:
        adj_factors = np.array(
            [1.0 / c if c > 0 else 1e6 for c in confidences]
        )
        Omega = np.diag(view_variances * adj_factors)
    else:
        Omega = np.diag(view_variances)

    # Master formula:
    #   E[R] = M^{-1} @ [(tau*Sigma)^{-1} Pi + P^T Omega^{-1} Q]
    #   where M = (tau*Sigma)^{-1} + P^T Omega^{-1} P
    try:
        inv_tau_Sigma = np.linalg.inv(tau_Sigma)
        inv_Omega = np.linalg.inv(Omega)

        M = inv_tau_Sigma + P.T @ inv_Omega @ P
        rhs = inv_tau_Sigma @ Pi + P.T @ inv_Omega @ Q

        inv_M = np.linalg.inv(M)
        posterior_mu = inv_M @ rhs

        # Posterior covariance includes both the original risk and the
        # residual estimation uncertainty.
        posterior_sigma = Sigma + inv_M

    except np.linalg.LinAlgError as e:
        logger.error(f"Matrix inversion failed: {e}")
        raise ValueError(
            "Singular matrix encountered in BL posterior calculation."
        ) from e

    mu_series = pd.Series(posterior_mu.flatten(), index=cov_matrix.index)
    sigma_df = pd.DataFrame(
        posterior_sigma,
        index=cov_matrix.index,
        columns=cov_matrix.columns,
    )

    return mu_series, sigma_df