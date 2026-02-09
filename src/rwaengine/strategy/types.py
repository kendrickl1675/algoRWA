"""
Data contracts for Black-Litterman views and optimization results.

Pydantic models defined here enforce type safety and basic sanity
constraints (e.g. confidence ∈ (0, 1], weights length matches assets)
so that malformed inputs are caught early rather than silently
corrupting downstream calculations.
"""
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class StrategyConfig(BaseModel):
    """Global parameters shared across BL strategy runs."""

    risk_aversion: float = Field(
        2.5, gt=0,
        description="Market risk-aversion coefficient (delta)",
    )
    tau: float = Field(
        0.05, gt=0,
        description="Uncertainty scaling factor for the prior covariance",
    )


class InvestorView(BaseModel):
    """A single investor view suitable for the P/Q formulation.

    Example: "AAPL will outperform GOOGL by 2 % annually" translates to
    ``assets=["AAPL", "GOOGL"]``, ``weights=[1.0, -1.0]``,
    ``expected_return=0.02``.
    """

    assets: List[str] = Field(
        ..., min_length=1,
        description="Tickers involved in this view",
    )
    weights: List[float] = Field(
        ..., min_length=1,
        description="Asset loadings (one row of the P matrix)",
    )
    expected_return: float = Field(
        ...,
        description="Expected excess return (one element of the Q vector)",
    )
    confidence: float = Field(
        1.0, gt=0, le=1.0,
        description="View confidence on a 0–1 scale (maps to Omega)",
    )
    description: Optional[str] = Field(
        None,
        description="Human-readable rationale for the view",
    )

    @field_validator("weights")
    @classmethod
    def weights_must_match_assets(cls, v: List[float], info) -> List[float]:
        """Ensure the weights vector has the same length as the assets list."""
        if "assets" in info.data and len(v) != len(info.data["assets"]):
            raise ValueError(
                f"weights length ({len(v)}) must match "
                f"assets length ({len(info.data['assets'])})"
            )
        return v


class OptimizationResult(BaseModel):
    """Output of a portfolio optimization run."""

    tickers: List[str]
    weights: List[float]
    expected_return: float
    volatility: float
    sharpe_ratio: float