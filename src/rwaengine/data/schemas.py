
"""
Strict data contracts for market data.

Pydantic models defined here act as the single source of truth for the
OHLCV schema.  Financial models (e.g. Black-Litterman) can fail silently
on bad data (High < Low, negative prices), so validation is enforced at
the boundary — before anything reaches the core engine.
"""
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MarketData(BaseModel):
    """Validated OHLCV record for a single ticker on a single trading day."""

    ticker: str = Field(..., description="Asset symbol (e.g. AAPL)")
    trade_date: date = Field(..., description="Calendar date of the trading session")

    # All prices must be strictly positive.
    open_price: float = Field(..., gt=0, description="Opening price")
    high_price: float = Field(..., gt=0, description="Intraday high")
    low_price: float = Field(..., gt=0, description="Intraday low")
    close_price: float = Field(..., gt=0, description="Closing price")

    # Volume of zero is valid (e.g. market halt), but negative is not.
    volume: int = Field(..., ge=0, description="Number of shares traded")

    # Adjusted close is optional but essential for accurate backtesting.
    adj_close: Optional[float] = Field(None, gt=0, description="Split/dividend-adjusted close")

    @field_validator("high_price")
    @classmethod
    def high_must_not_be_below_low(cls, v: float, info) -> float:
        """Ensure the high price is never lower than the low price."""
        values = info.data
        if "low_price" in values and v < values["low_price"]:
            raise ValueError(
                f"High price ({v}) is below low price ({values['low_price']})"
            )
        return v