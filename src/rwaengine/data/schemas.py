"""
File: src/rwaengine/data/schemas.py
Description: Defines strict data contracts for market data.
"""
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MarketData(BaseModel):
    """
    Standard Market Data Model (OHLCV)

    Why this matters:
    Financial models (like Black-Litterman) will crash silently if fed
    bad data (e.g., High < Low, or negative prices). We catch this early.
    """
    ticker: str = Field(..., description="Asset Symbol (e.g., AAPL)")
    trade_date: date = Field(..., description="Trading Date")

    # Validation: Prices must be positive
    open_price: float = Field(..., gt=0, description="Open Price")
    high_price: float = Field(..., gt=0, description="High Price")
    low_price: float = Field(..., gt=0, description="Low Price")
    close_price: float = Field(..., gt=0, description="Close Price")

    # Volume can be 0 (market halt), but not negative
    volume: int = Field(..., ge=0, description="Trading Volume")

    # Adjusted Close is optional but crucial for backtesting
    adj_close: Optional[float] = Field(None, gt=0, description="Adjusted Close")

    @field_validator('high_price')
    @classmethod
    def validate_high_low(cls, v: float, info):
        """Risk Check: High price cannot be lower than Low price."""
        values = info.data
        if 'low_price' in values and v < values['low_price']:
            raise ValueError(f"High price {v} < Low price {values['low_price']}")
        return v