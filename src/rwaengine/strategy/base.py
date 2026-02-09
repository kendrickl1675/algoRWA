
"""
Abstract interface for investor-view generators.

All concrete view sources (static JSON, ML-based, LLM-based) must
implement the ``generate_views`` method defined here.  This ensures the
Black-Litterman engine can consume views from any source through a
single uniform contract.
"""
from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from src.rwaengine.strategy.types import InvestorView


class ViewGenerator(ABC):
    """Contract that every investor-view source must satisfy."""

    @abstractmethod
    def generate_views(self, current_prices: pd.Series) -> List[InvestorView]:
        """Produce a list of investor views based on the current market state.

        Args:
            current_prices: Series indexed by ticker with the most recent
                            prices.  Implementations may use these to compute
                            implied market caps or to validate asset coverage.

        Returns:
            A list of standardised ``InvestorView`` objects ready for the
            Black-Litterman model.
        """