
"""
Manual (hardcoded) investor-view generator.

Accepts a pre-built list of view dictionaries — typically from a config
file or an interactive notebook — validates each one against the current
market-data universe, and returns only the views whose assets are present
in the price data.
"""
from typing import Dict, List

import pandas as pd
from loguru import logger

from src.rwaengine.strategy.base import ViewGenerator
from src.rwaengine.strategy.types import InvestorView


class ManualViewGenerator(ViewGenerator):
    """Produces investor views from a caller-supplied list of raw dicts."""

    def __init__(self, predefined_views: List[Dict]):
        """
        Args:
            predefined_views: Each dict must match the ``InvestorView`` schema
                              (keys: ``assets``, ``weights``, ``expected_return``,
                              ``confidence``, ``description``).
        """
        self.raw_views = predefined_views

    def generate_views(self, current_prices: pd.Series) -> List[InvestorView]:
        """Filter and parse the predefined views.

        Views referencing tickers that are absent from *current_prices* are
        silently skipped (with a warning), so downstream code never receives
        a view it cannot act on.

        Args:
            current_prices: Series indexed by ticker with the latest prices.

        Returns:
            List of validated ``InvestorView`` objects.
        """
        logger.info("Generating manual views...")

        valid_views: List[InvestorView] = []
        for v_data in self.raw_views:
            try:
                assets = v_data.get("assets", [])
                missing = [a for a in assets if a not in current_prices.index]

                if missing:
                    logger.warning(
                        f"View skipped — assets {missing} not in market data."
                    )
                    continue

                valid_views.append(InvestorView(**v_data))

            except Exception as e:
                logger.error(f"Invalid view data: {v_data} | Error: {e}")

        logger.success(f"Generated {len(valid_views)} valid investor views.")
        return valid_views