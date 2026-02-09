
"""
Static JSON investor-view generator.

Reads a JSON file keyed by portfolio name and returns the investor views
defined under that key.  Views whose assets are missing from the current
market data are silently skipped.

Expected JSON structure::

    {
      "mag_seven": [
        {
          "assets": ["NVDA"],
          "weights": [1.0],
          "expected_return": 0.30,
          "confidence": 0.85,
          "description": "Strong AI demand outlook"
        },
        ...
      ]
    }
"""
import json
import os
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

from src.rwaengine.strategy.base import ViewGenerator
from src.rwaengine.strategy.types import InvestorView


class JsonViewGenerator(ViewGenerator):
    """Loads investor views from a static JSON configuration file."""

    def __init__(
        self,
        portfolio_name: str,
        view_file: str = "portfolios/views.json",
    ):
        """
        Args:
            portfolio_name: Top-level key to look up in the JSON file.
            view_file: Path (relative to the working directory) to the views
                       JSON file.
        """
        self.portfolio_name = portfolio_name
        self.file_path = Path(os.getcwd()) / view_file

    def generate_views(self, current_prices: pd.Series) -> List[InvestorView]:
        """Parse the JSON file and return validated views.

        Args:
            current_prices: Series indexed by ticker with the latest prices.
                            Used to verify that every asset in a view is
                            present in the market data.

        Returns:
            List of ``InvestorView`` objects.  Empty if the file is missing,
            the portfolio key is absent, or all views fail validation.
        """
        logger.info(
            f"Loading static views from {self.file_path} "
            f"for portfolio '{self.portfolio_name}'..."
        )

        if not self.file_path.exists():
            logger.warning(f"View file not found: {self.file_path}")
            return []

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_views = data.get(self.portfolio_name, [])
            valid_views: List[InvestorView] = []

            for v_data in raw_views:
                assets = v_data.get("assets", [])
                if not all(a in current_prices.index for a in assets):
                    logger.warning(
                        f"Skipping view '{v_data.get('description')}' — "
                        "one or more assets missing from market data."
                    )
                    continue

                valid_views.append(InvestorView(**v_data))

            return valid_views

        except Exception as e:
            logger.error(f"Failed to load JSON views: {e}")
            return []