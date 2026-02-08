"""
Mode 1: Static JSON Configuration
"""
import json
import os
from typing import List
from pathlib import Path
from loguru import logger
import pandas as pd

from src.rwaengine.strategy.base import ViewGenerator
from src.rwaengine.strategy.types import InvestorView


class JsonViewGenerator(ViewGenerator):
    def __init__(self, portfolio_name: str, view_file: str = "portfolios/views.json"):
        self.portfolio_name = portfolio_name
        self.file_path = Path(os.getcwd()) / view_file

    def generate_views(self, current_prices: pd.Series) -> List[InvestorView]:
        logger.info(f"Loading static views from {self.file_path} for {self.portfolio_name}...")

        if not self.file_path.exists():
            logger.warning("View file not found. Returning empty views.")
            return []

        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)

            raw_views = data.get(self.portfolio_name, [])
            valid_views = []

            for v_data in raw_views:
                assets = v_data.get('assets', [])
                if not all(a in current_prices.index for a in assets):
                    logger.warning(f"Skipping view {v_data.get('description')} due to missing assets in market data.")
                    continue

                valid_views.append(InvestorView(**v_data))

            return valid_views

        except Exception as e:
            logger.error(f"Failed to load JSON views: {e}")
            return []