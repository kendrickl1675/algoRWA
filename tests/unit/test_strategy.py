"""
File: src/rwaengine/strategy/generators/manual.py
Description: Manually defined views (hardcoded or from config).
"""
import pandas as pd
from typing import List, Dict
from loguru import logger

from src.rwaengine.strategy.base import ViewGenerator
from src.rwaengine.strategy.types import InvestorView


class ManualViewGenerator(ViewGenerator):
    def __init__(self, predefined_views: List[Dict]):
        """
        Args:
            predefined_views: List of dicts matching InvestorView schema.
        """
        self.raw_views = predefined_views

    def generate_views(self, current_prices: pd.Series) -> List[InvestorView]:
        logger.info("Generating Manual Views...")

        valid_views = []
        for v_data in self.raw_views:
            try:
                # 校验资产是否存在于当前价格池中
                assets = v_data.get('assets', [])
                missing = [a for a in assets if a not in current_prices.index]

                if missing:
                    logger.warning(f"View skipped. Assets {missing} not in market data.")
                    continue

                # 转换为严格模型
                view = InvestorView(**v_data)
                valid_views.append(view)

            except Exception as e:
                logger.error(f"Invalid view data: {v_data} | Error: {e}")

        logger.success(f"Generated {len(valid_views)} valid investor views.")
        return valid_views