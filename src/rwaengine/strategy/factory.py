from typing import Literal
import pandas as pd
from loguru import logger

from src.rwaengine.strategy.base import ViewGenerator
from src.rwaengine.strategy.generators.json_loader import JsonViewGenerator
from src.rwaengine.strategy.generators.ml_predictor import MLViewGenerator
from src.rwaengine.strategy.generators.llm_agent import GeminiViewGenerator

StrategyMode = Literal["json", "ml", "llm"]


class StrategyFactory:
    @staticmethod
    def get_generator(
            mode: StrategyMode,
            **kwargs
    ) -> ViewGenerator:
        """
        Args:
            mode: 'json', 'ml', or 'llm'
            kwargs: 传递给具体 Generator 的参数
                - json: portfolio_name
                - ml: history_data
                - llm: api_key
        """
        logger.info(f"Initializing Strategy Engine: Mode={mode.upper()}")

        if mode == "json":
            return JsonViewGenerator(
                portfolio_name=kwargs.get("portfolio_name", "default")
            )

        elif mode == "ml":
            history = kwargs.get("history_data")
            if history is None:
                raise ValueError("ML Generator requires 'history_data'")
            return MLViewGenerator(history_data=history)

        elif mode == "llm":
            key = kwargs.get("api_key")
            if not key:
                raise ValueError("LLM Generator requires 'api_key'")
            return GeminiViewGenerator(api_key=key)

        else:
            raise ValueError(f"Unknown strategy mode: {mode}")