
"""
Factory for investor-view generators.

Maps a strategy mode string (``"json"``, ``"ml"``, ``"llm"``) to the
concrete ``ViewGenerator`` subclass that produces ``InvestorView`` objects
for the Black-Litterman model.
"""
from typing import Literal

from loguru import logger

from src.rwaengine.strategy.base import ViewGenerator
from src.rwaengine.strategy.generators.json_loader import JsonViewGenerator
from src.rwaengine.strategy.generators.llm_agent import GeminiViewGenerator
from src.rwaengine.strategy.generators.ml_predictor import MLViewGenerator

StrategyMode = Literal["json", "ml", "llm"]


class StrategyFactory:
    """Stateless factory that resolves a mode string to a generator instance."""

    @staticmethod
    def get_generator(mode: StrategyMode, **kwargs) -> ViewGenerator:
        """Create and return a ``ViewGenerator`` for the requested mode.

        Args:
            mode: One of ``"json"``, ``"ml"``, or ``"llm"``.
            **kwargs: Forwarded to the concrete generator constructor.
                - *json*: ``portfolio_name`` (str).
                - *ml*: ``history_data`` (pd.DataFrame).
                - *llm*: ``api_key`` (str).

        Raises:
            ValueError: If *mode* is not recognised or if a required kwarg
                        is missing.
        """
        logger.info(f"Initializing view generator: mode={mode.upper()}")

        if mode == "json":
            return JsonViewGenerator(
                portfolio_name=kwargs.get("portfolio_name", "default"),
            )

        if mode == "ml":
            history = kwargs.get("history_data")
            if history is None:
                raise ValueError("ML generator requires 'history_data'")
            return MLViewGenerator(history_data=history)

        if mode == "llm":
            key = kwargs.get("api_key")
            if not key:
                raise ValueError("LLM generator requires 'api_key'")
            return GeminiViewGenerator(api_key=key)

        raise ValueError(f"Unknown strategy mode: {mode}")