
"""
Portfolio definition loader.

Reads a JSON file that maps portfolio names (e.g. ``"mag_seven"``) to
lists of ticker symbols and caches the result in memory.  The file is
loaded eagerly at construction time so that configuration errors surface
immediately rather than mid-pipeline.

Expected JSON structure::

    {
      "mag_seven": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
      "default":   ["SPY"]
    }
"""
import json
import os
from pathlib import Path
from typing import Dict, List

from loguru import logger


class PortfolioLoader:
    """Loads and caches portfolio → ticker mappings from a JSON file."""

    def __init__(
        self,
        portfolio_file: str = "portfolios/portfolios.json",
    ):
        """
        Args:
            portfolio_file: Path to the portfolio definitions file, relative
                            to the current working directory.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file contains invalid JSON.
        """
        self.file_path = Path(os.getcwd()) / portfolio_file
        self._cache: Dict[str, List[str]] = {}
        self._load_portfolios()

    def _load_portfolios(self) -> None:
        """Read the JSON file into the in-memory cache.

        Raises:
            FileNotFoundError: If *self.file_path* does not exist.
            ValueError: If the file is not valid JSON.
        """
        if not self.file_path.exists():
            logger.critical(f"Portfolio file not found at: {self.file_path}")
            raise FileNotFoundError(
                f"Missing portfolio definition file: {self.file_path}"
            )

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            logger.info(
                f"Loaded portfolio definitions from {self.file_path.name}"
            )
        except json.JSONDecodeError as e:
            logger.critical(f"Invalid JSON in portfolio file: {e}")
            raise ValueError("Corrupted portfolio definition file") from e

    def get_tickers(self, portfolio_name: str) -> List[str]:
        """Return the list of ticker symbols for the given portfolio.

        Args:
            portfolio_name: Key in the JSON file (e.g. ``"mag_seven"``).

        Returns:
            List of ticker strings.

        Raises:
            KeyError: If *portfolio_name* is not present in the file.
        """
        if portfolio_name not in self._cache:
            available = list(self._cache.keys())
            logger.error(
                f"Portfolio '{portfolio_name}' not found. "
                f"Available: {available}"
            )
            raise KeyError(f"Unknown portfolio: {portfolio_name}")

        tickers = self._cache[portfolio_name]
        logger.info(
            f"Selected portfolio '{portfolio_name}': {len(tickers)} assets"
        )
        return tickers