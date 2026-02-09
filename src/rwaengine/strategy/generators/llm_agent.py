"""
LLM-based investor-view generator (Gemini + Google Search).

Uses Google's Gemini model with grounded Google Search to extract a
structured "Signal Scorecard" for each ticker, then deterministically
converts those scorecards into annualised ``InvestorView`` objects for the
Black-Litterman model.

Key design decisions:
  - Temperature is set to 0 for reproducible data extraction.
  - Confidence is tiered: Tier 1 (hard events) → 0.95, Tier 2 (analyst
    re-ratings) → 0.80.  Tier 3 and Neutral signals are discarded.
  - Implied returns are annualised and clamped to [-50 %, +80 %] to avoid
    degenerate BL posteriors.
"""
from datetime import date
from typing import List, Literal, Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai.types import GoogleSearch, Tool

from src.rwaengine.strategy.base import ViewGenerator
from src.rwaengine.strategy.types import InvestorView


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

class AnalystSignal(BaseModel):
    """Extracted price-target data for a single ticker."""

    has_consensus: bool = Field(
        description="Whether a clear analyst consensus exists."
    )
    target_price: Optional[float] = Field(
        description="Average analyst price target; null if unavailable."
    )
    current_price_ref: Optional[float] = Field(
        description="Current price as mentioned in the source material."
    )


class EvidenceTier(BaseModel):
    """Qualitative strength assessment of the supporting evidence."""

    tier: Literal["Tier 1", "Tier 2", "Tier 3"] = Field(
        description=(
            "Tier 1: hard event (earnings, M&A, SEC filing). "
            "Tier 2: institutional opinion (price-target change). "
            "Tier 3: noise (blogs, TA, general sentiment)."
        )
    )
    reasoning: str = Field(description="Justification for the chosen tier.")


class AssetViewScorecard(BaseModel):
    """Complete scorecard produced by the LLM for one ticker."""

    ticker: str
    sentiment: Literal["Bullish", "Bearish", "Neutral"]
    signal_data: AnalystSignal
    evidence_strength: EvidenceTier
    catalyst_summary: str = Field(
        description="One-sentence summary of the catalyst."
    )


class InvestorViewsScorecardList(BaseModel):
    """Top-level container returned by the LLM chain."""

    views: List[AssetViewScorecard]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class GeminiViewGenerator(ViewGenerator):
    """Produces investor views by querying Gemini with grounded search."""

    # Confidence mapping: controls the BL Omega (view uncertainty) matrix.
    _CONFIDENCE_MAP = {
        "Tier 1": 0.95,
        "Tier 2": 0.80,
        "Tier 3": 0.00,
    }

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Args:
            api_key: Google AI API key.
            model_name: Gemini model identifier.

        Raises:
            ValueError: If *api_key* is empty or ``None``.
        """
        if not api_key:
            raise ValueError("Gemini API key is required for the LLM strategy.")

        self._search_tool = Tool(google_search=GoogleSearch())

        # Temperature 0 for deterministic data extraction.
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
        )

        self.parser = PydanticOutputParser(
            pydantic_object=InvestorViewsScorecardList
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _construct_prompt(
        self, tickers: List[str], market_summary: str
    ) -> ChatPromptTemplate:
        """Build a two-message (system + human) prompt template."""

        system_template = """
You are a Quantitative Data Extractor. Your job is NOT to predict the future,
but to EXTRACT hard data points from Google Search results to populate a
"Signal Scorecard".

### STEP 1: GATHER EVIDENCE (Google Search)
For each ticker in {tickers}, search for news in the current month
({current_month_str}). Look specifically for:
1. **Analyst Price Targets**: Recent updates from major banks (GS, MS, JPM).
2. **Corporate Events**: Earnings reports, Guidance changes, M&A, Buybacks.

### STEP 2: FILL THE SCORECARD (Strict Rules)

**A. Signal Data**
- Extract the `target_price` (average consensus) and `current_price_ref`.
- If no clear target price is found in recent news, leave `target_price` as null.

**B. Evidence Tier**
- **"Tier 1" (High)**: Confirmed Hard Event (Earnings beat/miss > 5%, M&A,
  SEC Filing).
- **"Tier 2" (Medium)**: Institutional Opinion (Price Target Upgrade/Downgrade
  by Tier-1 Bank).
- **"Tier 3" (Low)**: Noise (Blog opinions, Technical Analysis, General
  Sentiment).

### CONSTRAINTS
- If Sentiment is Neutral or Evidence is Tier 3, the view will be discarded.
- Do NOT hallucinate prices.

{format_instructions}
"""

        human_template = """
[Market Context - Recent Prices]
{market_summary}

[Task]
Analyze {tickers} for date: {current_date}.
Populate the Scorecard JSON.
"""

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template),
        ])

    # ------------------------------------------------------------------
    # Scorecard → InvestorView conversion
    # ------------------------------------------------------------------

    def _calculate_implied_view(
        self,
        scorecard: AssetViewScorecard,
        current_price_fallback: float,
    ) -> Optional[InvestorView]:
        """Deterministically convert a scorecard into an annualised view.

        Tier 3 evidence and Neutral sentiment are filtered out entirely.
        When a concrete analyst target price is available, the implied return
        is computed directly; otherwise a directional base-case return is
        assigned based on evidence tier strength.

        Args:
            scorecard: Structured output from the LLM.
            current_price_fallback: Price to use when the scorecard's own
                                    ``current_price_ref`` is missing.

        Returns:
            An ``InvestorView`` or ``None`` if the signal is too weak.
        """
        # Discard low-quality or neutral signals.
        if (
            scorecard.evidence_strength.tier == "Tier 3"
            or scorecard.sentiment == "Neutral"
        ):
            return None

        confidence = self._CONFIDENCE_MAP.get(
            scorecard.evidence_strength.tier, 0.0
        )

        current_price = (
            scorecard.signal_data.current_price_ref or current_price_fallback
        )
        target_price = scorecard.signal_data.target_price

        if target_price and current_price > 0:
            # Compute implied upside and clamp to [-50 %, +80 %].
            raw_upside = (target_price - current_price) / current_price
            exp_return_annualized = max(min(raw_upside, 0.80), -0.50)
        else:
            # No price target available — assign a directional base-case.
            direction = 1.0 if scorecard.sentiment == "Bullish" else -1.0
            base_move = (
                0.20
                if scorecard.evidence_strength.tier == "Tier 1"
                else 0.10
            )
            exp_return_annualized = direction * base_move

        # Views below 2 % annualised are too small to meaningfully shift BL.
        if abs(exp_return_annualized) < 0.02:
            return None

        return InvestorView(
            assets=[scorecard.ticker],
            weights=[1.0],
            expected_return=exp_return_annualized,
            confidence=confidence,
            description=(
                f"[{scorecard.evidence_strength.tier}] "
                f"{scorecard.catalyst_summary} "
                f"(Annualised implied return: {exp_return_annualized:.2%})"
            ),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_views(self, current_prices: pd.Series) -> List[InvestorView]:
        """Query Gemini for each ticker and return calibrated investor views.

        Args:
            current_prices: Series indexed by ticker with the latest prices.

        Returns:
            List of ``InvestorView`` objects (may be empty if the LLM call
            fails or all signals are filtered out).
        """
        tickers = current_prices.index.tolist()
        today = date.today()
        current_month_str = today.strftime("%B %Y")

        logger.info(f"Scoring views for {tickers} via Gemini scorecard...")

        market_summary = current_prices.to_string()
        prompt = self._construct_prompt(tickers, market_summary)

        try:
            chain = (
                prompt
                | self.llm.bind_tools([self._search_tool])
                | self.parser
            )

            result = chain.invoke({
                "tickers": ", ".join(tickers),
                "market_summary": market_summary,
                "current_date": today.isoformat(),
                "current_month_str": current_month_str,
                "format_instructions": self.parser.get_format_instructions(),
            })

            final_views: List[InvestorView] = []
            for card in result.views:
                fallback_price = current_prices.get(card.ticker, 100.0)
                view = self._calculate_implied_view(card, fallback_price)
                if view:
                    final_views.append(view)

            logger.success(f"Generated {len(final_views)} calibrated views.")
            return final_views

        except Exception as e:
            logger.error(f"LLM scoring failed: {e}")
            return []