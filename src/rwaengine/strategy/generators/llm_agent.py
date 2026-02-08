"""
Mode 3: LLM-Based Generation (Gemini + Google Search) - v3.1 (Calibrated)
Description:
    - Fixes Time Horizon Mismatch: Converts inputs to Annualized Returns for BL Model.
    - Boosts Confidence Scores: Tier 2 now has enough weight to impact optimization.
"""
import os
from datetime import date
from typing import List, Optional, Literal
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

# === LangChain & Google GenAI Imports ===
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from google.genai.types import Tool, GoogleSearch

from src.rwaengine.strategy.base import ViewGenerator
from src.rwaengine.strategy.types import InvestorView

# === Schema (保持 v3.0 不变) ===
class AnalystSignal(BaseModel):
    has_consensus: bool = Field(description="Is there a clear analyst consensus?")
    target_price: Optional[float] = Field(description="The average analyst price target. Null if not found.")
    current_price_ref: Optional[float] = Field(description="The current price mentioned in the news.")

class EvidenceTier(BaseModel):
    tier: Literal["Tier 1", "Tier 2", "Tier 3"] = Field(
        description="Select evidence strength: Tier 1 (Hard Event/Earnings), Tier 2 (Analyst Re-rating), Tier 3 (Rumors)."
    )
    reasoning: str = Field(description="Why this tier was selected.")

class AssetViewScorecard(BaseModel):
    ticker: str
    sentiment: Literal["Bullish", "Bearish", "Neutral"]
    signal_data: AnalystSignal
    evidence_strength: EvidenceTier
    catalyst_summary: str = Field(description="One sentence summary of the catalyst.")

class InvestorViewsScorecardList(BaseModel):
    views: List[AssetViewScorecard]

# ==========================================================

class GeminiViewGenerator(ViewGenerator):
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        if not api_key:
            raise ValueError("Gemini API Key is required for LLM Strategy.")

        self._search_tool = Tool(google_search=GoogleSearch())

        # 保持 0.0 温度以确保提取数据的确定性
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
        )

        self.parser = PydanticOutputParser(pydantic_object=InvestorViewsScorecardList)

    def _construct_prompt(self, tickers: List[str], market_summary: str) -> ChatPromptTemplate:
        # Prompt 保持 v3.0 的提取逻辑，因为提取本身没问题，是后续计算出了问题
        system_template = """
        You are a Quantitative Data Extractor. Your job is NOT to predict the future, but to EXTRACT hard data points from Google Search results to populate a "Signal Scorecard".

        ### STEP 1: GATHER EVIDENCE (Google Search)
        For each ticker in {tickers}, search for news in the current month ({current_month_str}).
        Look specifically for:
        1. **Analyst Price Targets**: Recent updates from major banks (GS, MS, JPM).
        2. **Corporate Events**: Earnings reports, Guidance changes, M&A, Buybacks.

        ### STEP 2: FILL THE SCORECARD (Strict Rules)
        
        **A. Signal Data**
        - Extract the `target_price` (average consensus) and `current_price_ref`.
        - If no clear target price is found in recent news, leave `target_price` as null.
        
        **B. Evidence Tier**
        - **"Tier 1" (High)**: Confirmed Hard Event (Earnings beat/miss > 5%, M&A, SEC Filing).
        - **"Tier 2" (Medium)**: Institutional Opinion (Price Target Upgrade/Downgrade by Tier-1 Bank).
        - **"Tier 3" (Low)**: Noise (Blog opinions, Technical Analysis, General Sentiment).

        ### CONSTRAINTS
        - If Sentiment is Neutral or Evidence is Tier 3, the view will be discarded later.
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
            ("human", human_template)
        ])

    def _calculate_implied_view(self, scorecard: AssetViewScorecard, current_price_fallback: float) -> Optional[InvestorView]:
        """
        修正后的确定性计算逻辑 (v3.1)
        """
        # 1. 过滤无效信号
        if scorecard.evidence_strength.tier == "Tier 3" or scorecard.sentiment == "Neutral":
            return None

        # 提升 Tier 2 的置信度，使其方差 (Omega) 足够小，能够拉动高市值的 AAPL/MSFT
        # Tier 1: 0.95 (Almost Certain)
        # Tier 2: 0.80 (Strong Conviction)
        confidence_map = {
            "Tier 1": 0.95,
            "Tier 2": 0.80,
            "Tier 3": 0.00
        }
        confidence = confidence_map.get(scorecard.evidence_strength.tier, 0.0)

        # Calibrate Magnitude (Annualized)]
        exp_return_annualized = 0.0

        current_price = scorecard.signal_data.current_price_ref or current_price_fallback
        target_price = scorecard.signal_data.target_price

        if target_price and current_price > 0:

            raw_upside = (target_price - current_price) / current_price
            exp_return_annualized = max(min(raw_upside, 0.80), -0.50)

        else:
            direction = 1.0 if scorecard.sentiment == "Bullish" else -1.0

            base_annual_move = 0.20 if scorecard.evidence_strength.tier == "Tier 1" else 0.10
            exp_return_annualized = direction * base_annual_move

        if abs(exp_return_annualized) < 0.02:
            return None

        return InvestorView(
            assets=[scorecard.ticker],
            weights=[1.0],
            expected_return=exp_return_annualized,
            confidence=confidence,
            description=f"[{scorecard.evidence_strength.tier}] {scorecard.catalyst_summary} (Annualized Imp. Ret: {exp_return_annualized:.2%})"
        )

    def generate_views(self, current_prices: pd.Series) -> List[InvestorView]:
        tickers = current_prices.index.tolist()
        today = date.today()
        current_month_str = today.strftime("%B %Y")

        logger.info(f"  Agent scoring views for {tickers} [Mode: Scorecard v3.1]...")

        market_summary = current_prices.to_string()
        prompt = self._construct_prompt(tickers, market_summary)

        try:
            chain = prompt | self.llm.bind_tools([self._search_tool]) | self.parser

            result = chain.invoke({
                "tickers": ", ".join(tickers),
                "market_summary": market_summary,
                "current_date": today.isoformat(),
                "current_month_str": current_month_str,
                "format_instructions": self.parser.get_format_instructions()
            })

            final_views = []

            for card in result.views:
                fallback_price = current_prices.get(card.ticker, 100.0)
                view = self._calculate_implied_view(card, fallback_price)
                if view:
                    final_views.append(view)

            logger.success(f"Generated {len(final_views)} calibrated views.")
            return final_views

        except Exception as e:
            logger.error(f"LLM Scoring failed: {e}")
            return []