"""
File: main.py
Description: End-to-End Pipeline Execution for RWA Fund Engine.
Usage:
    uv run main.py --portfolio mag_seven
"""
import argparse
import os
import pandas as pd
from datetime import date, timedelta
from loguru import logger
from src.rwaengine.utils.logger import setup_logger
setup_logger()
from dotenv import load_dotenv
import yfinance as yf
import json

from src.rwaengine.data.adapters.yfinance_adapter import YFinanceAdapter
from src.rwaengine.strategy.generators.manual import ManualViewGenerator
from src.rwaengine.core.engine import BlackLittermanEngine
from src.rwaengine.execution.risk_manager import PortfolioRiskManager
from src.rwaengine.utils.portfolio_loader import PortfolioLoader
from src.rwaengine.oracle.nav_reporter import NAVReporter
from src.rwaengine.strategy.factory import StrategyFactory

TEST_PRIVATE_KEY = "0x0000000000000000000000000000000000000000000000000000000000000001"
load_dotenv()

def get_latest_market_caps(tickers):
    """Helper: Get real-time market caps"""
    caps = {}
    logger.info("Fetching real-time market caps...")
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            caps[t] = info.get('marketCap', 1e11)
        except Exception:
            caps[t] = 1e11
    return caps


def run_pipeline(portfolio_name: str, enable_risk: bool, strategy_mode: str):
    logger.info(f"=== Starting RWA Pipeline [Portfolio: {portfolio_name}] ===")

    try:
        loader = PortfolioLoader()
        tickers = loader.get_tickers(portfolio_name)
    except Exception as e:
        logger.critical(f"Failed: {e}")
        return

    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    adapter = YFinanceAdapter()

    try:
        df_market = adapter.fetch_history(tickers, start_date, end_date)
    except Exception as e:
        logger.critical(f"Data fetch failed: {e}")
        return

    prices = df_market.pivot(index='trade_date', columns='ticker', values='adj_close')
    prices = prices.ffill().dropna()

    if prices.shape[1] < len(tickers):
        tickers = prices.columns.tolist()

    market_caps = get_latest_market_caps(tickers)

    factory_kwargs = {
        "portfolio_name": portfolio_name,
        "history_data": prices,
        "api_key": os.getenv("GEMINI_API_KEY")
    }

    try:
        generator = StrategyFactory.get_generator(strategy_mode, **factory_kwargs)
        views = generator.generate_views(current_prices=prices.iloc[-1])

        if views:
            debug_filename = f"debug_views_{portfolio_name}.json"
            logger.info(f"Exporting generated views to {debug_filename}...")

            views_data = [v.model_dump() for v in views]

            with open(debug_filename, "w", encoding="utf-8") as f:
                json.dump(views_data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Strategy generation failed: {e}. Fallback to Empty Views.")
        views = []

    # 4. Core Engine
    engine = BlackLittermanEngine(prices=prices)
    raw_result = engine.run_optimization(market_caps=market_caps, views=views)

    # 5. Risk Management (Gatekeeper)
    if enable_risk:
        risk_manager = PortfolioRiskManager(cash_buffer_pct=0.05, max_weight_pct=0.30)
        final_result = risk_manager.apply_guardrails(raw_result)
    else:
        final_result = raw_result

    print("\n" + "=" * 50)
    print("📡 PREPARING ORACLE PAYLOAD")
    print("=" * 50)

    signer_key = os.getenv("RWA_SIGNER_KEY", TEST_PRIVATE_KEY)

    try:
        reporter = NAVReporter(private_key=signer_key, portfolio_id=portfolio_name)
        signed_response = reporter.package_results(final_result)

        print(f"Signer Address: {signed_response.signer_address}")
        print(f"Signature:      {signed_response.signature[:10]}...{signed_response.signature[-10:]}")

        output_file = f"oracle_output_{portfolio_name}.json"
        reporter.export_json(signed_response, output_file)

    except Exception as e:
        logger.error(f"Oracle reporting failed: {e}")

    df_res = pd.DataFrame({
        "Asset": final_result.tickers,
        "Weight": [f"{w:.2%}" for w in final_result.weights]
    })
    if "USDC" in df_res["Asset"].values:
        usdc_row = df_res[df_res["Asset"] == "USDC"]
        other_rows = df_res[df_res["Asset"] != "USDC"]
        df_res = pd.concat([usdc_row, other_rows])

    print("-" * 50)
    print(df_res.to_string(index=False))
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=str, default="default")
    parser.add_argument("--no-risk", action="store_true")
    parser.add_argument(
        "--strategy",
        type=str,
        default="json",
        choices=["json", "ml", "llm"],
        help="View generation mode: json (manual), ml (xgboost), llm (gemini)"
    )
    args = parser.parse_args()

    run_pipeline(
        portfolio_name=args.portfolio,
        enable_risk=not args.no_risk,
        strategy_mode=args.strategy
    )