"""
Production pipeline entry point.

Runs a single-shot optimisation cycle:
  1. Fetch market data for the configured portfolio.
  2. Generate investor views using the selected strategy mode.
  3. Run Black-Litterman optimisation.
  4. Optionally apply risk guardrails.
  5. Produce and sign an oracle payload for on-chain submission.
  6. Archive all outputs to a timestamped folder.

Usage::

    uv run main.py --portfolio mag_seven --strategy json
    uv run main.py --portfolio mag_seven --strategy ml --no-risk
"""
import argparse
import json
import os
import shutil
import sys
from datetime import date, datetime, timedelta

from dotenv import load_dotenv
from loguru import logger

from src.rwaengine.utils.logger import setup_logger

setup_logger()
load_dotenv()

from src.rwaengine.core.engine import BlackLittermanEngine  # noqa: E402
from src.rwaengine.data.adapters.yfinance_adapter import YFinanceAdapter  # noqa: E402
from src.rwaengine.execution.risk_manager import PortfolioRiskManager  # noqa: E402
from src.rwaengine.oracle.nav_reporter import NAVReporter  # noqa: E402
from src.rwaengine.strategy.factory import StrategyFactory  # noqa: E402
from src.rwaengine.utils.portfolio_loader import PortfolioLoader  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_outcome_dir(
    portfolio: str,
    strategy: str,
    no_risk: bool,
) -> str:
    """Create and return a timestamped output directory under ``outcomes/``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    risk_tag = "_Unconstrained" if no_risk else ""
    folder_name = (
        f"{timestamp}_{portfolio}_{strategy.upper()}_Production{risk_tag}"
    )

    target_dir = os.path.join("outcomes", folder_name)
    os.makedirs(target_dir, exist_ok=True)

    logger.info(f"Output directory created: {target_dir}")
    return target_dir


def save_json(data: dict, folder: str, filename: str) -> None:
    """Serialise *data* as pretty-printed JSON into *folder*/*filename*."""
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.success(f"Saved {filename}")


def archive_current_log(target_dir: str) -> None:
    """Copy today's log file into *target_dir* for post-mortem analysis."""
    log_dir = "logs"
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"rwa_engine_{today_str}.log"
        src_log = os.path.join(log_dir, log_filename)

        if os.path.exists(src_log):
            dst_log = os.path.join(target_dir, "execution.log")
            shutil.copy2(src_log, dst_log)
            logger.info(f"Archived execution log to {dst_log}")
        else:
            logger.warning("Log file not found for archiving.")
    except Exception as e:
        logger.warning(f"Failed to archive log: {e}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RWA Quant Engine — Production Pipeline",
    )
    parser.add_argument(
        "--portfolio", type=str, required=True,
        help="Portfolio ID (e.g. mag_seven)",
    )
    parser.add_argument(
        "--strategy", type=str, default="json",
        choices=["json", "ml", "llm"],
        help="Strategy mode for generating investor views",
    )
    parser.add_argument(
        "--no-risk", action="store_true",
        help="Disable risk guardrails (research only)",
    )
    args = parser.parse_args()

    output_dir = create_outcome_dir(args.portfolio, args.strategy, args.no_risk)

    try:
        logger.info(
            f"Starting engine for [{args.portfolio}] "
            f"using [{args.strategy.upper()}] strategy..."
        )

        # ---- Load tickers & fetch market data ----
        loader = PortfolioLoader()
        tickers = loader.get_tickers(args.portfolio)
        logger.info(f"Target assets: {tickers}")

        lookback_days = 365
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        adapter = YFinanceAdapter()
        fetch_list = list(set(tickers + ["SPY", "^VIX"]))

        df_market = adapter.fetch_history(
            fetch_list, start_date=start_date, end_date=end_date,
        )
        prices = (
            df_market
            .pivot(index="trade_date", columns="ticker", values="adj_close")
            .ffill()
            .dropna()
        )

        invest_prices = prices[tickers]
        current_prices = invest_prices.iloc[-1]

        logger.info(
            f"Market data loaded. Reference date: {current_prices.name.date()}"
        )

        # ---- Generate investor views ----
        factory_kwargs = {
            "portfolio_name": args.portfolio,
            "api_key": os.getenv("GEMINI_API_KEY"),
            "history_data": prices,
        }

        generator = StrategyFactory.get_generator(args.strategy, **factory_kwargs)
        views = generator.generate_views(current_prices)

        # Persist views for debugging / audit.
        views_debug = [
            {
                "assets": v.assets,
                "expected_return": v.expected_return,
                "confidence": v.confidence,
                "description": v.description,
            }
            for v in views
        ]
        save_json(views_debug, output_dir, "strategy_views.json")

        # ---- Black-Litterman optimisation ----
        engine = BlackLittermanEngine(prices=invest_prices)
        market_caps = {t: 1e12 for t in tickers}
        raw_result = engine.run_optimization(market_caps, views)

        # ---- Risk guardrails (optional) ----
        if args.no_risk:
            logger.warning("SKIPPING RISK MANAGEMENT (Unconstrained Mode)")
            final_allocation = dict(
                zip(raw_result.tickers, raw_result.weights)
            )
        else:
            logger.info("Applying risk guardrails...")
            risk_manager = PortfolioRiskManager(
                cash_buffer_pct=0.05, max_weight_pct=0.30,
            )
            risk_result = risk_manager.apply_guardrails(raw_result)
            final_allocation = dict(
                zip(risk_result.tickers, risk_result.weights)
            )

        # ---- Sign and save oracle payload ----
        logger.info("Signing oracle payload...")
        reporter = NAVReporter(private_key=os.getenv("RWA_SIGNER_KEY"))
        payload = reporter.generate_payload(
            portfolio_id=args.portfolio,
            allocation=final_allocation,
            nonce=int(datetime.now().timestamp()),
        )
        save_json(payload, output_dir, "oracle_payload.json")

        logger.success("-" * 30)
        logger.success("PRODUCTION RUN COMPLETE")
        logger.success(f"Results archived to: {output_dir}")
        logger.success("-" * 30)

        archive_current_log(output_dir)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        archive_current_log(output_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()