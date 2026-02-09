"""
Backtesting entry point with configurable view sources.

Usage::

    uv run run_backtest.py --view-source json
    uv run run_backtest.py --view-source ml --years 5
    uv run run_backtest.py --view-source llm --years 1   # caution: API costs
"""
import argparse
import os
import shutil
import sys
from datetime import date, datetime, timedelta

import pandas as pd
from loguru import logger

# Ensure the src directory is importable before any project imports.
sys.path.append("src")

from src.rwaengine.utils.logger import setup_logger  # noqa: E402

setup_logger()

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.rwaengine.analysis.backtester import Backtester  # noqa: E402
from src.rwaengine.analysis.benchmark import BenchmarkProvider  # noqa: E402
from src.rwaengine.analysis.plotter import PerformancePlotter  # noqa: E402
from src.rwaengine.analysis.strategies import BLStrategy, MarkowitzStrategy  # noqa: E402
from src.rwaengine.data.adapters.yfinance_adapter import YFinanceAdapter  # noqa: E402
from src.rwaengine.execution.risk_manager import PortfolioRiskManager  # noqa: E402
from src.rwaengine.utils.portfolio_loader import PortfolioLoader  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def archive_results(
    portfolio_name: str,
    years: int,
    no_risk: bool,
    source: str,
) -> None:
    """Move generated plot files into a timestamped outcomes folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    risk_label = "Unconstrained" if no_risk else "RiskManaged"
    folder_name = (
        f"{timestamp}_{portfolio_name}_{years}yr_{risk_label}_{source.upper()}"
    )

    target_dir = os.path.join("outcomes", folder_name)
    os.makedirs(target_dir, exist_ok=True)

    plot_files = [
        "comparison_result.png",
        "allocation_Black-Litterman.png",
        "allocation_Markowitz_MV.png",
    ]

    logger.info(f"Archiving results to: {target_dir} ...")
    for filename in plot_files:
        if os.path.exists(filename):
            try:
                shutil.move(filename, os.path.join(target_dir, filename))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest with multiple strategy comparisons."
    )
    parser.add_argument(
        "--portfolio", type=str, default="mag_seven",
        help="Portfolio key defined in portfolios.json",
    )
    parser.add_argument(
        "--years", type=int, default=3,
        help="Number of years for the backtest window",
    )
    parser.add_argument(
        "--no-risk", action="store_true",
        help="Disable risk guardrails (unconstrained research mode)",
    )
    parser.add_argument(
        "--view-source", type=str, default="json",
        choices=["json", "ml", "llm"],
        help="Source of investor views: json (static), ml (algo), llm (Gemini)",
    )
    parser.add_argument(
        "--view-file", type=str, default="portfolios/views_backtest.json",
        help="Path to the JSON view file (only used with --view-source=json)",
    )

    args = parser.parse_args()

    # ---- Data fetching ----
    loader = PortfolioLoader()
    tickers = loader.get_tickers(args.portfolio)

    # Always include SPY (benchmark) and VIX (fear gauge) in the download.
    fetch_list = list(set(tickers + ["SPY", "^VIX"]))

    # Extra warm-up year so rolling indicators have enough history.
    warmup_years = 1
    total_fetch_days = (args.years + warmup_years) * 365
    end_date = date.today()
    start_date = end_date - timedelta(days=total_fetch_days)

    logger.info(
        f"Fetching data for [{args.view_source.upper()}] strategy..."
    )
    logger.info(
        f"Backtest: {args.years} yr | "
        f"Fetch: {args.years + warmup_years} yr (incl. warm-up)"
    )

    adapter = YFinanceAdapter()
    df_market = adapter.fetch_history(fetch_list, start_date, end_date)
    prices = (
        df_market
        .pivot(index="trade_date", columns="ticker", values="adj_close")
        .ffill()
        .dropna()
    )

    spy_prices = prices["SPY"]
    vix_prices = prices.get("^VIX")
    invest_prices = prices

    # ---- Risk manager ----
    if args.no_risk:
        logger.warning("RUNNING IN UNCONSTRAINED (NO-RISK) MODE")
        risk_manager = None
    else:
        risk_manager = PortfolioRiskManager(
            cash_buffer_pct=0.05, max_weight_pct=0.30,
        )

    # ---- Strategy definitions ----
    strategies = [
        BLStrategy(
            name="Black-Litterman",
            risk_manager=risk_manager,
            portfolio_name=args.portfolio,
            view_source=args.view_source,
            view_file=args.view_file,
        ),
        MarkowitzStrategy(name="Markowitz (MV)", risk_manager=risk_manager),
    ]

    # ---- Walk-forward backtest ----
    backtester = Backtester(prices=invest_prices, strategies=strategies)

    # Skip the first ~252 trading days (1 year) to allow warm-up.
    sim_start_idx = 252
    if len(prices) < sim_start_idx + 20:
        logger.error("Not enough data to run the backtest. Exiting.")
        return

    sim_start_date = str(prices.index[sim_start_idx].date())
    df_strategies, weights_dict = backtester.run(start_date=sim_start_date)

    # ---- Benchmark curves ----
    logger.info("Computing benchmarks...")
    common_index = df_strategies.index
    spy_cum = BenchmarkProvider.calculate_spy(spy_prices, common_index)
    eq_cum = BenchmarkProvider.calculate_equal_weight(invest_prices, common_index)

    df_final = df_strategies.copy()
    df_final["Market (SPY)"] = spy_cum
    df_final["Equal Weight"] = eq_cum

    # ---- Plotting ----
    plotter = PerformancePlotter()

    title_suffix = f" ({args.view_source.upper()} Views)"
    if args.no_risk:
        title_suffix += " [Unconstrained]"

    plotter.plot_comparison(
        df_final,
        title=f"Comparative Returns{title_suffix}",
        vix_series=vix_prices,
    )
    plotter.plot_allocation_history(weights_dict)

    # ---- Archive outputs ----
    archive_results(args.portfolio, args.years, args.no_risk, args.view_source)


if __name__ == "__main__":
    main()