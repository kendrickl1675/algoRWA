"""
File: run_backtest.py
Description: Entry point with Configurable View Sources.
Usage:
    uv run run_backtest.py --view-source json
    uv run run_backtest.py --view-source llm --years 1 (Warning: Costly!)
"""
import pandas as pd
from datetime import date, timedelta, datetime
from loguru import logger
import sys
sys.path.append("src")
from src.rwaengine.utils.logger import setup_logger
setup_logger()
import argparse
import sys
import os
import shutil
from dotenv import load_dotenv

load_dotenv()

from src.rwaengine.data.adapters.yfinance_adapter import YFinanceAdapter
from src.rwaengine.utils.portfolio_loader import PortfolioLoader
from src.rwaengine.execution.risk_manager import PortfolioRiskManager
from src.rwaengine.analysis.strategies import BLStrategy, MarkowitzStrategy
from src.rwaengine.analysis.benchmark import BenchmarkProvider
from src.rwaengine.analysis.backtester import Backtester
from src.rwaengine.analysis.plotter import PerformancePlotter

def archive_results(portfolio_name: str, years: int, no_risk: bool, source: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    risk_label = "Unconstrained" if no_risk else "RiskManaged"
    folder_name = f"{timestamp}_{portfolio_name}_{years}yr_{risk_label}_{source.upper()}"

    base_dir = "outcomes"
    target_dir = os.path.join(base_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    files_to_move = [
        "comparison_result.png",
        "allocation_Black-Litterman.png",
        "allocation_Markowitz_MV.png"
    ]

    logger.info(f"📦 Archiving results to: {target_dir} ...")
    for filename in files_to_move:
        if os.path.exists(filename):
            try:
                shutil.move(filename, os.path.join(target_dir, filename))
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=str, default="mag_seven")
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--no-risk", action="store_true", help="Research Mode")

    parser.add_argument(
        "--view-source",
        type=str,
        default="json",
        choices=["json", "ml", "llm"],
        help="Source of Investor Views: json (Static), ml (Dynamic Algo), llm (Gemini Agent)"
    )
    parser.add_argument(
        "--view-file",
        type=str,
        default="portfolios/views_backtest.json",
        help="Path to JSON view file (only used if --view-source=json)"
    )

    args = parser.parse_args()

    loader = PortfolioLoader()
    tickers = loader.get_tickers(args.portfolio)
    fetch_list = list(set(tickers + ["SPY", "^VIX"]))
    end_date = date.today()
    WARMUP_YEARS = 1
    total_fetch_days = (args.years + WARMUP_YEARS) * 365
    start_date = end_date - timedelta(days=total_fetch_days)
    logger.info(f"   Fetching Data for [{args.view_source.upper()}] Strategy...")
    logger.info(f"   Target Backtest: {args.years} years | Fetching: {args.years + WARMUP_YEARS} years (incl. warm-up)")
    adapter = YFinanceAdapter()
    df_market = adapter.fetch_history(fetch_list, start_date, end_date)
    prices = df_market.pivot(index='trade_date', columns='ticker', values='adj_close').ffill().dropna()

    spy_prices = prices["SPY"]
    vix_prices = prices.get("^VIX")
    invest_prices = prices

    if args.no_risk:
        logger.warning("☢️  RUNNING IN NO-RISK MODE ☢️")
        risk_manager = None
    else:
        risk_manager = PortfolioRiskManager(cash_buffer_pct=0.05, max_weight_pct=0.30)

    strategies = [
        BLStrategy(
            name="Black-Litterman",
            risk_manager=risk_manager,
            portfolio_name=args.portfolio,
            view_source=args.view_source, # 传入配置
            view_file=args.view_file      # 传入文件路径
        ),
        MarkowitzStrategy(name="Markowitz (MV)", risk_manager=risk_manager)
    ]

    backtester = Backtester(prices=invest_prices, strategies=strategies)
    sim_start_idx = 252
    if len(prices) < sim_start_idx + 20: return
    sim_start_date = str(prices.index[sim_start_idx].date())

    df_strategies, weights_dict = backtester.run(start_date=sim_start_date)

    logger.info("Computing Benchmarks...")
    common_index = df_strategies.index
    spy_cum = BenchmarkProvider.calculate_spy(spy_prices, common_index)
    eq_cum = BenchmarkProvider.calculate_equal_weight(invest_prices, common_index)

    df_final = df_strategies.copy()
    df_final["Market (SPY)"] = spy_cum
    df_final["Equal Weight"] = eq_cum

    plotter = PerformancePlotter()

    title_suffix = f" ({args.view_source.upper()} Views)"
    if args.no_risk: title_suffix += " [Unconstrained]"

    plotter.plot_comparison(
        df_final,
        title=f"Comparative Returns{title_suffix}",
        vix_series=vix_prices  # 新参数
    )
    plotter.plot_allocation_history(weights_dict)

    archive_results(args.portfolio, args.years, args.no_risk, args.view_source)

if __name__ == "__main__":
    main()