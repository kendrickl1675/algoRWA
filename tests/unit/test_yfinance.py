"""
test_yfinance.py
"""
from datetime import date, timedelta
from src.rwaengine.data.adapters.yfinance_adapter import YFinanceAdapter


def test_run():
    # 1. 实例化 Adapter
    adapter = YFinanceAdapter(proxy=None)

    # 2. 设置时间窗口
    today = date.today()
    start_date = today - timedelta(days=5)

    print(f"Testing Fetch Range: {start_date} -> {today}")

    try:
        df = adapter.fetch_history(
            tickers=["AAPL", "GOOGL"],  # 这里的代码通常不会退市
            start_date=start_date,
            end_date=today
        )

        if df.empty:
            print("❌ Result is EMPTY. Check network or market holiday.")
        else:
            print("✅ Data Fetched Successfully:")
            print(df.head())
            print("\nColumns:", df.columns.tolist())
            print("\nIndex:", df.index.names)

    except Exception as e:
        print(f"❌ Test Failed with error: {e}")


if __name__ == "__main__":
    test_run()