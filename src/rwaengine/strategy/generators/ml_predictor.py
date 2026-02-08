"""
File: src/rwaengine/strategy/generators/ml_predictor.py
Description: XGBoost Alpha Strategy (Relative Views + VIX Features).
"""
import pandas as pd
import numpy as np
from typing import List, Literal, Tuple, Optional
from loguru import logger

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from src.rwaengine.strategy.base import ViewGenerator
from src.rwaengine.strategy.types import InvestorView

class MLViewGenerator(ViewGenerator):
    def __init__(self, history_data: pd.DataFrame):
        """
        Args:
            history_data: 包含 Tickers, SPY, ^VIX 的完整历史数据
        """
        self.history = history_data
        self.lookahead_days = 5  # 预测未来 1 周的超额收益

        # 尝试提取基准数据
        self.spy_series = self.history.get("SPY")
        self.vix_series = self.history.get("^VIX")

        if self.spy_series is None or self.vix_series is None:
            logger.warning("  SPY or ^VIX missing in history! ML features will be limited.")

        self.xgb_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 150,     # 略微增加树的数量
            'max_depth': 4,          # 略微增加深度以捕捉交互特征 (VIX * Price)
            'learning_rate': 0.03,   # 降低学习率，更稳健
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'n_jobs': -1
        }

    def _add_features(self, ticker_series: pd.Series) -> pd.DataFrame:
        """
        Advanced Feature Engineering: Asset + Market (SPY) + Fear (VIX)
        """
        df = ticker_series.to_frame(name='Close')

        # === 1. Asset Technicals ===
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['vol_20'] = df['log_ret'].rolling(20).std()
        df['roc_10'] = df['Close'].pct_change(10)

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # === 2. Market Context (SPY) ===
        if self.spy_series is not None:
            spy_df = self.spy_series.to_frame(name='SPY')
            # 相对强弱 (Relative Strength)
            # 资产净值 / SPY净值 的走势
            df['rel_strength'] = df['Close'] / spy_df['SPY']
            df['rel_strength_ma20'] = df['rel_strength'].rolling(20).mean()
            # 相对动量
            df['rel_mom'] = df['rel_strength'] / df['rel_strength'].shift(10) - 1

        # === 3. Fear Gauge (VIX) ===
        if self.vix_series is not None:
            vix_df = self.vix_series.to_frame(name='VIX')
            df['vix_level'] = vix_df['VIX']
            df['vix_ma50'] = vix_df['VIX'].rolling(50).mean()
            # VIX Regime: Current vs Average
            df['vix_gap'] = df['vix_level'] - df['vix_ma50']

        # === 4. Prediction Target (Alpha) ===
        # Target = Asset_Ret_Next_5d - SPY_Ret_Next_5d
        asset_fwd_ret = df['Close'].shift(-self.lookahead_days) / df['Close'] - 1

        if self.spy_series is not None:
            spy_fwd_ret = self.spy_series.shift(-self.lookahead_days) / self.spy_series - 1
            # [Core Change] 预测目标改为超额收益 (Alpha)
            df['target_alpha'] = asset_fwd_ret - spy_fwd_ret
        else:
            # Fallback to absolute return
            df['target_alpha'] = asset_fwd_ret

        return df.dropna()

    def _train_and_predict(self, df_features: pd.DataFrame) -> Tuple[float, float]:
        """
        Train XGBoost to predict Alpha
        """
        # 排除非特征列
        exclude_cols = ['Close', 'target_alpha', 'log_ret']
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]

        X = df_features[feature_cols]
        y = df_features['target_alpha']

        # TimeSeries Cross-Validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        model = xgb.XGBRegressor(**self.xgb_params)

        for train_idx, test_idx in tscv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            # Directional Accuracy (看对方向比预测准数值更重要)
            direction_acc = np.mean(np.sign(preds) == np.sign(y.iloc[test_idx]))
            scores.append(direction_acc)

        confidence = np.mean(scores) if scores else 0.5

        # Full Retrain & Predict
        model.fit(X, y)
        latest = X.iloc[[-1]]
        pred_alpha = model.predict(latest)[0]

        return pred_alpha, confidence

    def _amplify_signal(self, pred_alpha: float, volatility: float) -> float:
        """
        [Signal Enhancement v2]
        Input: Predicted Alpha (Weekly)
        Output: Annualized Absolute View
        """
        # 1. 噪音过滤
        if abs(pred_alpha) < 0.002: # 如果超额收益 < 0.2% (周), 忽略
            return 0.0

        # 2. 转换逻辑
        # 如果 Alpha > 0 (跑赢大盘): View = Base_Bull (15%) + Alpha_Boost
        # 如果 Alpha < 0 (跑输大盘): View = Base_Bear (-15%) + Alpha_Boost

        direction = np.sign(pred_alpha)

        # 基础观点 (锚定): 15% 年化 (足以战胜 4% 无风险利率)
        base_view = 0.15

        # 波动率补偿: 波动越大的资产，需要的 View 幅度越大才能改变权重
        vol_adj = (volatility * np.sqrt(252)) * 0.5

        # 最终观点 = 方向 * (基准 + 波动补偿)
        final_view = direction * (base_view + vol_adj)

        return final_view

    def generate_views(self, current_prices: pd.Series) -> List[InvestorView]:
        tickers = [t for t in current_prices.index if t not in ['SPY', '^VIX']]
        views = []

        logger.info(f"  ML Alpha Strategy (XGBoost): Analyzing {len(tickers)} assets vs SPY...")

        for ticker in tickers:
            if ticker not in self.history.columns: continue

            try:
                # 1. Prep Data
                df = self._add_features(self.history[ticker])
                if len(df) < 100: continue

                # 2. Predict Alpha
                pred_alpha, conf_score = self._train_and_predict(df)

                # 3. Construct View
                # 获取当前波动率用于校准
                curr_vol = df['vol_20'].iloc[-1]
                bl_view_return = self._amplify_signal(pred_alpha, curr_vol)

                if bl_view_return == 0.0: continue

                # 置信度映射
                final_conf = min(0.4 + conf_score, 0.90) # Base confidence raised

                view = InvestorView(
                    assets=[ticker],
                    weights=[1.0],
                    expected_return=bl_view_return,
                    confidence=final_conf,
                    description=f"AlphaPred: {pred_alpha*100:.2f}% (vs SPY) | VIX_Adj View"
                )
                views.append(view)

            except Exception as e:
                logger.error(f"ML failed for {ticker}: {e}")

        logger.success(f"Generated {len(views)} Alpha Views.")
        return views