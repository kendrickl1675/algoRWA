
"""
XGBoost-based alpha view generator.

Trains a per-asset XGBoost regressor to predict the weekly excess return
(alpha) of each ticker relative to SPY, incorporating VIX-derived fear
features.  Predicted alphas are converted to annualised absolute views
suitable for the Black-Litterman model.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
from loguru import logger

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from src.rwaengine.strategy.base import ViewGenerator
from src.rwaengine.strategy.types import InvestorView


class MLViewGenerator(ViewGenerator):
    """Generates investor views via XGBoost alpha prediction."""

    # Weekly forward return below this threshold is treated as noise.
    NOISE_THRESHOLD = 0.002  # 0.2 %

    def __init__(self, history_data: pd.DataFrame):
        """
        Args:
            history_data: Wide-format DataFrame containing asset tickers plus
                          ``SPY`` and ``^VIX`` columns used as market context.
        """
        self.history = history_data

        # Prediction horizon in trading days (≈ 1 week).
        self.lookahead_days = 5

        self.spy_series = self.history.get("SPY")
        self.vix_series = self.history.get("^VIX")

        if self.spy_series is None or self.vix_series is None:
            logger.warning(
                "SPY or ^VIX missing in history — ML features will be limited."
            )

        self.xgb_params = {
            "objective": "reg:squarederror",
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "n_jobs": -1,
        }

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _add_features(self, ticker_series: pd.Series) -> pd.DataFrame:
        """Build a feature matrix from a single asset's price series.

        Feature groups:
          1. **Asset technicals** — log return, 20-day vol, 10-day ROC, RSI-14.
          2. **Market context** — relative strength vs SPY, relative momentum.
          3. **Fear gauge** — VIX level, 50-day VIX MA, VIX gap.
          4. **Target** — forward 5-day alpha (excess return over SPY).

        Returns:
            DataFrame with all NaN rows dropped (due to rolling windows and
            the forward-looking target).
        """
        df = ticker_series.to_frame(name="Close")

        # 1. Asset technicals
        df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["vol_20"] = df["log_ret"].rolling(20).std()
        df["roc_10"] = df["Close"].pct_change(10)

        # RSI-14
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # 2. Market context (relative to SPY)
        if self.spy_series is not None:
            spy_df = self.spy_series.to_frame(name="SPY")
            df["rel_strength"] = df["Close"] / spy_df["SPY"]
            df["rel_strength_ma20"] = df["rel_strength"].rolling(20).mean()
            df["rel_mom"] = df["rel_strength"] / df["rel_strength"].shift(10) - 1

        # 3. Fear gauge (VIX)
        if self.vix_series is not None:
            vix_df = self.vix_series.to_frame(name="VIX")
            df["vix_level"] = vix_df["VIX"]
            df["vix_ma50"] = vix_df["VIX"].rolling(50).mean()
            df["vix_gap"] = df["vix_level"] - df["vix_ma50"]

        # 4. Prediction target: forward excess return vs SPY
        asset_fwd_ret = (
            df["Close"].shift(-self.lookahead_days) / df["Close"] - 1
        )

        if self.spy_series is not None:
            spy_fwd_ret = (
                self.spy_series.shift(-self.lookahead_days) / self.spy_series - 1
            )
            df["target_alpha"] = asset_fwd_ret - spy_fwd_ret
        else:
            # Fall back to absolute return when SPY is unavailable.
            df["target_alpha"] = asset_fwd_ret

        return df.dropna()

    # ------------------------------------------------------------------
    # Training & prediction
    # ------------------------------------------------------------------

    def _train_and_predict(
        self, df_features: pd.DataFrame
    ) -> Tuple[float, float]:
        """Train XGBoost on time-series CV folds and return the latest
        predicted alpha together with a directional-accuracy confidence score.

        Returns:
            Tuple of (predicted_alpha, confidence) where confidence is the
            mean directional accuracy across CV folds (0–1 scale).
        """
        exclude_cols = {"Close", "target_alpha", "log_ret"}
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]

        X = df_features[feature_cols]
        y = df_features["target_alpha"]

        # Evaluate directional accuracy via expanding-window time-series CV.
        tscv = TimeSeriesSplit(n_splits=3)
        scores: List[float] = []
        model = xgb.XGBRegressor(**self.xgb_params)

        for train_idx, test_idx in tscv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            # Directional accuracy matters more than point-estimate error.
            direction_acc = float(
                np.mean(np.sign(preds) == np.sign(y.iloc[test_idx]))
            )
            scores.append(direction_acc)

        confidence = np.mean(scores) if scores else 0.5

        # Full retrain on all available data, then predict the latest row.
        model.fit(X, y)
        pred_alpha = float(model.predict(X.iloc[[-1]])[0])

        return pred_alpha, confidence

    # ------------------------------------------------------------------
    # Signal calibration
    # ------------------------------------------------------------------

    def _amplify_signal(self, pred_alpha: float, volatility: float) -> float:
        """Convert a raw weekly alpha prediction into an annualised absolute
        view suitable for the Black-Litterman model.

        The mapping works as follows:
          - Signals below the noise threshold are discarded (returns 0).
          - A base annual view of 15 % (enough to beat 4 % risk-free) is
            assigned in the predicted direction.
          - A volatility adjustment is added so that higher-vol assets
            produce proportionally larger views (otherwise BL effectively
            ignores weak views on volatile assets).

        Args:
            pred_alpha: Predicted weekly excess return vs SPY.
            volatility: Current 20-day log-return standard deviation.

        Returns:
            Annualised absolute view, or 0.0 if the signal is noise.
        """
        if abs(pred_alpha) < self.NOISE_THRESHOLD:
            return 0.0

        direction = np.sign(pred_alpha)

        # Anchor: 15 % annualised is meaningful vs the 4 % risk-free rate.
        base_view = 0.15

        # Scale up for volatile assets so the view actually moves BL weights.
        vol_adj = (volatility * np.sqrt(252)) * 0.5

        return float(direction * (base_view + vol_adj))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_views(self, current_prices: pd.Series) -> List[InvestorView]:
        """Produce a list of investor views for all investable tickers.

        Args:
            current_prices: Series indexed by ticker with the latest prices.

        Returns:
            List of ``InvestorView`` objects (one per asset with a
            non-trivial alpha signal).
        """
        tickers = [t for t in current_prices.index if t not in ("SPY", "^VIX")]
        views: List[InvestorView] = []

        logger.info(
            f"ML Alpha Strategy (XGBoost): analysing {len(tickers)} assets vs SPY..."
        )

        for ticker in tickers:
            if ticker not in self.history.columns:
                continue

            try:
                df = self._add_features(self.history[ticker])
                if len(df) < 100:
                    continue

                pred_alpha, conf_score = self._train_and_predict(df)

                curr_vol = df["vol_20"].iloc[-1]
                bl_view_return = self._amplify_signal(pred_alpha, curr_vol)

                if bl_view_return == 0.0:
                    continue

                # Map CV directional accuracy to a BL confidence in [0.4, 0.9].
                final_conf = min(0.4 + conf_score, 0.90)

                views.append(
                    InvestorView(
                        assets=[ticker],
                        weights=[1.0],
                        expected_return=bl_view_return,
                        confidence=final_conf,
                        description=(
                            f"AlphaPred: {pred_alpha * 100:.2f}% (vs SPY) "
                            f"| VIX-adjusted view"
                        ),
                    )
                )

            except Exception as e:
                logger.error(f"ML prediction failed for {ticker}: {e}")

        logger.success(f"Generated {len(views)} alpha views.")
        return views