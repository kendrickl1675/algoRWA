"""
File: src/rwaengine/strategy/generators/ml_predictor.py
Description: ML-based Strategy Generator adapted from machine_learning_strategies.py
"""
import pandas as pd
import numpy as np
from typing import List, Literal
from loguru import logger

# ML 依赖
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

from src.rwaengine.strategy.base import ViewGenerator
from src.rwaengine.strategy.types import InvestorView

ModelType = Literal['Linear Regression', 'Random Forest', 'Gradient Boosting']

class MLViewGenerator(ViewGenerator):
    def __init__(self, history_data: pd.DataFrame, model_type: ModelType = 'Linear Regression'):
        """
        Args:
            history_data: 完整的历史价格数据 (Index: Date, Columns: Tickers)
            model_type: 选择的模型类型
        """
        self.history = history_data
        self.model_type = model_type
        self.scaler = StandardScaler()

    def _create_features(self, price_series: pd.Series, lag_days: int = 5) -> pd.DataFrame:
        """
        特征工程：移动平均 + 滞后特征
        """
        df = price_series.to_frame(columns=['Close'])

        df['5d_rolling_avg'] = df['Close'].rolling(window=5).mean()
        df['10d_rolling_avg'] = df['Close'].rolling(window=10).mean()

        for i in range(1, lag_days + 1):
            df[f'lag_{i}'] = df['Close'].shift(i)

        # 清洗 NaN (由于 rolling 和 shift 产生)
        df.dropna(inplace=True)
        return df

    def _get_model_instance(self):
        """工厂方法获取模型实例"""
        if self.model_type == 'Random Forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'Gradient Boosting':
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'Linear Regression':
            return LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def generate_views(self, current_prices: pd.Series) -> List[InvestorView]:
        """
        对每个资产单独训练模型并生成观点
        """
        logger.info(f"Training {self.model_type} models for {len(current_prices)} assets...")
        views = []
        tickers = current_prices.index.tolist()

        for ticker in tickers:
            if ticker not in self.history.columns:
                logger.warning(f"Skipping {ticker}: No history data available.")
                continue

            try:
                ticker_data = self.history[ticker]
                df_features = self._create_features(ticker_data)

                if len(df_features) < 30:
                    continue

                X = df_features.drop(columns=['Close'])
                y = df_features['Close']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                imputer = SimpleImputer(strategy='mean')
                X_train_imputed = imputer.fit_transform(X_train)
                X_test_imputed = imputer.transform(X_test)

                X_train_scaled = self.scaler.fit_transform(X_train_imputed)
                X_test_scaled = self.scaler.transform(X_test_imputed)

                model = self._get_model_instance()
                model.fit(X_train_scaled, y_train)

                score = model.score(X_test_scaled, y_test)

                if score <= 0.05:
                    continue

                confidence = min(max(score, 0.0), 1.0)

                latest_features_row = df_features.iloc[[-1]].drop(columns=['Close'])
                latest_features_imputed = imputer.transform(latest_features_row)
                latest_features_scaled = self.scaler.transform(latest_features_imputed)

                predicted_price = model.predict(latest_features_scaled)[0]
                current_price = current_prices[ticker]

                expected_return = (predicted_price - current_price) / current_price

                if abs(expected_return) > 0.005:
                    direction = 1.0 if expected_return > 0 else -1.0

                    view = InvestorView(
                        assets=[ticker],
                        weights=[direction],  # Long or Short
                        expected_return=abs(expected_return), # BL 模型通常接受绝对值 magnitude
                        confidence=confidence,
                        description=f"ML Prediction ({self.model_type}): R2={confidence:.2f}"
                    )
                    views.append(view)

            except Exception as e:
                logger.error(f"ML training failed for {ticker}: {e}")

        logger.success(f"Generated {len(views)} views from ML models.")
        return views