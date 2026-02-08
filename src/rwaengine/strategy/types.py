"""
File: src/rwaengine/strategy/types.py
Description: Data Contracts for Black-Litterman Views.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

class StrategyConfig(BaseModel):
    """
    通用策略配置
    """
    risk_aversion: float = Field(2.5, gt=0, description="风险厌恶系数 (Delta)")
    tau: float = Field(0.05, gt=0, description="不确定性缩放系数")

class InvestorView(BaseModel):
    """
    单一观点的数据容器
    例如: "AAPL 将跑赢 GOOGL 2%"
    """
    assets: List[str] = Field(..., min_length=1, description="涉及的资产列表")
    weights: List[float] = Field(..., min_length=1, description="资产在观点中的权重 (P矩阵的一行)")
    expected_return: float = Field(..., description="预期超额收益 (Q向量的一个元素)")
    confidence: float = Field(1.0, gt=0, le=1.0, description="观点置信度 (0-100%)")

    description: Optional[str] = Field(None, description="观点的文本描述/理由")

    @field_validator('weights')
    @classmethod
    def check_weights_match_assets(cls, v: List[float], info):
        if 'assets' in info.data and len(v) != len(info.data['assets']):
            raise ValueError("Weights count must match Assets count")
        return v

class OptimizationResult(BaseModel):
    """
    最终输出的仓位建议
    """
    tickers: List[str]
    weights: List[float]
    expected_return: float
    volatility: float
    sharpe_ratio: float