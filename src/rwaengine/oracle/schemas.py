"""
File: src/rwaengine/oracle/schemas.py
Description: Pydantic models for Oracle JSON payloads.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import time


class PortfolioAllocation(BaseModel):
    """
    单一资产的分配情况
    """
    symbol: str
    weight_bps: int = Field(..., description="权重基点 (1% = 100 bps)")


class OraclePayload(BaseModel):
    """
    发送给预言机的最终载荷 (EIP-191 Signed Data)
    """
    portfolio_id: str
    nonce: int = Field(..., description="唯一随机数/时间戳，防止重放攻击")
    timestamp: int = Field(default_factory=lambda: int(time.time()), description="生成时间")

    allocations: List[PortfolioAllocation]
    total_assets: int = Field(..., description="资产数量校验")

    risk_verified: bool = Field(True, description="是否通过风控层")


class SignedOracleResponse(BaseModel):
    """
    包含签名的完整响应包 (Off-Chain -> On-Chain)
    """
    data: OraclePayload
    signature: str = Field(..., description="0x开头的十六进制签名 (EIP-191)")
    signer_address: str = Field(..., description="签名者的公钥地址")