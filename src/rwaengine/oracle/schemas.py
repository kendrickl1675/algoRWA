
"""
Pydantic schemas for the on-chain oracle payload.

These models define the exact JSON structure that is serialized,
EIP-191 signed, and transmitted to the smart-contract oracle.
Strict typing ensures that a malformed payload can never reach the
signing step.
"""
import time
from typing import List

from pydantic import BaseModel, Field


class PortfolioAllocation(BaseModel):
    """Weight of a single asset expressed in basis points."""

    symbol: str
    weight_bps: int = Field(
        ...,
        description="Portfolio weight in basis points (1 % = 100 bps)",
    )


class OraclePayload(BaseModel):
    """Unsigned data body sent to the on-chain oracle.

    The ``nonce`` and ``timestamp`` fields together prevent replay attacks:
    the smart contract should reject any payload whose nonce has already
    been consumed or whose timestamp is too far in the past.
    """

    portfolio_id: str
    nonce: int = Field(
        ...,
        description="Unique nonce (typically a Unix timestamp) to prevent replay",
    )
    timestamp: int = Field(
        default_factory=lambda: int(time.time()),
        description="Payload generation time (Unix epoch seconds)",
    )
    allocations: List[PortfolioAllocation]
    total_assets: int = Field(
        ...,
        description="Asset count for on-chain integrity check",
    )
    risk_verified: bool = Field(
        True,
        description="Whether the allocation passed risk guardrails",
    )


class SignedOracleResponse(BaseModel):
    """Complete response envelope including the EIP-191 signature.

    This is the final artifact archived locally and optionally relayed
    to the on-chain oracle contract.
    """

    data: OraclePayload
    signature: str = Field(
        ...,
        description="Hex-encoded EIP-191 signature over the JSON payload",
    )
    signer_address: str = Field(
        ...,
        description="Ethereum address of the signing key",
    )