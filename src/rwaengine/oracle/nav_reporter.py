"""
Oracle reporter with EIP-191 message signing.

Converts a raw ``{ticker: weight}`` allocation into a structured,
Pydantic-validated payload, signs it with an Ethereum private key using
EIP-191 personal-sign, and returns the complete signed response ready
for on-chain submission or local archival.
"""
import time
from typing import Any, Dict

from eth_account import Account
from eth_account.messages import encode_defunct
from loguru import logger

from src.rwaengine.oracle.schemas import (
    OraclePayload,
    PortfolioAllocation,
    SignedOracleResponse,
)


class NAVReporter:
    """Builds, signs, and packages oracle payloads."""

    def __init__(self, private_key: str):
        """
        Args:
            private_key: Hex-encoded Ethereum private key (with or without
                         ``0x`` prefix).

        Raises:
            ValueError: If *private_key* is empty or ``None``.
        """
        if not private_key:
            raise ValueError("Private key is required for NAVReporter.")

        if not private_key.startswith("0x"):
            private_key = "0x" + private_key

        self._account = Account.from_key(private_key)
        logger.info(f"NAVReporter initialised. Signer: {self._account.address}")

    def generate_payload(
        self,
        portfolio_id: str,
        allocation: Dict[str, float],
        nonce: int,
    ) -> Dict[str, Any]:
        """Construct, sign, and return the oracle payload.

        Args:
            portfolio_id: Identifier for the target portfolio.
            allocation: Mapping of ``{ticker: weight}`` produced by the
                        optimizer (weights as decimals, e.g., 0.25 = 25 %).
            nonce: Anti-replay nonce (typically the current Unix timestamp).

        Returns:
            Dict representation of a ``SignedOracleResponse``, containing the
            structured payload, its EIP-191 signature, and the signer address.
        """
        logger.info("Packaging allocation for oracle transmission...")

        # Convert decimal weights to basis-point integers.
        allocations_list = [
            PortfolioAllocation(symbol=ticker, weight_bps=int(weight * 10_000))
            for ticker, weight in allocation.items()
        ]

        # Build the structured payload.
        payload = OraclePayload(
            portfolio_id=portfolio_id,
            allocations=allocations_list,
            total_assets=len(allocations_list),
            risk_verified=True,
            nonce=nonce,
            timestamp=int(time.time()),
        )

        # EIP-191 personal-sign over the canonical JSON representation.
        payload_json = payload.model_dump_json()
        message = encode_defunct(text=payload_json)
        signed_message = self._account.sign_message(message)

        logger.success(
            f"Payload signed by {self._account.address[:10]}..."
        )

        response = SignedOracleResponse(
            data=payload,
            signature=signed_message.signature.hex(),
            signer_address=self._account.address,
        )

        return response.model_dump()