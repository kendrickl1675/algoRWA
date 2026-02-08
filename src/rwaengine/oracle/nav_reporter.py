"""
File: src/rwaengine/oracle/nav_reporter.py
Description: Oracle Reporter with EIP-191 Signing Capability (Schema Enforced).
"""
import time
from typing import Dict, Any, List
from eth_account import Account
from eth_account.messages import encode_defunct
from loguru import logger

from src.rwaengine.oracle.schemas import OraclePayload, PortfolioAllocation, SignedOracleResponse

class NAVReporter:
    def __init__(self, private_key: str):
        """
        Initialize with an Ethereum private key for signing.
        """
        if not private_key:
            raise ValueError("Private key is required for NAVReporter")

        if not private_key.startswith("0x"):
            private_key = "0x" + private_key

        self._account = Account.from_key(private_key)
        logger.info(f"NAVReporter initialized. Signer: {self._account.address}")

    def generate_payload(self, portfolio_id: str, allocation: Dict[str, float], nonce: int) -> Dict[str, Any]:
        """
        Constructs a Typed Payload (via Pydantic), signs it, and returns the result.

        Args:
            portfolio_id: Portfolio Identifier
            allocation: Raw dictionary {ticker: weight} from Engine
            nonce: Timestamp nonce from Main pipeline

        Returns:
            Dict representation of SignedOracleResponse
        """
        logger.info("Packaging data for Oracle transmission (Schema Mode)...")


        allocations_list = []
        for ticker, weight in allocation.items():
            bps = int(weight * 10000)
            allocations_list.append(PortfolioAllocation(symbol=ticker, weight_bps=bps))


        payload = OraclePayload(
            portfolio_id=portfolio_id,
            allocations=allocations_list,
            total_assets=len(allocations_list),
            risk_verified=True,
            nonce=nonce,
            timestamp=int(time.time())
        )


        payload_json = payload.model_dump_json()

        message = encode_defunct(text=payload_json)
        signed_message = self._account.sign_message(message)

        logger.success(f"✍️ Payload signed by {self._account.address[:6]}...")


        response = SignedOracleResponse(
            data=payload,
            signature=signed_message.signature.hex(),
            signer_address=self._account.address
        )

        return response.model_dump()