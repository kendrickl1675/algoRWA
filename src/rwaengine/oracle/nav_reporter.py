"""
File: src/rwaengine/oracle/nav_reporter.py
Description: Converts optimization results into signed Web3-ready payloads.
"""
import json
from eth_account import Account
from eth_account.messages import encode_defunct
from loguru import logger
from typing import List

from src.rwaengine.strategy.types import OptimizationResult
from src.rwaengine.oracle.schemas import OraclePayload, PortfolioAllocation, SignedOracleResponse


class NAVReporter:
    def __init__(self, private_key: str, portfolio_id: str = "default"):
        """
        Args:
            private_key: 用于签名的以太坊私钥 (Hex string)
            portfolio_id: 当前组合的标识符
        """
        self.portfolio_id = portfolio_id
        self._account = Account.from_key(private_key)
        logger.info(f"NAVReporter initialized. Signer: {self._account.address}")

    def package_results(self, result: OptimizationResult) -> SignedOracleResponse:
        """
        将优化结果打包、转换精度并签名。
        """
        logger.info("Packaging data for Oracle transmission...")

        # 1.0 = 10000 bps, 0.01 = 100 bps
        allocations = []
        total_bps = 0

        for ticker, weight in zip(result.tickers, result.weights):
            bps = int(weight * 10000)
            allocations.append(PortfolioAllocation(symbol=ticker, weight_bps=bps))
            total_bps += bps

        if abs(total_bps - 10000) > 5:
            logger.warning(f"BPS Sum deviation detected: {total_bps}/10000")

        payload = OraclePayload(
            portfolio_id=self.portfolio_id,
            allocations=allocations,
            total_assets=len(allocations),
            risk_verified=True
        )


        payload_json = payload.model_dump_json()

        # 使用 encode_defunct 包装消息头 "\x19Ethereum Signed Message:\n..."
        message = encode_defunct(text=payload_json)
        signed_message = self._account.sign_message(message)

        logger.success("Payload signed successfully.")

        return SignedOracleResponse(
            data=payload,
            signature=signed_message.signature.hex(),
            signer_address=self._account.address
        )

    def export_json(self, response: SignedOracleResponse, filename: str = "oracle_output.json"):
        """辅助方法：写入磁盘供调试或上传"""
        with open(filename, "w") as f:
            f.write(response.model_dump_json(indent=2))
        logger.info(f"Oracle JSON exported to {filename}")