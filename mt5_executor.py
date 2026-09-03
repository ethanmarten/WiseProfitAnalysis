"""
mt5_executor.py - MetaApi Cloud MT5 Execution Engine.

Connects to MT5 accounts 24/7 in the cloud via MetaApi SDK (metaapi-cloud-sdk).
Fetches market data, calculates lot sizes, executes market/limit orders, sets SL/TP,
calculates daily PnL, and auto-closes positions when profit caps ($100) are hit.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, date

# MetaApi Cloud SDK Import
try:
    from metaapi_cloud_sdk import MetaApi
except ImportError:
    MetaApi = None  # Handled gracefully if metaapi-cloud-sdk is not installed yet

logger = logging.getLogger("mt5_executor")


class MetaApiMT5Executor:
    """Manages cloud MT5 account connection and order execution via MetaApi SDK."""

    def __init__(self, api_token: str, account_id: str):
        self.api_token = api_token
        self.account_id = account_id
        self.api: Optional[Any] = None
        self.account: Optional[Any] = None
        self.connection: Optional[Any] = None

    async def initialize(self) -> bool:
        """Establishes connection to MetaApi cloud service and RPC terminal."""
        if self.connection:
            return True

        if not MetaApi or not self.api_token or not self.account_id:
            logger.info("MetaApi SDK missing or credentials unconfigured. Running in Mock execution mode.")
            return False

        try:
            self.api = MetaApi(token=self.api_token)
            self.account = await self.api.metatrader_account_api.get_account(self.account_id)

            # Deploy/start cloud account if not deployed
            if self.account.state != "DEPLOYED":
                logger.info(f"Deploying MetaApi cloud account {self.account_id}...")
                await self.account.deploy()

            logger.info(f"Waiting for MT5 cloud account {self.account_id} to connect...")
            await self.account.wait_connected()

            self.connection = self.account.get_rpc_connection()
            await self.connection.connect()
            await self.connection.wait_synchronized()

            logger.info(f"Successfully connected to MT5 Cloud Account {self.account_id}")
            return True

        except Exception as e:
            err_msg = str(e)
            if "top up your account" in err_msg.lower() or "forbidden" in err_msg.lower():
                logger.warning(
                    f"MetaApi Cloud Account {self.account_id} deployment requires MetaApi subscription top-up: {err_msg}. "
                    "Falling back to Mock Execution Mode."
                )
            else:
                logger.error(f"Failed to initialize MetaApi connection for {self.account_id}: {e}")
            return False

    async def get_account_information(self) -> Dict[str, Any]:
        """Fetches account equity, balance, margin, and leverage."""
        if not self.connection:
            return {"balance": 10000.0, "equity": 10000.0, "free_margin": 10000.0, "mock": True}

        try:
            info = await self.connection.get_account_information()
            return {
                "balance": info.get("balance", 0.0),
                "equity": info.get("equity", 0.0),
                "free_margin": info.get("freeMargin", 0.0),
                "leverage": info.get("leverage", 100),
                "currency": info.get("currency", "USD"),
                "mock": False
            }
        except Exception as e:
            logger.error(f"Error fetching account information: {e}")
            return {"balance": 0.0, "equity": 0.0, "error": str(e)}

    async def fetch_candles(self, symbol: str = "XAUUSD", timeframe: str = "1m", limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetches historical candles from MetaApi RPC connection.
        Timeframes: 1m, 5m, 15m.
        """
        if not self.connection:
            # Fallback mock candles for local testing or uninitialized API
            now = datetime.now().timestamp()
            mock_candles = []
            base_price = 4455.0
            for i in range(limit):
                mock_candles.append({
                    "time": datetime.fromtimestamp(now - (limit - i) * 60).strftime("%Y-%m-%d %H:%M:%S"),
                    "open": base_price + (i * 0.1),
                    "high": base_price + (i * 0.1) + 0.5,
                    "low": base_price + (i * 0.1) - 0.5,
                    "close": base_price + (i * 0.1) + 0.2,
                    "volume": 100 + i
                })
            return mock_candles

        try:
            # MetaApi candle fetching
            tf_map = {"1m": "1m", "5m": "5m", "15m": "15m"}
            candles = await self.connection.get_historical_candles(
                symbol, tf_map.get(timeframe, "1m"), start_time=datetime.utcnow(), limit=limit
            )
            formatted = []
            for c in candles:
                formatted.append({
                    "time": c.get("time"),
                    "open": c.get("open"),
                    "high": c.get("high"),
                    "low": c.get("low"),
                    "close": c.get("close"),
                    "volume": c.get("tickVolume", 0)
                })
            return formatted
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol} {timeframe}: {e}")
            return []

    async def get_current_price(self, symbol: str = "XAUUSD") -> float:
        """Fetches latest bid/ask price for XAUUSD."""
        if not self.connection:
            return 4455.0

        try:
            price = await self.connection.get_symbol_price(symbol)
            return (price.get("bid", 0.0) + price.get("ask", 0.0)) / 2.0
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return 4455.0

    async def execute_trade(
        self,
        symbol: str,
        action: str,  # BUY or SELL
        lots: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "Gemini_SMC_Bot"
    ) -> Dict[str, Any]:
        """
        Executes market BUY or SELL order with Stop Loss & Take Profit via MetaApi.
        """
        if action.upper() not in ["BUY", "SELL"]:
            return {"success": False, "error": f"Invalid action {action}"}

        order_type = "ORDER_TYPE_BUY" if action.upper() == "BUY" else "ORDER_TYPE_SELL"

        if not self.connection:
            logger.info(f"[MOCK TRADE] Executed {action} {lots} lots {symbol} SL:{stop_loss} TP:{take_profit}")
            return {
                "success": True,
                "position_id": f"mock_{int(datetime.now().timestamp())}",
                "order_type": action,
                "lots": lots,
                "entry_price": 4455.0,
                "mock": True
            }

        try:
            options = {}
            if stop_loss:
                options["stopLoss"] = float(stop_loss)
            if take_profit:
                options["takeProfit"] = float(take_profit)
            if comment:
                options["comment"] = comment

            result = await self.connection.create_market_buy_order(
                symbol, lots, stop_loss=stop_loss, take_profit=take_profit, options=options
            ) if action.upper() == "BUY" else await self.connection.create_market_sell_order(
                symbol, lots, stop_loss=stop_loss, take_profit=take_profit, options=options
            )

            logger.info(f"Successfully placed trade: {result}")
            return {
                "success": True,
                "position_id": str(result.get("numericCode") or result.get("orderId")),
                "order_type": action,
                "lots": lots,
                "entry_price": result.get("price", 0.0),
                "raw": result
            }

        except Exception as e:
            logger.error(f"Error executing trade via MetaApi: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Fetches all active open positions."""
        if not self.connection:
            return []

        try:
            positions = await self.connection.get_positions()
            return [
                {
                    "id": p.get("id"),
                    "symbol": p.get("symbol"),
                    "type": p.get("type"),
                    "volume": p.get("volume"),
                    "open_price": p.get("openPrice"),
                    "current_price": p.get("currentPrice"),
                    "profit": p.get("profit"),
                    "sl": p.get("stopLoss"),
                    "tp": p.get("takeProfit")
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Error fetching open positions: {e}")
            return []

    async def close_position(self, position_id: str) -> bool:
        """Closes a specific open position by position ID."""
        if not self.connection:
            logger.info(f"[MOCK CLOSE] Position {position_id} closed.")
            return True

        try:
            await self.connection.close_position(position_id)
            logger.info(f"Closed MT5 position {position_id}")
            return True
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
            return False

    async def get_today_realized_profit(self) -> float:
        """Calculates cumulative realized profit for trades closed today."""
        if not self.connection:
            return 0.0

        try:
            today_start = datetime.combine(date.today(), datetime.min.time())
            history = await self.connection.get_deals_by_time_range(today_start, datetime.utcnow())
            total_pnl = sum(deal.get("profit", 0.0) + deal.get("commission", 0.0) + deal.get("swap", 0.0) for deal in history)
            return float(total_pnl)
        except Exception as e:
            logger.error(f"Error calculating today's PnL: {e}")
            return 0.0

    async def close_all_positions_due_to_profit_cap(self) -> int:
        """Emergency closes all active positions when $100 profit cap is reached."""
        positions = await self.get_open_positions()
        closed_count = 0
        for pos in positions:
            success = await self.close_position(pos["id"])
            if success:
                closed_count += 1
        return closed_count
