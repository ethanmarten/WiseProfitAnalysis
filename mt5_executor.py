"""
mt5_executor.py - MetaApi Cloud MT5 Execution Engine.

Connects to MT5 accounts 24/7 in the cloud via MetaApi SDK (metaapi-cloud-sdk).
Fetches market data, calculates lot sizes, executes market/limit orders, sets SL/TP,
calculates daily PnL, and auto-closes positions when profit caps ($100) are hit.

Safety model:
  * When the SDK or credentials are unavailable the executor enters MOCK mode.
  * Mock mode NEVER reports fake successful trades unless ALLOW_MOCK_TRADING=true,
    so a misconfigured production deploy fails loudly instead of pretending to trade.
  * Live connections are cached per account so the 60s engine loop reuses one
    synchronized RPC connection rather than redeploying the account each cycle.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, date, timezone

# MetaApi Cloud SDK Import
try:
    from metaapi_cloud_sdk import MetaApi
except ImportError:
    MetaApi = None  # Handled gracefully if metaapi-cloud-sdk is not installed yet

logger = logging.getLogger("mt5_executor")

# Mock trading is opt-in. Without it, a disconnected executor refuses to trade.
ALLOW_MOCK_TRADING = os.getenv("ALLOW_MOCK_TRADING", "false").lower() == "true"

# Cache of live executors keyed by MetaApi account id, so repeated engine cycles
# and dashboard requests share one synchronized connection.
_EXECUTOR_CACHE: Dict[str, "MetaApiMT5Executor"] = {}
_CACHE_LOCK = asyncio.Lock()


def _utcnow() -> datetime:
    """Naive UTC timestamp (datetime.utcnow is deprecated in Python 3.12)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def get_executor(api_token: str, account_id: str) -> "MetaApiMT5Executor":
    """Returns a shared, initialized executor for the given MetaApi account."""
    async with _CACHE_LOCK:
        executor = _EXECUTOR_CACHE.get(account_id)
        if executor is None:
            executor = MetaApiMT5Executor(api_token, account_id)
            _EXECUTOR_CACHE[account_id] = executor
        elif executor.api_token != api_token:
            # Credentials rotated: drop the stale connection.
            executor = MetaApiMT5Executor(api_token, account_id)
            _EXECUTOR_CACHE[account_id] = executor
    await executor.initialize()
    return executor


class MetaApiMT5Executor:
    """Manages cloud MT5 account connection and order execution via MetaApi SDK."""

    def __init__(self, api_token: str, account_id: str):
        self.api_token = api_token
        self.account_id = account_id
        self.api: Optional[Any] = None
        self.account: Optional[Any] = None
        self.connection: Optional[Any] = None
        self.last_error: Optional[str] = None
        self._init_lock = asyncio.Lock()

    @property
    def is_live(self) -> bool:
        """True when a real synchronized MetaApi connection is available."""
        return self.connection is not None

    async def initialize(self) -> bool:
        """Establishes connection to MetaApi cloud service and RPC terminal."""
        if self.connection:
            return True

        async with self._init_lock:
            if self.connection:
                return True

            if not MetaApi:
                self.last_error = "metaapi-cloud-sdk is not installed."
                logger.warning(f"{self.last_error} Running in MOCK mode.")
                return False

            if not self.api_token or not self.account_id:
                self.last_error = "MetaApi token or account id is missing (decryption may have failed)."
                logger.warning(f"{self.last_error} Running in MOCK mode.")
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

                self.last_error = None
                logger.info(f"Successfully connected to MT5 Cloud Account {self.account_id}")
                return True

            except Exception as e:
                err_msg = str(e)
                self.last_error = err_msg
                self.connection = None
                if "top up your account" in err_msg.lower() or "forbidden" in err_msg.lower():
                    logger.warning(
                        f"MetaApi Cloud Account {self.account_id} deployment requires a MetaApi subscription "
                        f"top-up: {err_msg}. Falling back to MOCK mode (no trades will be placed)."
                    )
                else:
                    logger.error(f"Failed to initialize MetaApi connection for {self.account_id}: {e}")
                return False

    async def get_account_information(self) -> Dict[str, Any]:
        """Fetches account equity, balance, margin, and leverage."""
        if not self.connection:
            return {
                "balance": 0.0,
                "equity": 0.0,
                "free_margin": 0.0,
                "mock": True,
                "error": self.last_error or "Not connected to MetaApi.",
            }

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
        Fetches historical candles from MetaApi.
        Timeframes: 1m, 5m, 15m.

        Historical candles live on the MetatraderAccount object (not the RPC
        connection) in metaapi-cloud-sdk, so the account is tried first and the
        connection is used as a fallback for older SDK versions.
        """
        if not self.connection:
            # No live connection: return nothing so callers fall back to a real
            # public data source instead of trading on fabricated candles.
            return []

        tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h"}
        tf = tf_map.get(timeframe, "1m")

        raw_candles = None
        for source, kwargs in (
            (self.account, {"start_time": None, "limit": limit}),
            (self.connection, {"start_time": None, "limit": limit}),
        ):
            if source is None or not hasattr(source, "get_historical_candles"):
                continue
            try:
                raw_candles = await source.get_historical_candles(symbol, tf, **kwargs)
                if raw_candles:
                    break
            except TypeError:
                # Older/newer signature without keyword arguments.
                try:
                    raw_candles = await source.get_historical_candles(symbol, tf, None, limit)
                    if raw_candles:
                        break
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"Positional candle fetch failed on {type(source).__name__}: {e}")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Candle fetch failed on {type(source).__name__}: {e}")

        if not raw_candles:
            logger.warning(f"No historical candles returned for {symbol} {tf} from MetaApi.")
            return []

        formatted = []
        for c in raw_candles:
            time_value = c.get("time")
            formatted.append({
                "time": time_value.isoformat() if hasattr(time_value, "isoformat") else time_value,
                "open": c.get("open"),
                "high": c.get("high"),
                "low": c.get("low"),
                "close": c.get("close"),
                "volume": c.get("tickVolume", 0)
            })
        return formatted

    async def get_current_price(self, symbol: str = "XAUUSD") -> float:
        """Fetches latest bid/ask price. Returns 0.0 when unavailable."""
        if not self.connection:
            return 0.0

        try:
            price = await self.connection.get_symbol_price(symbol)
            bid = float(price.get("bid") or 0.0)
            ask = float(price.get("ask") or 0.0)
            if bid and ask:
                return (bid + ask) / 2.0
            return bid or ask or 0.0
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return 0.0

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

        if lots <= 0:
            return {"success": False, "error": "Lot size must be greater than zero."}

        if not self.connection:
            if not ALLOW_MOCK_TRADING:
                # Fail loudly: a disconnected account must not report success.
                msg = (
                    f"Refusing to execute {action} {symbol}: no live MetaApi connection "
                    f"({self.last_error or 'not initialized'}). "
                    "Set ALLOW_MOCK_TRADING=true to simulate trades in development."
                )
                logger.error(msg)
                return {"success": False, "error": msg, "mock": True}

            logger.info(f"[MOCK TRADE] Executed {action} {lots} lots {symbol} SL:{stop_loss} TP:{take_profit}")
            return {
                "success": True,
                "position_id": f"mock_{int(_utcnow().timestamp())}",
                "order_type": action,
                "lots": lots,
                "entry_price": 0.0,
                "mock": True
            }

        try:
            options = {}
            if comment:
                options["comment"] = comment

            if action.upper() == "BUY":
                result = await self.connection.create_market_buy_order(
                    symbol, lots, stop_loss, take_profit, options
                )
            else:
                result = await self.connection.create_market_sell_order(
                    symbol, lots, stop_loss, take_profit, options
                )

            logger.info(f"Successfully placed trade: {result}")
            return {
                "success": True,
                "position_id": str(result.get("positionId") or result.get("orderId") or ""),
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

    async def get_unrealized_pnl(self) -> float:
        """Sums floating profit across all open positions."""
        positions = await self.get_open_positions()
        return float(sum(float(p.get("profit") or 0.0) for p in positions))

    async def close_position(self, position_id: str) -> bool:
        """Closes a specific open position by position ID."""
        if not self.connection:
            logger.warning(f"Cannot close position {position_id}: no live MetaApi connection.")
            return False

        try:
            await self.connection.close_position(position_id)
            logger.info(f"Closed MT5 position {position_id}")
            return True
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
            return False

    async def get_today_realized_profit(self) -> float:
        """Calculates cumulative realized profit for trades closed today (UTC)."""
        if not self.connection:
            return 0.0

        try:
            today_start = datetime.combine(_utcnow().date(), datetime.min.time())
            history = await self.connection.get_deals_by_time_range(today_start, _utcnow())
            deals = history.get("deals", history) if isinstance(history, dict) else history
            total_pnl = sum(
                float(deal.get("profit") or 0.0)
                + float(deal.get("commission") or 0.0)
                + float(deal.get("swap") or 0.0)
                for deal in deals
            )
            return float(total_pnl)
        except Exception as e:
            logger.error(f"Error calculating today's PnL: {e}")
            return 0.0

    async def close_all_positions_due_to_profit_cap(self) -> int:
        """Emergency closes all active positions when the daily profit cap is reached."""
        positions = await self.get_open_positions()
        closed_count = 0
        for pos in positions:
            if await self.close_position(pos["id"]):
                closed_count += 1
        return closed_count
