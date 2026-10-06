"""
local_mt5_bot.py - WiseProfit Bridge: local Windows MT5 executor.

Run this on the Windows PC that has MetaTrader 5 installed and an active
broker login. It:

  1. Initializes MT5 (terminal must already be logged in to the broker account).
    2. Polls the Render server's public /api/signals endpoint every few seconds.
    3. Executes the delivered signal via MetaTrader5.order_send.

The token is fetched at startup, NOT hard-coded, so rotating your WiseProfit
password does not require updating this file. Keep it next to a .env that
sets:

    RENDER_SERVER_URL=https://wiseprofitanalysis.onrender.com
    MT5_SYMBOL=XAUUSD
    POLL_INTERVAL_SECONDS=4

Requires:  pip install MetaTrader5 requests python-dotenv
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RENDER_URL = os.getenv("RENDER_SERVER_URL", "https://wiseprofitanalysis.onrender.com").rstrip("/")
MT5_SYMBOL = os.getenv("MT5_SYMBOL", "XAUUSD").strip().upper()
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL_SECONDS", "4"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "6"))
MARKET_DATA_INTERVAL = float(os.getenv("MARKET_DATA_INTERVAL_SECONDS", "30"))

# Treat the connection as dead after this many failed polls in a row.
MAX_CONSECUTIVE_FAILURES = 10

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("wp-bridge")

_session = requests.Session()
_session.headers.update({"User-Agent": "WiseProfit-Bridge/1.0"})
last_processed_signal_id = None
# Credentials must come from bridge/.env or the process environment.
WP_EMAIL = os.getenv("WP_EMAIL", "").strip()
WP_PASSWORD = os.getenv("WP_PASSWORD", "").strip()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
class BridgeError(RuntimeError):
    pass


def _login() -> str:
    """POST /api/login -> bearer token. Cached for the rest of the session."""
    if not WP_EMAIL or not WP_PASSWORD:
        raise BridgeError(
            "WP_EMAIL and WP_PASSWORD must be set in your .env file "
            "(these are your WiseProfit dashboard login credentials)."
        )
    log.info("Logging in to WiseProfit dashboard as %s ...", WP_EMAIL)
    res = _session.post(
        f"{RENDER_URL}/api/login",
        json={"email": WP_EMAIL, "password": WP_PASSWORD},
        timeout=REQUEST_TIMEOUT,
    )
    if res.status_code != 200:
        raise BridgeError(f"Login failed ({res.status_code}): {res.text[:200]}")
    data = res.json()
    token = data.get("access_token")
    if not token:
        raise BridgeError(f"Login response missing access_token: {data}")
    _session.headers["Authorization"] = f"Bearer {token}"
    log.info("Logged in successfully (user_id=%s).", data.get("user_id"))
    return token


def _get(path: str) -> Dict[str, Any]:
    res = _session.get(f"{RENDER_URL}{path}", timeout=REQUEST_TIMEOUT)
    if res.status_code == 401:
        raise BridgeError("Session expired; need to re-login.")
    if res.status_code >= 400:
        raise BridgeError(f"GET {path} -> {res.status_code}: {res.text[:200]}")
    return res.json()


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    res = _session.post(f"{RENDER_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    if res.status_code == 401:
        raise BridgeError("Session expired; need to re-login.")
    if res.status_code >= 400:
        raise BridgeError(f"POST {path} -> {res.status_code}: {res.text[:200]}")
    return res.json()


# ---------------------------------------------------------------------------
# MT5 execution
# ---------------------------------------------------------------------------
def initialize_mt5() -> bool:
    """Connect to the locally installed MetaTrader 5 terminal."""
    if mt5 is None:
        log.error("MetaTrader5 is not installed. Run: pip install MetaTrader5")
        return False
    if not mt5.initialize():
        log.error("MT5 initialize() failed: %s", mt5.last_error())
        return False

    acc = mt5.account_info()
    if acc is None:
        log.error("MT5 account_info() returned None — terminal not logged in.")
        mt5.shutdown()
        return False

    log.info("=" * 50)
    log.info(" WiseProfit Bridge connected to MT5")
    log.info(" Account: %s | Balance: %.2f %s", acc.login, acc.balance, acc.currency)
    log.info(" Server : %s", RENDER_URL)
    log.info(" Symbol : %s", MT5_SYMBOL)
    log.info("=" * 50)
    return True


def register_local_account() -> None:
    """Tell the backend which logged-in MT5 terminal will execute signals."""
    account = mt5.account_info()
    if account is None:
        raise BridgeError("MT5 account information is unavailable")
    response = _post(
        "/api/register-account",
        {
            "login": str(account.login),
            "server": str(account.server),
            "platform": "mt5",
        },
    )
    log.info(
        "Local MT5 account registered (login=%s, server=%s, bot_enabled=%s).",
        account.login,
        account.server,
        response.get("bot_enabled"),
    )


def send_market_data(symbol: str = "XAUUSD") -> bool:
    """Upload local M1/M5/M15 candles so Render can run Gemini analysis."""
    if not mt5.symbol_select(symbol, True):
        log.warning("Cannot select %s for market-data upload: %s", symbol, mt5.last_error())
        return False

    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
    }
    timeframes = {}
    for name, timeframe in timeframe_map.items():
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 50)
        if rates is None or len(rates) < 3:
            log.warning("Not enough %s candles for %s: %s", name, symbol, mt5.last_error())
            return False
        timeframes[name] = [
            {
                "time": int(row["time"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "tick_volume": int(row["tick_volume"]),
            }
            for row in rates
        ]

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.warning("Cannot read live %s quote: %s", symbol, mt5.last_error())
        return False
    quote = {
        "bid": float(tick.bid),
        "ask": float(tick.ask),
        "last": float(getattr(tick, "last", 0.0) or 0.0),
    }

    try:
        response = _session.post(
            f"{RENDER_URL}/api/market-data",
            json={"symbol": symbol, "quote": quote, "timeframes": timeframes},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            log.warning("Market-data upload failed (%s): %s", response.status_code, response.text[:200])
            return False
        return True
    except requests.RequestException as exc:
        log.warning("Market-data upload network error: %s", exc)
        return False


def _symbol_info(symbol: str) -> Optional[Any]:
    """Select the symbol in Market Watch if needed and return its info."""
    info = mt5.symbol_info(symbol)
    if info is None:
        log.warning("Symbol %s not found in Market Watch.", symbol)
        return None
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            log.warning("Failed to enable %s in Market Watch.", symbol)
            return None
        info = mt5.symbol_info(symbol)
    return info


def execute_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    """Place the order locally and return a dict shaped for /api/signals/{id}/ack."""
    symbol = signal.get("symbol", MT5_SYMBOL).upper()
    action = signal.get("action", "").upper()
    lots = float(signal.get("lots", 0.01))
    risk_amount = float(signal.get("risk_amount", 0.0) or 0.0)
    sl = signal.get("sl", signal.get("stop_loss"))
    tp = signal.get("tp", signal.get("take_profit"))

    if action not in ("BUY", "SELL"):
        return {"status": "failed", "error": f"Unknown action '{action}'"}

    info = _symbol_info(symbol)
    if info is None:
        return {"status": "failed", "error": f"Symbol {symbol} unavailable"}

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"status": "failed", "error": f"No live tick for {symbol}"}

    price = tick.ask if action == "BUY" else tick.bid

    # Convert the agreed dollar risk into broker volume when possible.
    if risk_amount > 0 and sl:
        tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
        tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
        stop_distance = abs(price - float(sl))
        if tick_size > 0 and tick_value > 0 and stop_distance > 0:
            lots = risk_amount / ((stop_distance / tick_size) * tick_value)

    # Normalize lots to the symbol's volume step and bounds so the broker
    # doesn't reject with "invalid price/volume".
    volume = max(info.volume_min, min(lots, info.volume_max))
    step = info.volume_step
    if step > 0:
        volume = round(volume / step) * step
    volume = float(round(volume, 2))

    filling_mode = info.filling_mode
    if filling_mode & mt5.ORDER_FILLING_IOC:
        type_filling = mt5.ORDER_FILLING_IOC
    elif filling_mode & mt5.ORDER_FILLING_FOK:
        type_filling = mt5.ORDER_FILLING_FOK
    else:
        type_filling = mt5.ORDER_FILLING_RETURN

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": float(price),
        "magic": 100200,
        "comment": "WiseProfit_Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": type_filling,
    }
    if sl:
        request["sl"] = float(sl)
    if tp:
        request["tp"] = float(tp)

    log.info(
        "Placing %s %.2f lots %s @ %.2f (SL=%s, TP=%s)",
        action, volume, symbol, price, sl, tp,
    )
    result = mt5.order_send(request)
    if result is None:
        return {"status": "failed", "error": f"order_send returned None: {mt5.last_error()}"}

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {
            "status": "failed",
            "error": f"retcode={result.retcode} comment={result.comment}",
        }

    log.info(
        "Order executed: position=%s price=%.2f",
        getattr(result, "order", None) or getattr(result, "position", None),
        result.price,
    )
    return {
        "status": "executed",
        "position_id": str(getattr(result, "position", "") or getattr(result, "order", "")),
        "entry_price": float(result.price),
    }


def send_account_state() -> bool:
    """Upload live MT5 positions and recently closed deals for PnL settlement."""
    account = mt5.account_info()
    if account is None:
        return False

    positions = []
    for position in mt5.positions_get() or []:
        positions.append({
            "position_id": str(position.ticket),
            "symbol": position.symbol,
            "type": "BUY" if position.type == mt5.POSITION_TYPE_BUY else "SELL",
            "volume": float(position.volume),
            "open_price": float(position.price_open),
            "sl": float(position.sl or 0.0),
            "tp": float(position.tp or 0.0),
            "profit": float(position.profit),
        })

    start = datetime.now() - timedelta(days=2)
    deals = mt5.history_deals_get(start, datetime.now()) or []
    closed_deals = []
    for deal in deals:
        if getattr(deal, "entry", None) != mt5.DEAL_ENTRY_OUT:
            continue
        closed_deals.append({
            "position_id": str(getattr(deal, "position_id", "")),
            "price": float(getattr(deal, "price", 0.0)),
            "profit": float(getattr(deal, "profit", 0.0)) + float(getattr(deal, "swap", 0.0)) + float(getattr(deal, "commission", 0.0)),
        })

    try:
        response = _session.post(
            f"{RENDER_URL}/api/bridge/account-state",
            json={
                "balance": float(account.balance),
                "equity": float(account.equity),
                "positions": positions,
                "closed_deals": closed_deals,
            },
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            log.warning("Account-state upload failed (%s): %s", response.status_code, response.text[:200])
            return False
        return True
    except requests.RequestException as exc:
        log.warning("Account-state upload network error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------
def run_loop() -> None:
    """Poll Render for one-time local-execution signals."""
    global last_processed_signal_id
    consecutive_failures = 0
    last_market_sync_time = 0.0

    while True:
        try:
            now = time.time()
            if now - last_market_sync_time >= MARKET_DATA_INTERVAL:
                if send_market_data(MT5_SYMBOL):
                    log.info("Uploaded fresh %s M1/M5/M15 candles to Render.", MT5_SYMBOL)
                send_account_state()
                last_market_sync_time = now

            payload = _get("/api/signals")
            sig = payload.get("signal")
            if sig:
                signal_id = str(sig.get("id"))
                if signal_id != str(last_processed_signal_id):
                    log.info("New signal #%s: %s on %s", signal_id, sig["action"], sig["symbol"])
                    outcome = execute_signal(sig)
                    _post(
                        f"/api/signals/{signal_id}/ack",
                        {
                            "claim_token": sig.get("claim_token", ""),
                            "status": outcome["status"],
                            "position_id": outcome.get("position_id"),
                            "entry_price": outcome.get("entry_price"),
                            "error": outcome.get("error"),
                        },
                    )
                    last_processed_signal_id = signal_id
                    log.info("Signal #%s result: %s", signal_id, outcome["status"])

            consecutive_failures = 0

        except BridgeError as e:
            consecutive_failures += 1
            log.warning("Bridge error (%s/%s): %s", consecutive_failures, MAX_CONSECUTIVE_FAILURES, e)
            if "re-login" in str(e).lower():
                try:
                    _login()
                    consecutive_failures = 0
                except BridgeError as le:
                    log.error("Re-login failed: %s", le)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log.error("Too many consecutive failures — sleeping 30s before retry.")
                time.sleep(30)
                consecutive_failures = 0
        except requests.RequestException as e:
            consecutive_failures += 1
            log.warning("Network error (%s/%s): %s", consecutive_failures, MAX_CONSECUTIVE_FAILURES, e)
        except KeyboardInterrupt:
            log.info("Stopping (Ctrl+C).")
            break
        except Exception as e:  # noqa: BLE001
            log.exception("Unexpected error in poll loop: %s", e)

        # Show a quiet heartbeat on the same line so users can see the bot is alive.
        tick = mt5.symbol_info_tick(MT5_SYMBOL)
        if tick is not None:
            print(
                f"[{time.strftime('%H:%M:%S')}] {MT5_SYMBOL} "
                f"Bid={tick.bid:.2f} Ask={tick.ask:.2f} | waiting...",
                end="\r", flush=True,
            )

        time.sleep(POLL_INTERVAL)


def main() -> None:
    if mt5 is None:
        log.error("MetaTrader5 is not installed. Run: pip install MetaTrader5")
        sys.exit(1)
    try:
        _login()
    except BridgeError as exc:
        log.error("Cannot start bridge: %s", exc)
        sys.exit(1)
    if not initialize_mt5():
        sys.exit(1)
    try:
        register_local_account()
    except BridgeError as exc:
        log.error("Cannot register local MT5 account: %s", exc)
        mt5.shutdown()
        sys.exit(1)
    log.info("Starting poll loop (interval=%.1fs). Press Ctrl+C to stop.", POLL_INTERVAL)
    try:
        run_loop()
    finally:
        mt5.shutdown()
        log.info("MT5 shutdown complete.")


if __name__ == "__main__":
    main()