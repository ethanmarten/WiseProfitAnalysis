"""
local_mt5_bot.py - WiseProfit Bridge: local Windows MT5 executor.

Run this on the Windows PC that has MetaTrader 5 installed and an active
broker login. It:

  1. Initializes MT5 (terminal must already be logged in to the broker account).
  2. Logs in to the WiseProfit dashboard (email + password) and stores the
     bearer token in memory for the lifetime of the script.
  3. Polls the Render server's /api/signals/pending endpoint every few seconds.
  4. For each pending signal:
       - claims it (POST /api/signals/{id}/claim) to lock it server-side,
       - executes the order via MetaTrader5.order_send,
       - acknowledges the outcome (POST /api/signals/{id}/ack) with the
         position id, actual fill price, and any error retcode.

The token is fetched at startup, NOT hard-coded, so rotating your WiseProfit
password does not require updating this file. Keep it next to a .env that
sets:

    RENDER_SERVER_URL=https://wiseprofitanalysis.onrender.com
    WP_EMAIL=your@email.com
    WP_PASSWORD=your-strong-password
    MT5_SYMBOL=XAUUSD
    POLL_INTERVAL_SECONDS=4

Requires:  pip install MetaTrader5 requests python-dotenv
"""

import os
import sys
import time
import logging
from typing import Optional, Dict, Any

try:
    import MetaTrader5 as mt5
except ImportError:
    print("[X] MetaTrader5 package is not installed.")
    print("    Run:  pip install MetaTrader5")
    sys.exit(1)

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RENDER_URL = os.getenv("RENDER_SERVER_URL", "https://wiseprofitanalysis.onrender.com").rstrip("/")
WP_EMAIL = os.getenv("WP_EMAIL", "").strip()
WP_PASSWORD = os.getenv("WP_PASSWORD", "")
MT5_SYMBOL = os.getenv("MT5_SYMBOL", "XAUUSD").strip().upper()
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL_SECONDS", "4"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "6"))

# Treat the connection as dead after this many failed polls in a row.
MAX_CONSECUTIVE_FAILURES = 10

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("wp-bridge")

_session = requests.Session()
_session.headers.update({"User-Agent": "WiseProfit-Bridge/1.0"})


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
    sl = signal.get("stop_loss")
    tp = signal.get("take_profit")

    if action not in ("BUY", "SELL"):
        return {"status": "failed", "error": f"Unknown action '{action}'"}

    info = _symbol_info(symbol)
    if info is None:
        return {"status": "failed", "error": f"Symbol {symbol} unavailable"}

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"status": "failed", "error": f"No live tick for {symbol}"}

    price = tick.ask if action == "BUY" else tick.bid

    # Normalize lots to the symbol's volume step and bounds so the broker
    # doesn't reject with "invalid price/volume".
    volume = max(info.volume_min, min(lots, info.volume_max))
    step = info.volume_step
    if step > 0:
        volume = round(volume / step) * step
    volume = float(round(volume, 2))

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": float(price),
        "magic": 100200,
        "comment": "WiseProfit_Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
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
        "position_id": str(getattr(result, "order", "") or getattr(result, "position", "")),
        "entry_price": float(result.price),
    }


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------
def run_loop() -> None:
    """Poll Render for pending signals and execute each one."""
    consecutive_failures = 0

    while True:
        try:
            pending = _get("/api/signals/pending")
            signals = pending.get("signals") or []

            for sig in signals:
                if sig.get("status") not in ("PENDING", "CLAIMED"):
                    continue

                log.info(
                    "Claiming signal #%s: %s %.2f %s @ %.2f (conf=%.0f%%)",
                    sig["id"], sig["action"], sig["lots"], sig["symbol"],
                    sig["entry_price"], sig.get("confidence", 0) * 100,
                )
                claimed = _post(f"/api/signals/{sig['id']}/claim", {})
                claim_token = claimed.get("claim_token")
                if not claim_token:
                    log.warning("Claim did not return a token: %s", claimed)
                    continue

                outcome = execute_signal(claimed.get("signal") or sig)
                _post(
                    f"/api/signals/{sig['id']}/ack",
                    {
                        "claim_token": claim_token,
                        "status": outcome["status"],
                        "position_id": outcome.get("position_id"),
                        "entry_price": outcome.get("entry_price"),
                        "error": outcome.get("error"),
                    },
                )
                log.info(
                    "Signal #%s acknowledged (%s).",
                    sig["id"],
                    outcome["status"].upper(),
                )

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
    if not initialize_mt5():
        sys.exit(1)
    try:
        _login()
    except BridgeError as e:
        log.error("Initial login failed: %s", e)
        mt5.shutdown()
        sys.exit(2)

    log.info("Starting poll loop (interval=%.1fs). Press Ctrl+C to stop.", POLL_INTERVAL)
    try:
        run_loop()
    finally:
        mt5.shutdown()
        log.info("MT5 shutdown complete.")


if __name__ == "__main__":
    main()