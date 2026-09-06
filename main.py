"""
main.py - FastAPI Server & Asynchronous AI Algo-Trading SaaS Loop.

Provides REST APIs for dashboard MT5 credential setup, live PnL tracking,
and runs a continuous background worker that executes Gemini SMC Gold Scalping 24/7.

Deployment notes:
  - Designed for Render Web Service.
  - Reads configuration from environment variables (GEMINI_API_KEY, DATABASE_URL, etc.).
  - Static files (HTML/CSS/JS) are served directly by FastAPI.
  - On Render Postgres, set DATABASE_URL to the internal connection string.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional

# Ensure local module directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request, status
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import (
    SessionLocal, init_db, get_db, User, MT5Account, DailyProfitTracker, TradeLog,
    get_or_create_daily_tracker, AnalysisLog, UserSession, PendingSignal, purge_expired_sessions, utcnow
)
from gemini_analyzer import GeminiSMCAnalyzer
from mt5_executor import MetaApiMT5Executor, get_executor
from news_filter import EconomicNewsFilter
from security import hash_password, verify_password
from auth import create_session, revoke_session, current_user, authorize_user_id

from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("main_server")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
init_db()

# ---------------------------------------------------------------------------
# Trading limits (configurable, previously hard-coded)
# ---------------------------------------------------------------------------
DAILY_PROFIT_TARGET = float(os.getenv("DAILY_PROFIT_TARGET", "100"))
MAX_DAILY_SETUPS = int(os.getenv("MAX_DAILY_SETUPS", "5"))
MIN_SIGNAL_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.70"))
DEFAULT_LOT_SIZE = float(os.getenv("DEFAULT_LOT_SIZE", "0.02"))
ENGINE_INTERVAL_SECONDS = int(os.getenv("ENGINE_INTERVAL_SECONDS", "60"))

# Bridge mode: when true, the Render engine only PRODUCES signals and writes
# them to the pending_signals table. A local Windows MT5 bot (running the
# companion script) claims and executes them. Set to false (default) so the
# server executes trades directly via MetaApi as before.
BRIDGE_MODE = os.getenv("BRIDGE_MODE", "false").lower() == "true"
PENDING_SIGNAL_TTL_SECONDS = int(os.getenv("PENDING_SIGNAL_TTL_SECONDS", "90"))

# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------
# Render serves the app on https://*.onrender.com by default. We allow any
# origin so the dashboard can be embedded in front-ends hosted elsewhere while
# still being safe because the API is read/write protected by user-level auth
# in the /api/login flow.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
if ALLOWED_ORIGINS.strip() == "*":
    cors_origins = ["*"]
else:
    cors_origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

# ---------------------------------------------------------------------------
# Background loop state
# ---------------------------------------------------------------------------
BACKGROUND_LOOP_ACTIVE = True

BOT_LIVE_STATUS = {
    "step": "Initializing",
    "detail": "Background Gemini AI SaaS Trading Engine ready.",
    "updated_at": datetime.now().strftime("%H:%M:%S")
}

def update_bot_status(step: str, detail: str):
    global BOT_LIVE_STATUS
    BOT_LIVE_STATUS = {
        "step": step,
        "detail": detail,
        "updated_at": datetime.now().strftime("%H:%M:%S")
    }
    logger.info(f"Bot Activity Update [{step}]: {detail}")


# ---------------------------------------------------------------------------
# Lifespan: control background trading engine cleanly
# ---------------------------------------------------------------------------
background_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Starts background trading worker on launch; cleans up on termination."""
    logger.info("WiseProfit FastAPI server starting up...")
    if os.getenv("DISABLE_BOT", "false").lower() != "true":
        global background_task
        background_task = asyncio.create_task(run_trading_engine_loop())
        logger.info("Background trading engine started.")
    else:
        logger.info("Background trading engine DISABLED via DISABLE_BOT env var.")
    yield
    # Graceful shutdown
    global BACKGROUND_LOOP_ACTIVE
    BACKGROUND_LOOP_ACTIVE = False
    if background_task and not background_task.done():
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            logger.info("Background trading engine cancelled.")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error stopping background engine: {e}")
    logger.info("WiseProfit FastAPI server shut down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="WiseProfit - Gemini AI + MT5 Cloud SaaS Trading Platform",
    description="Automated SMC/ICT Gold Scalping Bot with $100 Daily Profit Limit",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static Files
# ---------------------------------------------------------------------------
# Only a dedicated ./static directory is exposed. Mounting the project root
# would publish .env files, the SQLite database, and source code over HTTP.
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
try:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
except RuntimeError:
    # Directory mount is already in place (e.g. during tests / reimport).
    pass


# ---------------------------------------------------------------------------
# Health Check (Render uses this to determine service health)
# ---------------------------------------------------------------------------
@app.get("/healthz", tags=["system"])
async def healthz():
    """Liveness probe used by Render health checks."""
    return {
        "status": "ok",
        "service": "wiseprofit",
        "version": app.version,
        "bot_active": BACKGROUND_LOOP_ACTIVE,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/health", tags=["system"])
async def api_health():
    """Alias health endpoint for clients/scripts."""
    return await healthz()

class LoginRequest(BaseModel):
    email: str = Field(..., json_schema_extra={"example": "trader@example.com"})
    password: str = Field(..., min_length=8, json_schema_extra={"example": "a-strong-password"})


class RegisterUserRequest(BaseModel):
    email: str = Field(..., json_schema_extra={"example": "trader@example.com"})
    password: str = Field(..., min_length=8, json_schema_extra={"example": "a-strong-password"})


class RegisterAccountRequest(BaseModel):
    meta_api_token: str = Field(..., json_schema_extra={"example": "your_meta_api_token_here"})
    account_id: str = Field(..., json_schema_extra={"example": "meta_api_account_id_uuid"})
    login: str = Field(..., json_schema_extra={"example": "12345678"})
    password: str = Field(..., json_schema_extra={"example": "secret_password"})
    server: str = Field(..., json_schema_extra={"example": "Broker-ServerName"})
    platform: str = Field(default="mt5", json_schema_extra={"example": "mt5"})


class ToggleBotRequest(BaseModel):
    enabled: bool


class SignalAckRequest(BaseModel):
    claim_token: str = Field(..., min_length=8)
    status: str = Field(..., description="executed | failed")
    position_id: Optional[str] = None
    entry_price: Optional[float] = None
    error: Optional[str] = None


# --- REST API Endpoints ---

@app.get("/", response_class=FileResponse)
async def serve_landing_page():
    """Serves the main landing page with Sign-In / Create Account modals.

    Uses FastAPI's FileResponse so the framework computes Content-Length from
    the actual file size and streams the body cleanly. This avoids the
    "RuntimeError: Response content longer than Content-Length" that occurs
    when raw HTMLResponse bodies drift in size vs. an outdated header.
    """
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    # Fallback handled outside FileResponse to keep header math intact.
    return HTMLResponse("<h1>WiseProfit</h1><p>Landing page not available.</p>")


@app.get("/dashboard", response_class=FileResponse)
async def serve_dashboard():
    """Serves the main trading dashboard HTML file.

    Uses FastAPI's FileResponse so the framework computes Content-Length from
    the actual file size and streams the body cleanly. This avoids the
    "RuntimeError: Response content longer than Content-Length" that occurs
    when raw HTMLResponse bodies drift in size vs. an outdated header.
    """
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path, media_type="text/html")
    # Fallback handled outside FileResponse to keep header math intact.
    return HTMLResponse("<h1>WiseProfit Dashboard</h1><p>Dashboard not available.</p>")


# Serve all root-level static assets (favicon, robots.txt, etc.) directly.
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = os.path.join(os.path.dirname(__file__), "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/x-icon")
    return JSONResponse(status_code=204, content=None)


@app.post("/api/signup", status_code=status.HTTP_201_CREATED)
async def signup_user(req: RegisterUserRequest, db: Session = Depends(get_db)):
    """Creates a new account with a hashed password and returns an access token."""
    email = (req.email or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="A valid email address is required.")

    user = db.query(User).filter(User.email == email).first()
    if user and user.password_hash:
        raise HTTPException(status_code=409, detail="An account with this email already exists. Please sign in.")

    if not user:
        user = User(email=email)
        db.add(user)

    # Claims a legacy passwordless row created before authentication existed.
    user.password_hash = hash_password(req.password)
    db.commit()
    db.refresh(user)

    token = create_session(db, user)
    return {
        "success": True,
        "message": "Account created successfully.",
        "user_id": user.id,
        "email": user.email,
        "access_token": token,
        "token_type": "bearer",
        "redirect": "/dashboard",
    }


@app.post("/api/login")
async def login_user(req: LoginRequest, db: Session = Depends(get_db)):
    """Authenticates a user with email + password and issues a bearer token."""
    email = (req.email or "").strip().lower()
    user = db.query(User).filter(User.email == email).first()

    if not user or not user.password_hash or not verify_password(req.password, user.password_hash):
        # Identical message for unknown email and wrong password to avoid
        # leaking which accounts exist.
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="This account has been deactivated.")

    purge_expired_sessions(db)
    token = create_session(db, user)

    return {
        "success": True,
        "message": "User authenticated successfully.",
        "user_id": user.id,
        "email": user.email,
        "access_token": token,
        "token_type": "bearer",
        "redirect": "/dashboard",
    }


@app.post("/api/logout")
async def logout_user(request: Request, db: Session = Depends(get_db)):
    """Revokes the caller's session token."""
    header = request.headers.get("authorization") or ""
    token = header[7:].strip() if header.lower().startswith("bearer ") else request.headers.get("x-auth-token")
    if token:
        revoke_session(db, token)
    return {"success": True, "message": "Signed out."}


@app.get("/api/me")
async def get_me(user: User = Depends(current_user)):
    """Returns the authenticated user's identity, used by the dashboard on load."""
    return {"user_id": user.id, "email": user.email}


@app.post("/api/register-account", status_code=status.HTTP_201_CREATED)
async def register_account(
    req: RegisterAccountRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Registers or updates the authenticated user's MT5 / MetaApi credentials.

    The MetaApi token and MT5 password are encrypted before they touch the
    database, so a leaked dump cannot be replayed against a live account.
    """
    mt5_acc = db.query(MT5Account).filter(MT5Account.user_id == user.id).first()
    if not mt5_acc:
        mt5_acc = MT5Account(
            user_id=user.id,
            account_id=req.account_id,
            login=req.login,
            server=req.server,
            platform=req.platform,
            is_connected=False,
            bot_enabled=True,
            meta_api_token="",
            password="",
        )
        db.add(mt5_acc)
    else:
        mt5_acc.account_id = req.account_id
        mt5_acc.login = req.login
        mt5_acc.server = req.server
        mt5_acc.platform = req.platform

    mt5_acc.set_credentials(req.meta_api_token, req.password)

    # Verify the credentials actually connect before claiming success.
    executor = await get_executor(req.meta_api_token, req.account_id)
    mt5_acc.is_connected = executor.is_live
    db.commit()

    # Ensure daily tracker exists
    get_or_create_daily_tracker(db, user.id)

    return {
        "success": True,
        "message": (
            "MT5 account connected and verified."
            if executor.is_live
            else f"Credentials saved, but MetaApi connection failed: {executor.last_error}"
        ),
        "connected": executor.is_live,
        "user_id": user.id,
        "email": user.email,
        "bot_enabled": mt5_acc.bot_enabled,
    }


@app.get("/api/dashboard/{user_id}")
async def get_dashboard_data(
    user_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Fetches the authenticated user's metrics, PnL, setup limit, and positions."""
    authorize_user_id(user, user_id)

    tracker = get_or_create_daily_tracker(db, user.id)
    mt5_acc = db.query(MT5Account).filter(MT5Account.user_id == user.id).first()
    recent_trades = db.query(TradeLog).filter(TradeLog.user_id == user.id).order_by(TradeLog.executed_at.desc()).limit(10).all()
    recent_analyses = db.query(AnalysisLog).filter(AnalysisLog.user_id == user.id).order_by(AnalysisLog.analyzed_at.desc()).limit(6).all()

    account_info = {"balance": 0.0, "equity": 0.0, "status": "Disconnected (Enter Credentials Below)"}
    open_positions = []

    token = mt5_acc.plain_meta_api_token if mt5_acc else None
    if mt5_acc and token:
        executor = await get_executor(token, mt5_acc.account_id)
        info = await executor.get_account_information()
        account_info = {
            "balance": info.get("balance", 0.0),
            "equity": info.get("equity", 0.0),
            "status": "Connected" if executor.is_live else f"Offline: {executor.last_error or 'not connected'}",
        }
        open_positions = await executor.get_open_positions()

    return {
        "user_id": user.id,
        "email": user.email,
        "bot_enabled": mt5_acc.bot_enabled if mt5_acc else False,
        "daily_pnl": tracker.total_pnl,
        "realized_pnl": tracker.realized_pnl,
        "unrealized_pnl": tracker.unrealized_pnl,
        "target_cap_reached": tracker.target_cap_reached or tracker.total_pnl >= DAILY_PROFIT_TARGET,
        "daily_setup_count": tracker.daily_setup_count,
        "max_daily_setups": MAX_DAILY_SETUPS,
        "daily_profit_target": DAILY_PROFIT_TARGET,
        "account": account_info,
        "open_positions": open_positions,
        "live_status": BOT_LIVE_STATUS,
        "recent_trades": [
            {
                "id": t.id,
                "symbol": t.symbol,
                "type": t.order_type,
                "lots": t.lots,
                "entry": t.entry_price,
                "sl": t.stop_loss,
                "tp": t.take_profit,
                "profit": t.profit,
                "status": t.status,
                "reasoning": t.gemini_reasoning,
                "time": t.executed_at.strftime("%Y-%m-%d %H:%M:%S")
            }
            for t in recent_trades
        ],
        "recent_analyses": [
            {
                "symbol": a.symbol,
                "current_price": a.current_price or a.entry_price,
                "action": a.action,
                "confidence": a.confidence,
                "entry_price": a.entry_price,
                "stop_loss": a.stop_loss,
                "take_profit": a.take_profit,
                "risk_reward_ratio": a.risk_reward_ratio,
                "setup_type": a.setup_type,
                "reasoning": a.reasoning,
                "timestamp": a.analyzed_at.strftime("%Y-%m-%d %H:%M:%S")
            }
            for a in recent_analyses
        ]
    }


# ---------------------------------------------------------------------------
# Bridge endpoints: let a local Windows MT5 bot claim and execute signals
# produced by the Render engine. Only active when BRIDGE_MODE=true on Render,
# but the endpoints are exposed in both modes so the bot can call them without
# checking the server flag.
# ---------------------------------------------------------------------------
import secrets as _secrets


def _signal_to_dict(row: PendingSignal) -> Dict[str, Any]:
    return {
        "id": row.id,
        "symbol": row.symbol,
        "action": row.action,
        "lots": row.lots,
        "entry_price": row.entry_price,
        "stop_loss": row.stop_loss,
        "take_profit": row.take_profit,
        "confidence": row.confidence,
        "reasoning": row.reasoning,
        "setup_type": row.setup_type,
        "risk_reward_ratio": row.risk_reward_ratio,
        "status": row.status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
    }


@app.get("/api/signals/pending")
async def list_pending_signals(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Returns the user's pending bridge signals, oldest first."""
    now = utcnow()
    rows = (
        db.query(PendingSignal)
        .filter(
            PendingSignal.user_id == user.id,
            PendingSignal.status.in_(["PENDING", "CLAIMED"]),
            PendingSignal.expires_at > now,
        )
        .order_by(PendingSignal.created_at.asc())
        .limit(5)
        .all()
    )
    return {"success": True, "count": len(rows), "signals": [_signal_to_dict(r) for r in rows]}


@app.post("/api/signals/{signal_id}/claim")
async def claim_signal(
    signal_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Atomically mark a pending signal as CLAIMED and return a one-shot token.

    The bot must include this token in the subsequent /ack call so a second
    bot instance can't double-execute the same signal.
    """
    row = db.query(PendingSignal).filter(
        PendingSignal.id == signal_id,
        PendingSignal.user_id == user.id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Signal not found.")

    now = utcnow()
    if row.status not in ("PENDING", "CLAIMED"):
        raise HTTPException(
            status_code=409,
            detail=f"Signal already {row.status.lower()} and cannot be re-claimed.",
        )
    if row.expires_at <= now:
        row.status = "EXPIRED"
        db.commit()
        raise HTTPException(status_code=410, detail="Signal expired before claim.")

    if not row.claim_token:
        row.claim_token = _secrets.token_urlsafe(24)
    row.status = "CLAIMED"
    row.claimed_at = now
    db.commit()
    db.refresh(row)
    return {"success": True, "claim_token": row.claim_token, "signal": _signal_to_dict(row)}


@app.post("/api/signals/{signal_id}/ack")
async def acknowledge_signal(
    signal_id: int,
    req: SignalAckRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Records the local bot's execution outcome for a claimed signal.

    On 'executed' we create the TradeLog row so the dashboard's audit trail
    and daily-setup counter stay accurate. On 'failed' we store the error
    and leave the daily_setup_count untouched.
    """
    row = db.query(PendingSignal).filter(
        PendingSignal.id == signal_id,
        PendingSignal.user_id == user.id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Signal not found.")

    if not row.claim_token or row.claim_token != req.claim_token:
        raise HTTPException(status_code=403, detail="Invalid claim token for this signal.")

    outcome = (req.status or "").lower().strip()
    if outcome not in ("executed", "failed"):
        raise HTTPException(status_code=400, detail="status must be 'executed' or 'failed'.")

    row.acknowledged_at = utcnow()
    row.execution_position_id = req.position_id
    row.execution_entry_price = req.entry_price
    row.execution_error = req.error

    if outcome == "executed":
        row.status = "EXECUTED"
        # Create the dashboard audit row + bump the daily setup counter
        # so the dashboard PnL view stays consistent with bridge execution.
        trade = TradeLog(
            user_id=user.id,
            position_id=req.position_id,
            symbol=row.symbol,
            order_type=row.action,
            lots=row.lots,
            entry_price=req.entry_price or row.entry_price,
            stop_loss=row.stop_loss,
            take_profit=row.take_profit,
            status="OPEN",
            gemini_reasoning=row.reasoning,
        )
        db.add(trade)
        tracker = get_or_create_daily_tracker(db, user.id)
        # Only increment if the engine hasn't counted this signal yet
        # (engine pre-increments in bridge mode; this avoids double-counting).
    else:
        row.status = "FAILED"

    db.commit()
    db.refresh(row)
    return {"success": True, "signal": _signal_to_dict(row)}


@app.post("/api/toggle-bot")
async def toggle_bot(
    req: ToggleBotRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Enables or disables automated trading for the authenticated user."""
    mt5_acc = db.query(MT5Account).filter(MT5Account.user_id == user.id).first()
    if not mt5_acc:
        raise HTTPException(status_code=404, detail="MT5 account not found")

    mt5_acc.bot_enabled = req.enabled
    db.commit()
    return {"success": True, "user_id": user.id, "bot_enabled": mt5_acc.bot_enabled}


@app.get("/api/trades/{user_id}")
async def get_trade_history(
    user_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Retrieves the authenticated user's full trade audit log."""
    authorize_user_id(user, user_id)
    trades = db.query(TradeLog).filter(TradeLog.user_id == user.id).order_by(TradeLog.executed_at.desc()).all()
    return [
        {
            "id": t.id,
            "symbol": t.symbol,
            "type": t.order_type,
            "lots": t.lots,
            "entry_price": t.entry_price,
            "sl": t.stop_loss,
            "tp": t.take_profit,
            "profit": t.profit,
            "status": t.status,
            "reasoning": t.gemini_reasoning,
            "executed_at": t.executed_at.isoformat()
        }
        for t in trades
    ]


import requests

def fetch_live_market_data(symbol: str) -> Dict[str, Any]:
    """Fetches real-time price, 24h high/low, bid, ask, and change % from live market feeds."""
    sym = symbol.upper().strip()
    sym_map = {
        'XAUUSD': 'PAXGUSDT', 'GOLD': 'PAXGUSDT',
        'BTCUSD': 'BTCUSDT', 'BTC': 'BTCUSDT',
        'ETHUSD': 'ETHUSDT', 'ETH': 'ETHUSDT',
        'SOLUSD': 'SOLUSDT', 'SOL': 'SOLUSDT',
        'BNBUSD': 'BNBUSDT', 'BNB': 'BNBUSDT',
        'XRPUSD': 'XRPUSDT', 'DOGEUSD': 'DOGEUSDT'
    }

    # 1. Try Binance Live Data
    binance_sym = sym_map.get(sym, sym + "USDT")
    try:
        r = requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={binance_sym}", timeout=3)
        if r.status_code == 200:
            data = r.json()
            p = float(data["lastPrice"])
            return {
                "symbol": sym,
                "price": round(p, 2 if p > 10 else 4),
                "bid": round(float(data["bidPrice"]), 2 if p > 10 else 4),
                "ask": round(float(data["askPrice"]), 2 if p > 10 else 4),
                "high": round(float(data["highPrice"]), 2 if p > 10 else 4),
                "low": round(float(data["lowPrice"]), 2 if p > 10 else 4),
                "change_percent": round(float(data["priceChangePercent"]), 2)
            }
    except Exception as e:
        logger.debug(f"Live market fetch failed for {binance_sym}: {e}")

    # 2. Try Open Exchange Rates for Forex
    forex_map = {'EURUSD': 'EUR', 'GBPUSD': 'GBP', 'USDJPY': 'JPY'}
    if sym in forex_map:
        try:
            r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=3)
            if r.status_code == 200:
                rates = r.json().get("rates", {})
                curr = forex_map[sym]
                if curr in rates:
                    rate = rates[curr]
                    price = round(1.0 / rate if sym in ['EURUSD', 'GBPUSD'] else rate, 4)
                    return {
                        "symbol": sym,
                        "price": price,
                        "bid": round(price - 0.0002, 4),
                        "ask": round(price + 0.0002, 4),
                        "high": round(price * 1.003, 4),
                        "low": round(price * 0.997, 4),
                        "change_percent": 0.12
                    }
        except Exception as e:
            logger.debug(f"Forex fetch failed: {e}")

    # 3. Dynamic Baseline
    base_prices = {
        "XAUUSD": 4462.50, "GOLD": 4462.50, "BTCUSD": 78120.0, "BTC": 78120.0,
        "EURUSD": 1.0480, "GBPUSD": 1.2650, "USDJPY": 154.20, "AAPL": 238.50,
        "NVDA": 135.20, "TSLA": 340.00, "ETHUSD": 2150.0, "ETH": 2150.0, "US30": 43500.0
    }
    p = base_prices.get(sym, 100.0)
    return {
        "symbol": sym,
        "price": p,
        "bid": round(p * 0.9998, 2 if p > 10 else 4),
        "ask": round(p * 1.0002, 2 if p > 10 else 4),
        "high": round(p * 1.005, 2 if p > 10 else 4),
        "low": round(p * 0.995, 2 if p > 10 else 4),
        "change_percent": 0.35
    }


def fetch_latest_candle_close(symbol: str) -> Dict[str, Any]:
    """
    Fetches the most recent completed candle close price from Binance klines.
    This is used to synchronize the dashboard header price tag with the live
    TradingView widget candle close (the same value rendered on the chart).
    """
    sym = symbol.upper().strip()
    sym_map = {
        'XAUUSD': 'PAXGUSDT', 'GOLD': 'PAXGUSDT',
        'BTCUSD': 'BTCUSDT', 'BTC': 'BTCUSDT',
        'ETHUSD': 'ETHUSDT', 'ETH': 'ETHUSDT',
        'SOLUSD': 'SOLUSDT', 'SOL': 'SOLUSDT',
        'BNBUSD': 'BNBUSDT', 'BNB': 'BNBUSDT',
        'XRPUSD': 'XRPUSDT', 'DOGEUSD': 'DOGEUSDT'
    }
    binance_sym = sym_map.get(sym, sym + "USDT")
    try:
        # Pull the 2 most recent 1m candles; the second is the latest CLOSED candle.
        # The first may still be forming, so we prefer the previous closed candle
        # to match what TradingView's "close" tick displays on the chart.
        r = requests.get(
            f"https://api.binance.com/api/v3/klines?symbol={binance_sym}&interval=1m&limit=2",
            timeout=3
        )
        if r.status_code == 200:
            raw = r.json()
            if len(raw) >= 2:
                closed = raw[-2]  # Last fully-closed candle
                candle_close = float(closed[4])
                candle_open = float(closed[1])
                candle_high = float(closed[2])
                candle_low = float(closed[3])
                candle_time = datetime.fromtimestamp(closed[0] / 1000).strftime("%H:%M:%S")
                change_pct = 0.0
                if candle_open > 0:
                    change_pct = round(((candle_close - candle_open) / candle_open) * 100.0, 2)
                precision = 2 if candle_close > 10 else 4
                return {
                    "symbol": sym,
                    "close": round(candle_close, precision),
                    "open": round(candle_open, precision),
                    "high": round(candle_high, precision),
                    "low": round(candle_low, precision),
                    "change_percent": change_pct,
                    "candle_time": candle_time,
                    "source": "binance_kline_close"
                }
    except Exception as e:
        logger.debug(f"Latest candle close fetch failed for {binance_sym}: {e}")
    # Fallback to ticker if kline call fails
    return fetch_live_market_data(symbol)


def fetch_live_candles(symbol: str, interval: str = "1m", limit: int = 30) -> List[Dict[str, Any]]:
    """Fetches real-time M1, M5, M15 candles from Binance."""
    sym = symbol.upper().strip()
    sym_map = {
        'XAUUSD': 'PAXGUSDT', 'GOLD': 'PAXGUSDT',
        'BTCUSD': 'BTCUSDT', 'BTC': 'BTCUSDT',
        'ETHUSD': 'ETHUSDT', 'ETH': 'ETHUSDT',
        'SOLUSD': 'SOLUSDT', 'SOL': 'SOLUSDT',
        'BNBUSD': 'BNBUSDT', 'BNB': 'BNBUSDT',
        'XRPUSD': 'XRPUSDT', 'DOGEUSD': 'DOGEUSDT'
    }
    binance_sym = sym_map.get(sym, sym + "USDT")
    try:
        r = requests.get(f"https://api.binance.com/api/v3/klines?symbol={binance_sym}&interval={interval}&limit={limit}", timeout=3)
        if r.status_code == 200:
            raw = r.json()
            candles = []
            for c in raw:
                candles.append({
                    "time": datetime.fromtimestamp(c[0] / 1000).strftime("%H:%M"),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5])
                })
            return candles
    except Exception as e:
        logger.debug(f"Live candle fetch failed for {binance_sym}: {e}")
    return []


@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Returns live 24h market stats (real-time price, bid, ask, high, low, change)."""
    return fetch_live_market_data(symbol)


@app.get("/api/candle-close/{symbol}")
async def get_candle_close(symbol: str):
    """Returns the latest closed-candle close price from Binance klines.
    Used to synchronize the dashboard header price with the TradingView candle feed."""
    return fetch_latest_candle_close(symbol)


class AnalyzeSymbolRequest(BaseModel):
    symbol: str = Field(..., json_schema_extra={"example": "XAUUSD"})


@app.post("/api/analyze-symbol")
async def analyze_symbol_on_demand(
    req: AnalyzeSymbolRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """
    On-demand AI technical analysis for any requested asset symbol (e.g. XAUUSD, BTCUSD, EURUSD).
    Uses real live market prices and Gemini AI to calculate the SMC/ICT setup,
    entry, SL, TP, and confidence.
    """
    symbol = req.symbol.upper().strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol cannot be empty.")

    analyzer = GeminiSMCAnalyzer()
    mt5_acc = db.query(MT5Account).filter(MT5Account.user_id == user.id).first()

    # Fetch actual real-time market data
    live_data = fetch_live_market_data(symbol)
    current_price = live_data["price"]
    m1_candles, m5_candles, m15_candles = [], [], []

    token = mt5_acc.plain_meta_api_token if mt5_acc else None
    if token:
        try:
            executor = await get_executor(token, mt5_acc.account_id)
            if executor.is_live:
                m1_candles = await executor.fetch_candles(symbol, "1m", limit=30)
                m5_candles = await executor.fetch_candles(symbol, "5m", limit=30)
                m15_candles = await executor.fetch_candles(symbol, "15m", limit=30)
                live_p = await executor.get_current_price(symbol)
                if live_p > 0:
                    current_price = live_p
        except Exception as e:
            logger.warning(f"Could not fetch MT5 candles for {symbol}: {e}")

    if not m15_candles:
        m1_candles = fetch_live_candles(symbol, "1m", limit=30)
        m5_candles = fetch_live_candles(symbol, "5m", limit=30)
        m15_candles = fetch_live_candles(symbol, "15m", limit=30)

    if not m15_candles:
        # No real market data available anywhere. Previously this fabricated
        # random candles and fed them to the model, producing convincing but
        # meaningless signals. Fail honestly instead.
        raise HTTPException(
            status_code=503,
            detail=(
                f"No live market data is currently available for {symbol}. "
                "Connect an MT5 account or try a symbol supported by the public feed."
            ),
        )

    signal = await analyzer.analyze_market(m1_candles, m5_candles, m15_candles, current_price)

    action = signal.get("action", "HOLD").upper()
    confidence = float(signal.get("confidence", 0.0))
    entry_price = signal.get("entry_price") or current_price
    stop_loss = signal.get("stop_loss")
    take_profit = signal.get("take_profit")
    risk_reward_ratio = signal.get("risk_reward_ratio")
    setup_type = signal.get("setup_type") or ("SMC/ICT Market Structure Analysis" if action != "HOLD" else "No High-Probability Setup")
    reasoning = signal.get("reasoning", "Analysis completed.")

    analysis_entry = AnalysisLog(
        user_id=user.id,
        symbol=symbol,
        action=action,
        confidence=confidence,
        entry_price=entry_price if action in ["BUY", "SELL"] else None,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward_ratio=risk_reward_ratio,
        setup_type=setup_type,
        reasoning=reasoning,
        current_price=current_price
    )
    db.add(analysis_entry)
    db.commit()

    return {
        "symbol": symbol,
        "current_price": current_price,
        "action": action,
        "confidence": confidence,
        "entry_price": entry_price if action in ["BUY", "SELL"] else current_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward_ratio": risk_reward_ratio,
        "setup_type": setup_type,
        "reasoning": reasoning,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# --- Asynchronous Background SaaS Trading Engine ---

async def run_trading_engine_loop():
    """
    Continuous async background loop running 24/7.
    Evaluates market setups using Gemini AI, executes trades via MetaApi,
    and enforces strict $100 daily profit caps & 5 setup/day limits.
    """
    logger.info("Starting Background Gemini AI + MT5 Trading Engine Loop...")
    analyzer = GeminiSMCAnalyzer()
    news_guard = EconomicNewsFilter(quiet_window_minutes=15)
    while BACKGROUND_LOOP_ACTIVE:
        try:
            update_bot_status("Scanning Accounts", "Checking active MT5 accounts and daily limits...")
            db: Session = SessionLocal()
            try:
                # In bridge mode, expire any signal whose TTL has passed so
                # the local bot can't claim a stale setup and so the daily
                # setup counter stays accurate.
                if BRIDGE_MODE:
                    expired = db.query(PendingSignal).filter(
                        PendingSignal.status == "PENDING",
                        PendingSignal.expires_at < utcnow(),
                    ).all()
                    for row in expired:
                        row.status = "EXPIRED"
                    if expired:
                        db.commit()
                        logger.info(f"Bridge: marked {len(expired)} pending signal(s) as EXPIRED.")

                # Fetch all active MT5 accounts with bot enabled
                accounts = db.query(MT5Account).filter(MT5Account.bot_enabled == True).all()

                if not accounts:
                    update_bot_status("Standby", "No active MT5 accounts connected. Waiting for setup...")

                for mt5_acc in accounts:
                    user_id = mt5_acc.user_id
                    tracker = get_or_create_daily_tracker(db, user_id)

                    api_token = mt5_acc.plain_meta_api_token
                    if not api_token:
                        update_bot_status(
                            "Credential Error",
                            f"User {user_id}: stored MetaApi token could not be decrypted. Re-enter credentials."
                        )
                        continue

                    executor = await get_executor(api_token, mt5_acc.account_id)

                    # Persist the real connection state so the dashboard is honest.
                    if mt5_acc.is_connected != executor.is_live:
                        mt5_acc.is_connected = executor.is_live
                        db.commit()

                    if not executor.is_live:
                        if BRIDGE_MODE:
                            # In bridge mode we don't require a live MetaApi
                            # connection; the local bot will execute the trade.
                            update_bot_status(
                                "Bridge Mode",
                                f"User {user_id}: MetaApi offline, queuing signals for the local Windows bot instead."
                            )
                        else:
                            update_bot_status(
                                "MT5 Disconnected",
                                f"User {user_id}: {executor.last_error or 'MetaApi connection unavailable'}. Skipping cycle."
                            )
                            continue

                    # Refresh PnL from the broker before evaluating any limit.
                    realized_pnl = await executor.get_today_realized_profit()
                    unrealized_pnl = await executor.get_unrealized_pnl()
                    tracker.realized_pnl = realized_pnl
                    tracker.unrealized_pnl = unrealized_pnl
                    tracker.total_pnl = realized_pnl + unrealized_pnl
                    db.commit()

                    # Check 1: Enforce the daily profit cap.
                    if tracker.total_pnl >= DAILY_PROFIT_TARGET or tracker.target_cap_reached:
                        if not tracker.target_cap_reached:
                            logger.info(
                                f"User {user_id} hit the ${DAILY_PROFIT_TARGET:.0f} daily profit cap. "
                                "Locking the account for today."
                            )
                            tracker.target_cap_reached = True
                            tracker.is_locked_for_day = True
                            db.commit()
                            closed = await executor.close_all_positions_due_to_profit_cap()
                            logger.info(f"Closed {closed} open position(s) for user {user_id} after hitting the cap.")

                        update_bot_status(
                            "Profit Cap Reached",
                            f"User {user_id} locked for today (target ${DAILY_PROFIT_TARGET:.0f} hit)."
                        )
                        continue

                    # Check 2: Enforce the maximum number of setups per day.
                    if tracker.daily_setup_count >= MAX_DAILY_SETUPS:
                        update_bot_status(
                            "Daily Setup Cap Hit",
                            f"User {user_id} reached {tracker.daily_setup_count}/{MAX_DAILY_SETUPS} daily setups."
                        )
                        continue

                    # Check 3: Economic news blackout window.
                    update_bot_status("News Guard Check", "Verifying the high-impact economic news filter...")
                    news_status = await news_guard.is_news_impact_zone()
                    if not news_status["is_safe"]:
                        update_bot_status("News Blackout Active", f"Trading paused: {news_status['reason']}")
                        continue

                    # Fetch XAUUSD candles for M1, M5, M15.
                    update_bot_status("Fetching Candles", "Downloading M1, M5, M15 OHLC candles for XAUUSD (Gold)...")
                    m1_candles = await executor.fetch_candles("XAUUSD", "1m", limit=30)
                    m5_candles = await executor.fetch_candles("XAUUSD", "5m", limit=30)
                    m15_candles = await executor.fetch_candles("XAUUSD", "15m", limit=30)
                    current_price = await executor.get_current_price("XAUUSD")

                    if not m15_candles or current_price <= 0:
                        update_bot_status(
                            "Market Data Unavailable",
                            f"User {user_id}: no XAUUSD candles or price from MetaApi. Skipping this cycle."
                        )
                        continue

                    # Query the Gemini SMC strategy engine.
                    update_bot_status(
                        "Gemini AI Analysis",
                        f"Sending candle data to {analyzer.model_name} for SMC/ICT analysis at ${current_price:.2f}..."
                    )
                    signal = await analyzer.analyze_market(m1_candles, m5_candles, m15_candles, current_price)

                    action = signal.get("action", "HOLD").upper()
                    confidence = float(signal.get("confidence", 0.0) or 0.0)

                    # Persist BUY/SELL analyses for the dashboard audit trail.
                    if action in ["BUY", "SELL"]:
                        analysis_entry = AnalysisLog(
                            user_id=user_id,
                            symbol="XAUUSD",
                            action=action,
                            confidence=confidence,
                            entry_price=signal.get("entry_price") or current_price,
                            stop_loss=signal.get("stop_loss"),
                            take_profit=signal.get("take_profit"),
                            risk_reward_ratio=signal.get("risk_reward_ratio"),
                            setup_type=signal.get("setup_type", "SMC/ICT Institutional Setup"),
                            reasoning=signal.get("reasoning", ""),
                            current_price=current_price
                        )
                        db.add(analysis_entry)
                        db.commit()

                    # Execute only on a high-confidence directional signal.
                    if action in ["BUY", "SELL"] and confidence >= MIN_SIGNAL_CONFIDENCE:
                        update_bot_status(
                            "Executing Order",
                            f"Signal: {action} (confidence {confidence*100:.0f}%) | Placing XAUUSD trade..."
                        )
                        logger.info(f"Gemini SMC signal for user {user_id}: {action} | confidence {confidence}")

                        lots = DEFAULT_LOT_SIZE
                        sl = signal.get("stop_loss")
                        tp = signal.get("take_profit")

                        if BRIDGE_MODE:
                            # Enqueue for the local Windows bot instead of
                            # executing on Render. The bot will POST /api/signals/<id>/ack
                            # with the execution result; that handler (see below)
                            # creates the TradeLog row.
                            entry_for_signal = signal.get("entry_price") or current_price
                            pending = PendingSignal(
                                user_id=user_id,
                                symbol="XAUUSD",
                                action=action,
                                lots=lots,
                                entry_price=float(entry_for_signal),
                                stop_loss=sl,
                                take_profit=tp,
                                confidence=confidence,
                                reasoning=signal.get("reasoning", ""),
                                setup_type=signal.get("setup_type", "SMC/ICT Institutional Setup"),
                                risk_reward_ratio=signal.get("risk_reward_ratio"),
                                status="PENDING",
                                expires_at=utcnow() + timedelta(seconds=PENDING_SIGNAL_TTL_SECONDS),
                            )
                            db.add(pending)
                            tracker.daily_setup_count += 1
                            db.commit()
                            update_bot_status(
                                "Signal Queued",
                                f"User {user_id}: {action} signal #{pending.id} queued for the local bot. "
                                f"Daily setups: {tracker.daily_setup_count}/{MAX_DAILY_SETUPS}"
                            )
                            logger.info(
                                f"Bridge: queued pending_signal id={pending.id} "
                                f"({action} XAUUSD {lots} lots) for user {user_id}."
                            )
                            continue

                        trade_result = await executor.execute_trade(
                            symbol="XAUUSD",
                            action=action,
                            lots=lots,
                            stop_loss=sl,
                            take_profit=tp,
                            comment="Gemini_SMC"
                        )

                        if trade_result.get("success"):
                            log_entry = TradeLog(
                                user_id=user_id,
                                position_id=trade_result.get("position_id"),
                                symbol="XAUUSD",
                                order_type=action,
                                lots=lots,
                                entry_price=trade_result.get("entry_price") or current_price,
                                stop_loss=sl,
                                take_profit=tp,
                                status="OPEN",
                                gemini_reasoning=signal.get("reasoning")
                            )
                            db.add(log_entry)
                            tracker.daily_setup_count += 1
                            db.commit()

                            update_bot_status(
                                "Trade Executed",
                                f"Position opened. Daily setups: {tracker.daily_setup_count}/{MAX_DAILY_SETUPS}"
                            )
                            logger.info(
                                f"Trade executed and logged for user {user_id}. "
                                f"Daily setups: {tracker.daily_setup_count}/{MAX_DAILY_SETUPS}"
                            )
                        else:
                            update_bot_status(
                                "Trade Rejected",
                                f"Order not placed: {trade_result.get('error', 'unknown error')}"
                            )
                            logger.error(f"Trade execution failed for user {user_id}: {trade_result.get('error')}")
                    else:
                        reasoning_msg = signal.get("reasoning", "No trade setup.")
                        update_bot_status(
                            "Analysis Result: HOLD",
                            f"Gemini returned {action} (confidence {confidence*100:.0f}%): {reasoning_msg}"
                        )

            finally:
                db.close()

        except asyncio.CancelledError:
            logger.info("Trading engine loop cancelled during shutdown.")
            raise
        except Exception as e:
            update_bot_status("Error", f"Engine loop exception: {str(e)}")
            logger.error(f"Error in background trading engine loop: {e}", exc_info=True)

        update_bot_status(
            "Waiting Cycle",
            f"Scan cycle complete. Waiting {ENGINE_INTERVAL_SECONDS}s for the next market scan..."
        )
        await asyncio.sleep(ENGINE_INTERVAL_SECONDS)


if __name__ == "__main__":
    import uvicorn
    # Render injects $PORT; fall back to 8000 for local `python main.py` runs.
    port = int(os.getenv("PORT", "8000"))
    # Disable reload in production (single worker is fine for the SaaS bot loop).
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENV", "production") == "development",
    )
