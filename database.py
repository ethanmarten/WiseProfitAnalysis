"""
database.py - Database Models & Management for Gemini MT5 SaaS Platform.

Manages Users, MT5 Account Credentials, Daily PnL tracking ($100 profit cap),
Setup counters (max 5/day), and Trade Audit Logs using SQLAlchemy.
"""

from datetime import datetime, date, timedelta, timezone
import logging
import os
from typing import Generator, Optional
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, Text, inspect, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

logger = logging.getLogger("database")


def utcnow() -> datetime:
    """Timezone-naive UTC timestamp (datetime.utcnow is deprecated in 3.12)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

# Production-safe database URL resolution:
# 1. If DATABASE_URL is set in the environment (e.g. Render PostgreSQL), use it directly.
# 2. Otherwise, fall back to a local SQLite file for development/testing.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trading_platform.db")

# Render/Heroku provide postgres URLs prefixed with "postgres://", but
# SQLAlchemy 2.x requires "postgresql://". Normalize if needed.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

# For Postgres on Render, enable connection pooling with pre-ping so dead links
# are recycled before they cause 500s on long-lived dynos.
if DATABASE_URL.startswith("postgresql"):
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=300,
    )
else:
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """User account model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    is_active = Column(Boolean, default=True)

    # Trading mode preference: AUTO or MANUAL
    trading_mode = Column(String(10), default="AUTO")  # AUTO: engine executes, MANUAL: user approval

    mt5_account = relationship("MT5Account", back_populates="user", uselist=False, cascade="all, delete-orphan")
    daily_trackers = relationship("DailyProfitTracker", back_populates="user", cascade="all, delete-orphan")
    trade_logs = relationship("TradeLog", back_populates="user", cascade="all, delete-orphan")
    analysis_logs = relationship("AnalysisLog", back_populates="user", cascade="all, delete-orphan")
    pending_signals = relationship("PendingSignal", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")


class UserSession(Base):
    """Opaque bearer tokens issued at login. Only the token hash is persisted."""
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token_hash = Column(String(64), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    expires_at = Column(DateTime, nullable=False)

    user = relationship("User", back_populates="sessions")

    @property
    def is_expired(self) -> bool:
        return utcnow() >= self.expires_at


class MT5Account(Base):
    """Stores local MT5 account metadata; trading runs on the user's PC."""
    __tablename__ = "mt5_accounts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    account_id = Column(String(255), nullable=False, default="local-mt5")
    login = Column(String(100), nullable=True)
    server = Column(String(100), nullable=True)
    # Legacy columns remain nullable so existing databases can be upgraded.
    # They are never read, written with credentials, or used for execution.
    legacy_token = Column("meta_api_token", String(512), nullable=True)
    legacy_password = Column("password", String(512), nullable=True)
    platform = Column(String(10), default="mt5")  # mt4 or mt5
    is_connected = Column(Boolean, default=False)
    bot_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="mt5_account")

class DailyProfitTracker(Base):
    """Tracks daily cumulative PnL, setup count, and enforces $100 profit cap."""
    __tablename__ = "daily_profit_trackers"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    tracking_date = Column(Date, default=date.today, index=True)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    daily_setup_count = Column(Integer, default=0)
    target_cap_reached = Column(Boolean, default=False)  # True when PnL >= $100
    is_locked_for_day = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="daily_trackers")


class TradeLog(Base):
    """Audit trail of executed trades."""
    __tablename__ = "trade_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    position_id = Column(String(100), nullable=True)
    symbol = Column(String(20), default="XAUUSD")
    order_type = Column(String(10), nullable=False)  # BUY or SELL
    lots = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    profit = Column(Float, default=0.0)
    status = Column(String(20), default="OPEN")  # OPEN, CLOSED, CANCELLED
    gemini_reasoning = Column(Text, nullable=True)
    executed_at = Column(DateTime, default=utcnow)
    closed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="trade_logs")


class PendingSignal(Base):
    """Queue of signals waiting for an external (local-MT5) executor to claim them.

    The Render background engine writes a row here whenever it produces a
    directional signal. The user's Windows MT5 bot polls GET /api/signals/pending,
    receives one, executes it locally, and POSTs an acknowledgement with the
    outcome (success / failure / slippage). Rows that are not acknowledged
    within `expires_at` are marked EXPIRED by the engine and never re-issued.

    Status flow:
        PENDING  -> bot has not seen it yet
        CLAIMED  -> bot called /ack with status=in_progress (single-consumer lock)
        EXECUTED -> bot reported a successful order_send
        FAILED   -> bot reported a failed order_send (retcode != DONE)
        EXPIRED  -> TTL passed; engine will re-evaluate next cycle
    """
    __tablename__ = "pending_signals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    action = Column(String(10), nullable=False)  # BUY or SELL
    lots = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    confidence = Column(Float, default=0.0)
    reasoning = Column(Text, nullable=True)
    setup_type = Column(String(100), nullable=True)
    risk_reward_ratio = Column(Float, nullable=True)

    status = Column(String(20), default="PENDING", index=True)  # PENDING/CLAIMED/EXECUTED/FAILED/EXPIRED
    claim_token = Column(String(64), nullable=True, index=True)  # unique per-claim lock
    claimed_at = Column(DateTime, nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    execution_position_id = Column(String(100), nullable=True)
    execution_entry_price = Column(Float, nullable=True)
    execution_error = Column(Text, nullable=True)

    created_at = Column(DateTime, default=utcnow, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)

    user = relationship("User", back_populates="pending_signals")


class AnalysisLog(Base):
    """Stores history of market analyses (manual searches and background scans)."""
    __tablename__ = "analysis_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    action = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, default=0.0)
    entry_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    risk_reward_ratio = Column(Float, nullable=True)
    setup_type = Column(String(100), nullable=True)
    reasoning = Column(Text, nullable=True)
    current_price = Column(Float, nullable=True)
    analyzed_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="analysis_logs")


def _run_lightweight_migrations() -> None:
    """Adds columns introduced after the first release to existing databases.

    The project has no Alembic setup, so new nullable columns are added in place
    to avoid breaking deployments that already hold live user data.
    """
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("users")}
    if "password_hash" not in existing:
        logger.info("Migrating: adding users.password_hash column.")
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255)"))

    # Add trading_mode column if it doesn't exist
    if "trading_mode" not in existing:
        logger.info("Migrating: adding users.trading_mode column.")
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN trading_mode VARCHAR(10) DEFAULT 'AUTO'"))

    # pending_signals table is created by Base.metadata.create_all on first run;
    # on existing databases the lightweight migrations below backfill required
    # indexes that the production CREATE TABLE may have skipped.
    if "pending_signals" not in inspector.get_table_names():
        return  # Table will be created by create_all() on the next call.

    sig_indexes = {ix["name"] for ix in inspector.get_indexes("pending_signals")}
    with engine.begin() as conn:
        if "ix_pending_signals_user_id" not in sig_indexes:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_pending_signals_user_id ON pending_signals (user_id)"))
        if "ix_pending_signals_status" not in sig_indexes:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_pending_signals_status ON pending_signals (status)"))
        if "ix_pending_signals_expires_at" not in sig_indexes:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_pending_signals_expires_at ON pending_signals (expires_at)"))


def init_db():
    """Initializes tables in database and applies lightweight migrations."""
    Base.metadata.create_all(bind=engine)
    try:
        _run_lightweight_migrations()
    except Exception as exc:  # noqa: BLE001 - migrations must never block boot
        logger.error(f"Lightweight migration failed: {exc}")


def purge_expired_sessions(db: Session) -> int:
    """Deletes expired session tokens. Returns the number removed."""
    removed = db.query(UserSession).filter(UserSession.expires_at < utcnow()).delete()
    db.commit()
    return removed


def get_db() -> Generator[Session, None, None]:
    """Dependency injection helper for FastAPI endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_daily_tracker(db: Session, user_id: int) -> DailyProfitTracker:
    """Helper to fetch or instantiate today's profit tracker for a given user."""
    today = date.today()
    tracker = db.query(DailyProfitTracker).filter(
        DailyProfitTracker.user_id == user_id,
        DailyProfitTracker.tracking_date == today
    ).first()

    if not tracker:
        tracker = DailyProfitTracker(
            user_id=user_id,
            tracking_date=today,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_pnl=0.0,
            daily_setup_count=0,
            target_cap_reached=False,
            is_locked_for_day=False
        )
        db.add(tracker)
        db.commit()
        db.refresh(tracker)

    return tracker
