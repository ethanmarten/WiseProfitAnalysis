"""
database.py - Database Models & Management for Gemini MT5 SaaS Platform.

Manages Users, MT5 Account Credentials, Daily PnL tracking ($100 profit cap),
Setup counters (max 5/day), and Trade Audit Logs using SQLAlchemy.
"""

from datetime import datetime, date
import os
from typing import Generator, Optional
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Date, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

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
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    mt5_account = relationship("MT5Account", back_populates="user", uselist=False, cascade="all, delete-orphan")
    daily_trackers = relationship("DailyProfitTracker", back_populates="user", cascade="all, delete-orphan")
    trade_logs = relationship("TradeLog", back_populates="user", cascade="all, delete-orphan")
    analysis_logs = relationship("AnalysisLog", back_populates="user", cascade="all, delete-orphan")


class MT5Account(Base):
    """Stores user MT5 Credentials and MetaApi cloud tokens."""
    __tablename__ = "mt5_accounts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    meta_api_token = Column(String(255), nullable=False)
    account_id = Column(String(255), nullable=False)  # MetaApi Account ID
    login = Column(String(100), nullable=False)
    password = Column(String(255), nullable=False)
    server = Column(String(100), nullable=False)
    platform = Column(String(10), default="mt5")  # mt4 or mt5
    is_connected = Column(Boolean, default=False)
    bot_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

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
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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
    executed_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="trade_logs")


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
    analyzed_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="analysis_logs")


def init_db():
    """Initializes tables in database."""
    Base.metadata.create_all(bind=engine)


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
