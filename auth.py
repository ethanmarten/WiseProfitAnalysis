"""
auth.py - Authentication dependencies for the WiseProfit API.

Provides:
  * create_session / revoke_session helpers backed by the UserSession table.
  * `current_user` FastAPI dependency that resolves an Authorization bearer
    token to a User, rejecting missing/expired tokens with HTTP 401.
  * `authorize_user_id` guard so one authenticated user cannot read or mutate
    another user's account, trades, or dashboard by changing a path parameter.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from database import User, UserSession, get_db, utcnow
from security import generate_session_token, hash_token

logger = logging.getLogger("auth")

SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "72"))


def create_session(db: Session, user: User) -> str:
    """Issues a new bearer token for the user and returns the plaintext token."""
    token = generate_session_token()
    record = UserSession(
        user_id=user.id,
        token_hash=hash_token(token),
        expires_at=utcnow() + timedelta(hours=SESSION_TTL_HOURS),
    )
    db.add(record)
    db.commit()
    return token


def revoke_session(db: Session, token: str) -> bool:
    """Deletes the session matching the supplied token."""
    deleted = db.query(UserSession).filter(
        UserSession.token_hash == hash_token(token)
    ).delete()
    db.commit()
    return bool(deleted)


def _extract_token(request: Request) -> Optional[str]:
    """Reads the bearer token from the Authorization header or X-Auth-Token."""
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    fallback = request.headers.get("x-auth-token")
    return fallback.strip() if fallback else None


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Resolves the authenticated user or raises HTTP 401."""
    token = _extract_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Sign in to obtain an access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    session = db.query(UserSession).filter(
        UserSession.token_hash == hash_token(token)
    ).first()

    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token.")

    if session.is_expired:
        db.delete(session)
        db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired. Please sign in again.")

    user = db.query(User).filter(User.id == session.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is inactive.")

    return user


def authorize_user_id(user: User, requested_user_id: int) -> None:
    """Blocks horizontal privilege escalation across user IDs."""
    if user.id != requested_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to access another user's data.",
        )
