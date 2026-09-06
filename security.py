"""
security.py - Password hashing, session tokens, and at-rest encryption.

Design goals:
  * No heavy dependencies: password hashing uses stdlib PBKDF2-HMAC-SHA256.
  * Credentials (MetaApi token, MT5 password) are encrypted with Fernet using a
    key derived from the SECRET_KEY environment variable, so a leaked database
    file alone does not expose live trading accounts.
  * Session tokens are opaque random strings; only their SHA-256 hash is stored.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
from typing import Optional

logger = logging.getLogger("security")

PBKDF2_ITERATIONS = 240_000
_SALT_BYTES = 16

# ---------------------------------------------------------------------------
# Secret key resolution
# ---------------------------------------------------------------------------
_DEV_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".secret_key")


def _resolve_secret_key() -> str:
    """Returns the app secret. Falls back to a persisted local dev key."""
    key = os.getenv("SECRET_KEY", "").strip()
    if key:
        return key

    # Development fallback: persist a generated key so encrypted rows stay
    # readable across restarts. Never used when SECRET_KEY is configured.
    try:
        if os.path.exists(_DEV_KEY_FILE):
            with open(_DEV_KEY_FILE, "r", encoding="utf-8") as fh:
                stored = fh.read().strip()
                if stored:
                    return stored
        generated = secrets.token_urlsafe(48)
        with open(_DEV_KEY_FILE, "w", encoding="utf-8") as fh:
            fh.write(generated)
        logger.warning(
            "SECRET_KEY not set. Generated a development key at .secret_key. "
            "Set SECRET_KEY in the environment before deploying to production."
        )
        return generated
    except OSError:
        logger.error("Could not persist a development secret key; using an ephemeral one.")
        return secrets.token_urlsafe(48)


SECRET_KEY = _resolve_secret_key()


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Returns 'pbkdf2_sha256$iterations$salt$hash' for the given password."""
    if not password:
        raise ValueError("Password must not be empty.")
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return "$".join([
        "pbkdf2_sha256",
        str(PBKDF2_ITERATIONS),
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    ])


def verify_password(password: str, stored: Optional[str]) -> bool:
    """Constant-time verification of a password against a stored hash."""
    if not password or not stored:
        return False
    try:
        algorithm, iterations, salt_b64, hash_b64 = stored.split("$")
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    try:
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(iterations)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(candidate, expected)


# ---------------------------------------------------------------------------
# Session tokens
# ---------------------------------------------------------------------------
def generate_session_token() -> str:
    """Creates an opaque bearer token to hand to the browser."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """Stores only the hash of a token so DB reads cannot impersonate users."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Credential encryption at rest
# ---------------------------------------------------------------------------
_ENCRYPTION_PREFIX = "enc:v1:"


def _fernet():
    """Builds a Fernet cipher from SECRET_KEY, or None if unavailable."""
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        logger.warning(
            "cryptography is not installed; MT5 credentials will be stored in plaintext. "
            "Run `pip install cryptography` to enable encryption at rest."
        )
        return None
    key_material = hashlib.sha256(SECRET_KEY.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(key_material))


def encrypt_secret(value: Optional[str]) -> Optional[str]:
    """Encrypts a sensitive value for database storage."""
    if value is None or value == "":
        return value
    if value.startswith(_ENCRYPTION_PREFIX):
        return value  # Already encrypted.
    cipher = _fernet()
    if cipher is None:
        return value
    token = cipher.encrypt(value.encode("utf-8")).decode("ascii")
    return _ENCRYPTION_PREFIX + token


def decrypt_secret(value: Optional[str]) -> Optional[str]:
    """Decrypts a stored value; passes through legacy plaintext rows unchanged."""
    if not value or not value.startswith(_ENCRYPTION_PREFIX):
        return value  # Legacy plaintext row written before encryption existed.
    cipher = _fernet()
    if cipher is None:
        return None
    try:
        from cryptography.fernet import InvalidToken
        return cipher.decrypt(value[len(_ENCRYPTION_PREFIX):].encode("ascii")).decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - InvalidToken and friends
        logger.error(f"Failed to decrypt stored credential (wrong SECRET_KEY?): {exc}")
        return None
