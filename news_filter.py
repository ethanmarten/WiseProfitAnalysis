"""
news_filter.py - Economic Calendar & High-Impact News Guard.

Pulls this week's economic calendar from the public ForexFactory JSON feed and
blocks trading within +/- 15 minutes of any high-impact USD/XAU release
(NFP, CPI, FOMC, PPI, Interest Rate decisions).

The feed is cached in memory for one hour so the 60-second trading loop does not
hammer the upstream endpoint, and every failure path degrades to "safe to trade"
only when explicitly configured, otherwise it fails closed on fetch errors.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import requests

logger = logging.getLogger("news_filter")

# Public weekly calendar feed (no API key required).
FOREXFACTORY_FEED = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

CACHE_TTL_SECONDS = 3600
HIGH_IMPACT_CURRENCIES = {"USD", "XAU", "ALL"}

# Titles that always count as high impact for gold, even if the feed
# mislabels their impact rating.
CRITICAL_KEYWORDS = (
    "non-farm", "nonfarm", "nfp", "cpi", "fomc", "ppi",
    "federal funds", "interest rate", "unemployment rate", "gdp",
    "powell", "core pce",
)


def _utcnow() -> datetime:
    """Naive UTC timestamp used consistently across comparisons."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class EconomicNewsFilter:
    """Evaluates upcoming high-impact economic releases."""

    def __init__(self, quiet_window_minutes: int = 15, fail_open: Optional[bool] = None):
        self.quiet_window_minutes = quiet_window_minutes
        # When the calendar cannot be fetched we block trading by default; set
        # NEWS_FILTER_FAIL_OPEN=true to keep trading during feed outages.
        if fail_open is None:
            fail_open = os.getenv("NEWS_FILTER_FAIL_OPEN", "false").lower() == "true"
        self.fail_open = fail_open
        self._cache: List[Dict[str, Any]] = []
        self._cache_time: Optional[datetime] = None
        self._last_fetch_failed = False

    # ------------------------------------------------------------------
    # Feed handling
    # ------------------------------------------------------------------
    def _parse_event_time(self, raw: Optional[str]) -> Optional[datetime]:
        """Parses ForexFactory ISO timestamps (e.g. 2026-01-05T13:30:00-05:00) to naive UTC."""
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    def _fetch_feed_sync(self) -> List[Dict[str, Any]]:
        """Blocking HTTP call executed in a worker thread."""
        response = requests.get(FOREXFACTORY_FEED, timeout=8, headers={"User-Agent": "WiseProfit/1.0"})
        response.raise_for_status()
        payload = response.json()

        events: List[Dict[str, Any]] = []
        for item in payload:
            event_time = self._parse_event_time(item.get("date"))
            if not event_time:
                continue
            title = (item.get("title") or "").strip()
            impact = (item.get("impact") or "").strip().upper()
            currency = (item.get("country") or item.get("currency") or "").strip().upper()

            is_critical = any(keyword in title.lower() for keyword in CRITICAL_KEYWORDS)
            if impact == "HIGH" or is_critical:
                impact = "HIGH"

            events.append({
                "title": title,
                "impact": impact,
                "time": event_time,
                "currency": currency,
            })
        return events

    async def fetch_economic_events(self) -> List[Dict[str, Any]]:
        """Returns this week's calendar events, cached for one hour."""
        now = _utcnow()
        if self._cache_time and (now - self._cache_time).total_seconds() < CACHE_TTL_SECONDS:
            return self._cache

        try:
            events = await asyncio.to_thread(self._fetch_feed_sync)
            self._cache = events
            self._cache_time = now
            self._last_fetch_failed = False
            logger.info(f"Economic calendar refreshed: {len(events)} events loaded.")
            return events
        except Exception as e:  # noqa: BLE001 - network/parse errors
            logger.error(f"Error fetching news events: {e}")
            self._last_fetch_failed = True
            # Serve a stale cache rather than trading blind.
            return self._cache

    # ------------------------------------------------------------------
    # Guard evaluation
    # ------------------------------------------------------------------
    async def is_news_impact_zone(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Checks if the current time falls within the blackout window of a high-impact event."""
        check_time = now or _utcnow()
        events = await self.fetch_economic_events()

        if self._last_fetch_failed and not events and not self.fail_open:
            return {
                "is_safe": False,
                "reason": "Economic calendar unavailable; trading paused (fail-closed).",
                "event": None,
                "event_time": None,
            }

        window = timedelta(minutes=self.quiet_window_minutes)

        for event in events:
            if event.get("impact") != "HIGH":
                continue
            if event.get("currency") not in HIGH_IMPACT_CURRENCIES:
                continue
            event_time = event.get("time")
            if not isinstance(event_time, datetime):
                continue
            if event_time - window <= check_time <= event_time + window:
                return {
                    "is_safe": False,
                    "reason": f"High Impact News Blackout Zone ({event.get('title')})",
                    "event": event.get("title"),
                    "event_time": event_time.isoformat(),
                }

        return {
            "is_safe": True,
            "reason": "Clear of high-impact economic news releases.",
        }

    async def next_high_impact_event(self, now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Returns the next upcoming high-impact event, useful for dashboard display."""
        check_time = now or _utcnow()
        events = await self.fetch_economic_events()
        upcoming = [
            e for e in events
            if e.get("impact") == "HIGH"
            and e.get("currency") in HIGH_IMPACT_CURRENCIES
            and isinstance(e.get("time"), datetime)
            and e["time"] >= check_time
        ]
        if not upcoming:
            return None
        nxt = min(upcoming, key=lambda e: e["time"])
        return {
            "title": nxt["title"],
            "currency": nxt["currency"],
            "time": nxt["time"].isoformat(),
            "minutes_away": int((nxt["time"] - check_time).total_seconds() // 60),
        }
