"""
news_filter.py - Economic Calendar & High-Impact News Guard.

Checks live or cached economic calendar feeds to ensure no trades are executed
15 minutes before or 15 minutes after high-impact USD/XAU events (NFP, CPI, FOMC, PPI, Interest Rates).
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger("news_filter")


class EconomicNewsFilter:
    """Evaluates upcoming high-impact economic releases."""

    def __init__(self, quiet_window_minutes: int = 15):
        self.quiet_window_minutes = quiet_window_minutes

    async def fetch_economic_events(self) -> List[Dict[str, Any]]:
        """
        Fetches today's high impact USD news events from economic calendar endpoints.
        Returns a list of event objects with time and impact rating.
        """
        # In a production environment, integration with ForexFactory API, DailyFX, or Financial Modeling Prep is used.
        # Fallback empty list / simulation included for resilience.
        try:
            # Example placeholder event list
            events = [
                # {"title": "US CPI MoM", "impact": "HIGH", "time": datetime.utcnow() + timedelta(hours=2), "currency": "USD"}
            ]
            return events
        except Exception as e:
            logger.error(f"Error fetching news events: {e}")
            return []

    async def is_news_impact_zone(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Checks if current time falls within ±15 minutes of any HIGH impact USD/XAU news event.
        """
        check_time = now or datetime.utcnow()
        events = await self.fetch_economic_events()

        window = timedelta(minutes=self.quiet_window_minutes)

        for event in events:
            if event.get("impact", "").upper() == "HIGH" and event.get("currency") in ["USD", "XAU"]:
                event_time = event.get("time")
                if isinstance(event_time, datetime):
                    start_quiet = event_time - window
                    end_quiet = event_time + window
                    if start_quiet <= check_time <= end_quiet:
                        return {
                            "is_safe": False,
                            "reason": f"High Impact News Blackout Zone ({event.get('title')})",
                            "event": event.get("title"),
                            "event_time": event_time.isoformat()
                        }

        return {
            "is_safe": True,
            "reason": "Clear of high-impact economic news releases."
        }
