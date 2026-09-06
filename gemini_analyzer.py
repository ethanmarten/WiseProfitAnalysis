"""
gemini_analyzer.py - SMC/ICT Gold Scalping Analysis via Google Gemini AI.

Fetches/accepts M1, M5, M15 OHLC candle data for XAUUSD, constructs an SMC prompt
instructing Gemini to detect Liquidity Sweeps, Change of Character (CHoCH), and Fair Value Gaps (FVG),
and returns structured JSON signals.
"""

import asyncio
import os
import json
import logging
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger("gemini_analyzer")

# Model is configurable so a deprecated default never silently breaks the bot.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Minimum acceptable risk-to-reward ratio enforced locally, independent of what
# the model claims in its response.
MIN_RISK_REWARD = float(os.getenv("MIN_RISK_REWARD", "1.8"))

# Define Pydantic schema for strict Gemini JSON structured output
class SMCTradeSignal(BaseModel):
    action: str = Field(description="Trade recommendation: 'BUY', 'SELL', or 'HOLD'")
    confidence: float = Field(description="Confidence level between 0.0 and 1.0")
    entry_price: Optional[float] = Field(default=None, description="Suggested market or limit entry price")
    stop_loss: Optional[float] = Field(default=None, description="Strict stop loss price in USD")
    take_profit: Optional[float] = Field(default=None, description="Target take profit price in USD")
    risk_reward_ratio: Optional[float] = Field(default=None, description="Risk to Reward Ratio (e.g. 2.5)")
    setup_type: Optional[str] = Field(default=None, description="Identified SMC setup: 'Bullish FVG', 'Bearish FVG', 'CHoCH + Sweep', etc.")
    reasoning: str = Field(description="Detailed breakdown of Liquidity Sweep, CHoCH, and FVG detection")


class GeminiSMCAnalyzer:
    """Gemini AI market analyzer enforcing SMC/ICT trading methodology."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._is_valid_key(self.api_key):
            logger.warning("GEMINI_API_KEY is missing or set to placeholder. Analyzer will return HOLD.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name or DEFAULT_MODEL

    def _is_valid_key(self, key: Optional[str]) -> bool:
        """Helper to verify key is present and not a placeholder."""
        if not key:
            return False
        invalid_placeholders = ["your_gemini_api_key_here", "your_api_key", "change_me", "xxx"]
        return key.strip().lower() not in invalid_placeholders

    def format_candles_summary(self, candles: List[Dict[str, Any]]) -> str:
        """Formats raw candle lists into a clean text table string for Gemini prompt."""
        lines = ["Time | Open | High | Low | Close | Volume"]
        for c in candles[-20:]:  # Take last 20 candles per timeframe
            time_str = str(c.get("time", ""))
            o = float(c.get('open') or 0.0)
            h = float(c.get('high') or 0.0)
            l = float(c.get('low') or 0.0)
            cl = float(c.get('close') or 0.0)
            v = float(c.get('volume') or 0.0)
            lines.append(
                f"{time_str} | {o:.2f} | {h:.2f} | {l:.2f} | {cl:.2f} | {v:.0f}"
            )
        return "\n".join(lines)

    async def analyze_market(
        self,
        m1_candles: List[Dict[str, Any]],
        m5_candles: List[Dict[str, Any]],
        m15_candles: List[Dict[str, Any]],
        current_price: float
    ) -> Dict[str, Any]:
        """
        Queries Gemini AI with M1, M5, M15 candle data to evaluate SMC/ICT setups for XAUUSD.
        Returns a dictionary matching SMCTradeSignal.
        """
        # Attempt to pick up GEMINI_API_KEY dynamically if environment changed
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass

        env_key = os.getenv("GEMINI_API_KEY")
        if self._is_valid_key(env_key):
            self.api_key = env_key
            self.client = genai.Client(api_key=env_key)

        if not self._is_valid_key(self.api_key):
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": "GEMINI_API_KEY is missing or invalid placeholder in .env. Please set a valid Gemini API key from AI Studio."
            }

        if not m1_candles and not m5_candles and not m15_candles:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": "No candle data available for analysis; refusing to trade blind.",
            }

        m1_text = self.format_candles_summary(m1_candles)
        m5_text = self.format_candles_summary(m5_candles)
        m15_text = self.format_candles_summary(m15_candles)

        system_instruction = """
        You are an elite Institutional Algo Trader specializing in Smart Money Concepts (SMC) and Inner Circle Trader (ICT) methodology for XAUUSD (Gold).
        Your core objective is high-probability intraday scalping.
        
        Strict Rules for Analysis:
        1. Identify Higher Timeframe (M15) bias, Key Highs/Lows, and Premium/Discount Zones.
        2. Identify M5 Change of Character (CHoCH) or Market Structure Shift (MSS).
        3. Identify M1/M5 Fair Value Gap (FVG) or Order Block (OB) for precise entry.
        4. Detect Liquidity Sweeps (ruling out false breakouts above buy-side or below sell-side liquidity).
        5. Stop Loss MUST be placed beyond the recent swing high/low (max 30-50 pips / $3-$5 in Gold).
        6. Take Profit MUST yield a Risk-to-Reward Ratio (RRR) of at least 1:2.0.
        7. If conditions are ambiguous, returning 'HOLD' is MANDATORY. Never force trades.
        """

        prompt = f"""
        Current XAUUSD Market Price: ${current_price:.2f}

        --- M15 timeframe (HTF Bias & Structure) ---
        {m15_text}

        --- M5 timeframe (Structure Change CHoCH & FVG Context) ---
        {m5_text}

        --- M1 timeframe (Micro Entry & Liquidity Sweep) ---
        {m1_text}

        Evaluate these multi-timeframe candle datasets for XAUUSD. Determine if there is a valid SMC/ICT scalping setup active right now.
        Respond STRICTLY with the requested JSON schema.
        """

        try:
            # google-genai's generate_content is blocking; run it off the event
            # loop so the 24/7 trading engine and API requests are not stalled.
            response = await asyncio.to_thread(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=SMCTradeSignal,
                        temperature=0.1,  # Low temperature for deterministic analysis
                    ),
                )
            )

            raw_json = response.text
            signal_data = json.loads(raw_json)
            signal_data.setdefault("reasoning", "")

            # Extra sanity check for Gold (XAUUSD) SL/TP boundaries
            action = signal_data.get("action", "HOLD").upper()
            if action in ["BUY", "SELL"]:
                entry = signal_data.get("entry_price") or current_price
                sl = signal_data.get("stop_loss")
                tp = signal_data.get("take_profit")

                def reject(message: str) -> None:
                    signal_data["action"] = "HOLD"
                    signal_data["reasoning"] += f" [Rejected: {message}]"

                if not sl or not tp:
                    reject("Missing SL or TP values")
                elif action == "BUY" and (sl >= entry or tp <= entry):
                    reject("Invalid BUY SL/TP geometry")
                elif action == "SELL" and (sl <= entry or tp >= entry):
                    reject("Invalid SELL SL/TP geometry")
                else:
                    # Independently verify the risk-to-reward ratio rather than
                    # trusting the model's self-reported number.
                    risk = abs(entry - sl)
                    reward = abs(tp - entry)
                    if risk <= 0:
                        reject("Zero-distance stop loss")
                    else:
                        actual_rr = reward / risk
                        signal_data["risk_reward_ratio"] = round(actual_rr, 2)
                        if actual_rr < MIN_RISK_REWARD:
                            reject(f"Risk-reward {actual_rr:.2f} below minimum {MIN_RISK_REWARD}")

            return signal_data

        except Exception as e:
            logger.error(f"Error during Gemini SMC market analysis: {e}", exc_info=True)
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Gemini API Analysis Error: {str(e)}"
            }
