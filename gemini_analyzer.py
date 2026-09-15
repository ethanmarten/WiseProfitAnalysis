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
import re
from random import uniform
from typing import Dict, Any, List, Optional
import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger("gemini_analyzer")

# Model is configurable; use the current model recommended by the API response.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# Keep the current supported model as the only default. A fallback must be
# explicitly configured because model availability differs between accounts.
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()
MODEL_RETRIES = max(1, int(os.getenv("GEMINI_MODEL_RETRIES", "2")))
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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

    @staticmethod
    def _is_valid_key(key: Optional[str]) -> bool:
        if not key:
            return False
        invalid_placeholders = {"your_api_key", "change_me", "xxx"}
        return key.strip().lower() not in invalid_placeholders

    async def _analyze_with_groq(
        self,
        prompt: str,
        system_instruction: str,
    ) -> Dict[str, Any]:
        """Uses Groq's OpenAI-compatible API when Gemini is unavailable."""
        groq_key = os.getenv("GROQ_API_KEY")
        if not self._is_valid_key(groq_key):
            raise RuntimeError("GROQ_API_KEY is not configured")

        def request_sync() -> Dict[str, Any]:
            response = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.getenv("GROQ_MODEL", GROQ_MODEL),
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content)
            content = str(content).strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError("Groq returned a non-object JSON response")
            return result

        result = await asyncio.to_thread(request_sync)
        result.setdefault("reasoning", "")
        return result

    @staticmethod
    def _is_transient_api_error(error: Exception) -> bool:
        """Returns true for temporary provider capacity or rate-limit errors."""
        message = str(error).lower()
        return bool(re.search(r"(?:code['\"]?\s*[:=]\s*|\b)(429|500|502|503|504)\b", message))

    async def _generate_content(
        self,
        model: str,
        prompt: str,
        system_instruction: str,
    ) -> Any:
        """Calls Gemini with bounded retries for temporary service failures."""
        for attempt in range(MODEL_RETRIES):
            try:
                return await asyncio.to_thread(
                    lambda: self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            response_mime_type="application/json",
                            response_schema=SMCTradeSignal,
                            temperature=0.1,
                        ),
                    )
                )
            except Exception as error:
                if not self._is_transient_api_error(error) or attempt == MODEL_RETRIES - 1:
                    raise
                delay = (2 ** attempt) + uniform(0, 0.5)
                logger.warning(
                    "Gemini model %s returned a temporary error; retrying in %.1fs (%s/%s): %s",
                    model,
                    delay,
                    attempt + 1,
                    MODEL_RETRIES - 1,
                    error,
                )
                await asyncio.sleep(delay)

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

        if not self._is_valid_key(self.api_key) and not self._is_valid_key(os.getenv("GROQ_API_KEY")):
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": "No valid GEMINI_API_KEY or GROQ_API_KEY is configured."
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
        You are a disciplined institutional market-structure analyst for XAUUSD and other liquid assets.
        Read the market as a complete auction, not as an isolated candle or indicator.

        Analyze in this order:
        1. Establish M15 directional bias, trend/range condition, dealing range, premium/discount,
           recent swing highs/lows, and important support/resistance.
        2. Use M5 to confirm or reject that bias through BOS, CHoCH/MSS, displacement, and whether
           price is accepting or rejecting a key level.
        3. Use M1 only for execution timing: liquidity sweep, FVG or order-block reaction,
           retest quality, and evidence of trapped traders or a false breakout.
        4. Explain who is likely in control (buyers, sellers, or neither), where liquidity rests,
           and whether the current move is continuation, reversal, or noise.
        5. Do not invent news or fundamental events. Treat news as unknown unless it is supplied.
        6. Return BUY or SELL only when the higher-timeframe bias, structure confirmation, and
           execution trigger agree. Otherwise return HOLD with confidence 0.0-0.69.
        7. For BUY: stop_loss < entry_price < take_profit. For SELL: take_profit < entry_price < stop_loss.
           Risk-reward must be at least 1.8, and the stop must be beyond meaningful structure.
        8. Return ONLY valid JSON with exactly these fields: action, confidence, entry_price,
           stop_loss, take_profit, risk_reward_ratio, setup_type, reasoning.
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
            if self.client is None:
                raise RuntimeError("Gemini client is not configured")

            models = [self.model_name]
            if FALLBACK_MODEL and FALLBACK_MODEL not in models:
                models.append(FALLBACK_MODEL)
            response = None
            last_error = None
            for model in models:
                try:
                    response = await self._generate_content(model, prompt, system_instruction)
                    break
                except Exception as error:
                    last_error = error
                    if not self._is_transient_api_error(error) or model == models[-1]:
                        raise
                    logger.warning("Gemini model %s unavailable; trying configured model fallback: %s", model, error)

            if response is None:
                raise last_error or RuntimeError("Gemini returned no response")

            signal_data = json.loads(response.text)
            signal_data.setdefault("reasoning", "")

        except Exception as gemini_error:
            groq_key = os.getenv("GROQ_API_KEY")
            if not self._is_valid_key(groq_key):
                raise
            logger.warning("Gemini analysis failed; switching to Groq: %s", gemini_error)
            signal_data = await self._analyze_with_groq(prompt, system_instruction)

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
