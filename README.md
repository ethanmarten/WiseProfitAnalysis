# WiseProfit: Gemini AI + Local MetaTrader 5

WiseProfit is a Render-hosted Gemini AI signal service for XAUUSD. It analyzes M1, M5, and M15 candles using SMC/ICT methodology, enforces a $100 daily profit target, a maximum of five setups per day, and a high-impact news filter. Execution happens only on a local Windows MetaTrader 5 terminal through the Python `MetaTrader5` package.

---

## 🌟 Key Features

1. **24/7 Cloud Analysis**:
   - Render runs Gemini analysis and publishes signals; the local terminal executes them.
2. **AI-Powered SMC/ICT Analysis**:
   - Sends M1, M5, M15 OHLC candles to **Google Gemini AI** (`google-genai` SDK) with structured JSON prompts to detect:
     - **Liquidity Sweeps** (Buy-side / Sell-side liquidity grabs)
     - **CHoCH (Change of Character)** / Market Structure Shift (MSS)
     - **Fair Value Gaps (FVG)** & Order Blocks (OB) for precise entry.
3. **Risk & Capital Management**:
   - **$100 Daily Profit Cap**: Automatically locks trading for the day once cumulative profit hits $100 and auto-closes open positions.
   - **Max 5 Setups / Day**: Prevents over-trading.
   - **News Blackout Window**: Prevents trades 15 minutes before and after high-impact economic news events.
4. **Modern Dashboard UI**:
   - Responsive Glassmorphism dark-mode UI for setup counters, signal status, and audit trade logs.

---

## 📁 Project Architecture

```
gemini_mt5_saas/
├── main.py                # FastAPI server, REST API endpoints, and 24/7 background AI engine loop
├── gemini_analyzer.py     # Gemini AI client, SMC prompt engineering, and JSON signal parser
├── database.py            # SQLAlchemy models (User, MT5Account, DailyProfitTracker, TradeLog, PendingSignal)
├── news_filter.py         # Economic calendar evaluator (high-impact news guard)
├── auth.py / security.py  # Bearer-token auth + at-rest credential encryption
├── dashboard.html         # Responsive frontend web dashboard
├── bridge/
│   └── local_mt5_bot.py   # Optional: Windows MT5 executor that consumes /api/signals
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🔧 Installation & Setup Instructions

### 1. Clone & Install Dependencies
```bash
cd gemini_mt5_saas
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Set your Gemini API Key in your environment:
```bash
# Windows PowerShell:
$env:GEMINI_API_KEY="your_google_gemini_api_key"

# Linux/macOS:
export GEMINI_API_KEY="your_google_gemini_api_key"
```

### 3. Run the FastAPI Server
```bash
python main.py
# Or using uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access Dashboard & Connect MT5
1. Open your browser and navigate to `http://localhost:8000`.
2. Sign in and optionally save your local MT5 login/server label.
3. Start `bridge/local_mt5_bot.py` on the Windows computer running MT5.

---

## 🌉 Bridge Mode (Local Windows MT5 + Render AI)

The Render server analyzes markets and queues signals. The local Windows MT5
bridge is the only component that sends orders to a broker.

1. **On Render**, set the environment variable:
   ```
   BRIDGE_MODE=true
   ```
   Then redeploy. The Render engine queues each approved signal in the
   `pending_signals` table with a 90-second TTL.

2. **On your Windows PC** (where MT5 is already installed and logged in):
   ```powershell
   cd bridge
   pip install MetaTrader5 requests python-dotenv
   copy .env.example .env
   # edit .env and set RENDER_SERVER_URL and MT5_SYMBOL
   python local_mt5_bot.py
   ```

3. The bot polls `GET /api/signals`, receives one signal in this format, and
   places the order via `MetaTrader5.order_send`:

   ```json
   {"signal":{"id":"101","symbol":"XAUUSD","action":"BUY","lots":0.01,"sl":2030.5,"tp":2050.0,"timestamp":1710000000}}
   ```

   When no signal is available, the response is `{"signal": null}`. The
   endpoint reserves a signal on delivery so it cannot be sent twice.

The server never receives broker passwords and never executes trades through a
cloud trading SDK.
