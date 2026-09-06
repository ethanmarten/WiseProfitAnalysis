# 🚀 Gemini AI + MetaTrader 5 (MetaApi) Cloud SaaS Trading Platform

A fully automated, 24/7 cloud-based algorithmic trading platform that integrates **Google Gemini AI** with **MetaTrader 5 (MT5)** via the **MetaApi Cloud SDK**. The bot analyzes **XAUUSD (Gold)** on M1, M5, and M15 timeframes using **Smart Money Concepts (SMC) & Inner Circle Trader (ICT)** methodology, enforcing a strict **$100 daily profit target cap** and maximum **5 setups per day**.

---

## 🌟 Key Features

1. **24/7 Cloud Execution (No PC Required)**:
   - Uses `metaapi-cloud-sdk` instead of local MT5 bindings, allowing seamless execution on cloud servers (AWS, DigitalOcean, GCP, Render).
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
   - Responsive Glassmorphism dark-mode UI to enter MT5 & MetaApi credentials, view real-time PnL, setup counters, equity, and audit trade logs.

---

## 📁 Project Architecture

```
gemini_mt5_saas/
├── main.py                # FastAPI server, REST API endpoints, and 24/7 background AI engine loop
├── gemini_analyzer.py     # Gemini AI client, SMC prompt engineering, and JSON signal parser
├── mt5_executor.py        # MetaApi Cloud SDK integration, position execution, SL/TP management
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
2. Input your **User Email**, **MetaApi Token**, **Account ID**, **MT5 Login**, **Password**, and **Broker Server**.
3. Click **Save & Connect 24/7 Bot**. The system will start analyzing Gold (XAUUSD) and trading automatically!

---

## 🌉 Bridge Mode (Local Windows MT5 + Render AI)

By default the Render server analyzes markets **and** executes trades through
MetaApi Cloud. If your broker blocks MetaApi — or you simply prefer executing
on your local Windows MT5 terminal — switch the system to **Bridge Mode**.

1. **On Render**, set the environment variable:
   ```
   BRIDGE_MODE=true
   ```
   Then redeploy. The Render engine will keep producing Gemini signals, but
   instead of calling MetaApi it queues each signal in the `pending_signals`
   table with a 90-second TTL.

2. **On your Windows PC** (where MT5 is already installed and logged in):
   ```powershell
   cd bridge
   pip install MetaTrader5 requests python-dotenv
   copy .env.example .env
   # edit .env and set WP_EMAIL, WP_PASSWORD, MT5_SYMBOL
   python local_mt5_bot.py
   ```

3. The bot signs in to your WiseProfit dashboard, polls
   `GET /api/signals/pending`, claims each signal with a one-shot token,
   places the order via `MetaTrader5.order_send`, then POSTs the result back
   to `POST /api/signals/{id}/ack`. The dashboard's audit log shows every
   trade as if Render had executed it itself.

> **Tip:** to disable bridge mode, set `BRIDGE_MODE=false` (or remove the
> var) on Render and the bot will go back to using MetaApi directly.
