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
├── database.py            # SQLAlchemy models (User, MT5Account, DailyProfitTracker, TradeLog)
├── news_filter.py         # Economic calendar evaluator (high-impact news guard)
├── dashboard.html         # Responsive frontend web dashboard
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
