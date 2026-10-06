# Supabase setup

WiseProfit uses Supabase PostgreSQL through SQLAlchemy. No Supabase SDK is required for the backend: the existing models and migrations create and update the tables in the connected database.

## 1. Create the database connection

1. Create or open the Supabase project.
2. Open **Project Settings > Database**.
3. Copy the **Transaction Pooler** connection string.
4. Replace `[YOUR-PASSWORD]` with the database password.
5. Set this value as `DATABASE_URL` in Render and in the local `.env` when running locally:

```text
postgresql://postgres.[PROJECT-REF]:[PASSWORD]@[POOLER-HOST]:6543/postgres
```

Do not commit the connection string. `psycopg2-binary` is already in `requirements.txt`.

## 2. Configure Render

Set these environment variables in the Render service:

- `DATABASE_URL`: the Supabase Transaction Pooler URI
- `SECRET_KEY`: a new random value
- `GEMINI_API_KEY`: a valid Gemini key
- `GROQ_API_KEY`: optional fallback
- `OPENROUTER_API_KEY`: optional third fallback
- `OPENROUTER_MODEL`: for example `google/gemini-2.5-flash`

Deploy once. On startup, `database.py` runs `create_all()` and the lightweight migrations. This creates the users, MT5 accounts, daily trackers, analyses, pending signals, trade logs, and sessions tables. It also adds the persisted MT5 balance, equity, open positions, and state timestamp columns to existing installations.

## 3. Supabase dashboard checks

After the first successful deployment, open **Table Editor** and confirm these tables exist:

- `users`
- `user_sessions`
- `mt5_accounts`
- `daily_profit_trackers`
- `analysis_logs`
- `pending_signals`
- `trade_logs`

The backend connects with the database owner through the server-side `DATABASE_URL`, so these tables do not need public anonymous access. Do not expose the database password or service-role key in browser JavaScript.

## 4. MT5 verification

1. Put `WP_EMAIL`, `WP_PASSWORD`, and `RENDER_SERVER_URL` in `bridge/.env`.
2. Start the local bridge while MetaTrader 5 is logged in.
3. Confirm the bridge log says that the account registered and market data uploaded.
4. Refresh the dashboard. `MT5 Account Equity` and `Balance` should match the local terminal.
5. The bridge sends open positions every market-data interval and sends closed deals from the previous two days. The dashboard then updates unrealized/realized PnL and signed trade profit.

If the values are still zero, inspect the bridge log for a `401`, `403`, or `/api/bridge/account-state` error. The local terminal must be logged in and the bridge must use the same WiseProfit account as the dashboard.
