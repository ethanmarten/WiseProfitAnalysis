import os
import time
import requests
import MetaTrader5 as mt5
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

RENDER_URL = os.getenv("RENDER_SERVER_URL", "https://wiseprofitanalysis.onrender.com")

# Global variable to prevent executing the same signal multiple times
last_processed_signal_id = None

def initialize_mt5():
    """Initializes the connection to MetaTrader 5"""
    if not mt5.initialize():
        print("[-] Failed to connect to MT5:", mt5.last_error())
        return False
    
    acc = mt5.account_info()
    if acc is None:
        print("[-] Failed to retrieve account information.")
        return False

    print("==========================================")
    print("   Bot Started & Connected Successfully   ")
    print(f"   Account: {acc.login} | Balance: {acc.balance} USD")
    print(f"   Server URL: {RENDER_URL}")
    print("==========================================")
    return True

def get_filling_type(symbol):
    """Automatically selects the appropriate filling type for the broker"""
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.ORDER_FILLING_IOC
    
    filling_modes = info.filling_mode
    if filling_modes & mt5.ORDER_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    elif filling_modes & mt5.ORDER_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    else:
        return mt5.ORDER_FILLING_RETURN

def execute_trade(symbol, action, lots, sl=None, tp=None):
    """Executes trade orders safely on MetaTrader 5"""
    # Enable symbol in Market Watch if not already selected
    if not mt5.symbol_select(symbol, True):
        print(f"[-] Failed to select symbol {symbol} on MT5 platform.")
        return False

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"[-] Failed to get live tick data for {symbol}.")
        return False

    is_buy = action.upper() == "BUY"
    trade_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    price = tick.ask if is_buy else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lots),
        "type": trade_type,
        "price": price,
        "magic": 100200,
        "comment": "WiseProfit_Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": get_filling_type(symbol),
    }

    if sl: request["sl"] = float(sl)
    if tp: request["tp"] = float(tp)

    result = mt5.order_send(request)
    
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err_msg = result.comment if result else "Unknown Error"
        print(f"\n[X] Trade execution failed for {symbol}: {err_msg}")
        return False
    else:
        print(f"\n[+] {action} order executed successfully on {symbol} at price {price}")
        return True

def start_bot_loop():
    global last_processed_signal_id
    print("\nMonitoring market & fetching signals from server...")
    
    try:
        while True:
            # 1. Display live XAUUSD tick prices
            if mt5.symbol_select("XAUUSD", True):
                tick = mt5.symbol_info_tick("XAUUSD")
                if tick:
                    print(f"[{time.strftime('%H:%M:%S')}] XAUUSD - Bid: {tick.bid} | Ask: {tick.ask} | Waiting for signal...", end="\r")

            # 2. Query Render server for signals
            try:
                res = requests.get(f"{RENDER_URL}/api/signals", timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    
                    if data.get("signal"):
                        sig = data["signal"]
                        signal_id = sig.get("id") or f"{sig['symbol']}_{sig['action']}_{sig.get('timestamp')}"
                        
                        # Prevent duplicate trade execution
                        if signal_id != last_processed_signal_id:
                            print(f"\n[!] New signal received: {sig['action']} on {sig['symbol']}")
                            
                            success = execute_trade(
                                symbol=sig['symbol'], 
                                action=sig['action'], 
                                lots=sig.get('lots', 0.01), 
                                sl=sig.get('sl'), 
                                tp=sig.get('tp')
                            )
                            
                            if success:
                                last_processed_signal_id = signal_id

            except Exception:
                pass  # Continue loop silently if server connection drops temporarily

            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nBot stopped successfully by user.")
        mt5.shutdown()

if __name__ == "__main__":
    if initialize_mt5():
        start_bot_loop()
