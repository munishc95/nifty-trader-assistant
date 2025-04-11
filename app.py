import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import pytz
import requests
import os

st.set_page_config(layout="wide")

# ✅ Initialize demo_mode in session_state
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False

# ✅ Sidebar toggle for Demo Mode
st.sidebar.checkbox("🧪 Enable Demo Mode", value=st.session_state.demo_mode, key="demo_mode", help="Simulates market hours and option prices for testing on weekends.")

# ✅ Override current time if Demo Mode is enabled
if st.session_state.demo_mode:
    now = pytz.timezone('Asia/Kolkata').localize(datetime.datetime(2025, 4, 10, 10, 0))  # Pick a past weekday
else:
    now = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))



# Market hours check (before autorefresh)
now = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

if not st.session_state.demo_mode and not (market_open <= now <= market_close):
    st.warning("⏸ Market is closed. App will resume at 9:15 AM IST.")
    
    # Show last known price (closing or fallback)
    end_of_day = now.replace(hour=15, minute=30, second=0, microsecond=0)
    start_of_day = now.replace(hour=9, minute=15, second=0, microsecond=0)
    df = yf.download("^NSEI", interval="1m", start=start_of_day, end=end_of_day)
    if not df.empty:
        closing_price = df['Close'].iloc[-1]
        st.metric("🔒 Closing Price", f"{float(closing_price):.2f}")
    else:
        st.info("No intraday data available to show closing price.")

    # Show daily P&L summary or fallback
    st.markdown("---")
    st.subheader("📋 Daily Trade Summary")
    if 'trades' in st.session_state:
        trades_df = pd.DataFrame(st.session_state.trades)
        if not trades_df.empty:
            trades_df['Date'] = pd.to_datetime(trades_df['Buy Time']).dt.date
            today = now.date()
            today_trades = trades_df[trades_df['Date'] == today]
            if not today_trades.empty:
                total_pnl = today_trades['PnL'].sum()
                st.metric("🔄 Daily P&L", f"₹{total_pnl:,.2f}")
            else:
                st.info("💬 0 trades were taken today.")
        else:
            st.info("💬 0 trades were taken today.")
    else:
        st.info("💬 0 trades were taken today.")

    st.stop()


st_autorefresh(interval=1000, limit=None, key="data_refresh")

st.title("🔴 NIFTY Live Paper Trading - Intraday Strategy")

if 'capital' not in st.session_state:
    st.session_state.capital = 10000
    st.session_state.hold = False
    st.session_state.entry_price = 0.0
    st.session_state.position = None
    st.session_state.trades = []
    st.session_state.option_data = None

# Telegram Alert
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, data=payload)
    except Exception as e:
        st.error(f"Telegram alert failed: {e}")

# Strategy logic
def apply_strategy(df):
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Signal'] = 0
    df.loc[(df['EMA_5'] > df['EMA_20']) & (df['RSI'] > 50), 'Signal'] = 1
    df.loc[(df['EMA_5'] < df['EMA_20']) & (df['RSI'] < 50), 'Signal'] = -1
    return df

# Fetch NIFTY price data
def fetch_data():
    now = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))

    # DEMO MODE: return synthetic price data for testing
    if st.session_state.demo_mode:
        st.info("🧪 Demo Mode is enabled. Data shown is simulated for testing purposes.")
        timestamps = pd.date_range(end=now, periods=5, freq='1min')
        prices = np.linspace(22800, 22880, num=5)
        df = pd.DataFrame({
            'Datetime': timestamps,
            'Close': prices
        })
        df.set_index('Datetime', inplace=True)
        return df

    # LIVE MODE: try fetching real data
    try:
        start = now - datetime.timedelta(minutes=5)
        df = yf.download("^NSEI", interval="1m", start=start, end=now)
        return df
    except Exception as e:
        st.error(f"Live data fetch error: {e}")
        return pd.DataFrame()


# Fetch NSE option chain data (only on signal)
def fetch_option_price(strike, call_or_put):
    if st.session_state.demo_mode:
        return 250 if call_or_put == "CE" else 180  # Dummy prices
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9"
    }
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers)
    response = session.get(url, headers=headers)
    data = response.json()
    for item in data['records']['data']:
        if item['strikePrice'] == strike:
            if call_or_put == "CE" and 'CE' in item:
                return item['CE']['lastPrice']
            elif call_or_put == "PE" and 'PE' in item:
                return item['PE']['lastPrice']
    return None

# Option suggestion
def get_option_recommendation(ltp, signal):
    strike = int(round(float(ltp) / 50.0) * 50)
    if signal == 1:
        return f"NIFTY {strike} CE", strike, "CE"
    elif signal == -1:
        return f"NIFTY {strike} PE", strike, "PE"
    return "No Trade", None, None

# Constants
LOT_SIZE = 50
sl_pct = 0.02
tp_pct = 0.01

st.markdown("---")
st.subheader("📈 Live NIFTY Price Monitoring")
df = fetch_data()

if df.empty:
    st.error("Failed to fetch NIFTY data.")
else:
    df = apply_strategy(df)
    latest = df.iloc[-1]
    signal = latest['Signal'].item()
    ltp = float(latest['Close'])

    st.metric("Current NIFTY", value=f"{ltp:.2f}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Signal", "Buy" if signal == 1 else "Sell" if signal == -1 else "Hold")
    with col2:
        st.metric("Capital", f"₹{st.session_state.capital:,.2f}")

    name, strike, cp = get_option_recommendation(ltp, signal)
    st.markdown(f"### 📌 Option Suggestion: `{name}`")

    if (signal in [1, -1]) and not st.session_state.hold:
        try:
            option_price = fetch_option_price(strike, cp)
            if option_price is not None:
                entry_price = option_price
                sl = entry_price * (1 - sl_pct)
                tp = entry_price * (1 + tp_pct)
                lots = int(st.session_state.capital // (entry_price * LOT_SIZE))

                st.session_state.hold = True
                st.session_state.entry_price = entry_price
                st.session_state.position = {
                    "Buy Time": latest.name.tz_convert('Asia/Kolkata'),
                    "Buy Price": entry_price,
                    "SL": sl,
                    "TP": tp,
                    "Option": name,
                    "Lots": lots
                }

                msg = f"🔼 BUY {name} @ ₹{entry_price:.2f}\n🎯 Target: ₹{tp:.2f} | 🛑 SL: ₹{sl:.2f} | 📦 {lots} lot(s)"
                send_telegram_alert(msg)
                st.success(msg)
        except Exception as e:
            st.error(f"Option data error: {e}")

    elif st.session_state.hold:
        try:
            option_price = fetch_option_price(strike, cp)
            if option_price is not None:
                current_price = option_price
                entry_price = st.session_state.entry_price
                pnl = 0
                exit = False

                if current_price <= st.session_state.position['SL']:
                    pnl = (current_price - entry_price) * LOT_SIZE * st.session_state.position['Lots']
                    exit = True
                    reason = "🛑 STOP LOSS"
                elif current_price >= st.session_state.position['TP']:
                    pnl = (current_price - entry_price) * LOT_SIZE * st.session_state.position['Lots']
                    exit = True
                    reason = "✅ TARGET HIT"

                if exit:
                    st.session_state.capital += pnl
                    trade = {
                        **st.session_state.position,
                        "Sell Time": latest.name.tz_convert('Asia/Kolkata'),
                        "Sell Price": current_price,
                        "PnL": pnl,
                        "Capital": st.session_state.capital,
                        "Reason": reason
                    }
                    st.session_state.trades.append(trade)
                    st.session_state.hold = False
                    st.session_state.position = None

                    msg = f"{reason} | Exit ₹{current_price:.2f} | P&L ₹{pnl:.2f}"
                    send_telegram_alert(msg)
                    st.success(msg)
        except Exception as e:
            st.error(f"Option fetch error: {e}")

    if st.session_state.trades:
        st.markdown("---")
        st.subheader("📋 Trade Log")
        st.dataframe(pd.DataFrame(st.session_state.trades)[[
            "Buy Time", "Option", "Buy Price", "Sell Time", "Sell Price", "PnL", "Capital", "Reason"
        ]])