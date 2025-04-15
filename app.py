
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

# Initialize session state
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False

if "hold" not in st.session_state:
    st.session_state.hold = False

if "entry_price" not in st.session_state:
    st.session_state.entry_price = 0.0

if "capital" not in st.session_state:
    st.session_state.capital = 100000

if "position" not in st.session_state:
    st.session_state.position = None

if "trades" not in st.session_state:
    st.session_state.trades = []

# Sidebar checkbox for demo mode
demo_toggle = st.sidebar.checkbox(
    "🧪 Enable Demo Mode",
    value=st.session_state.demo_mode,
    key="demo_mode_checkbox",
    help="Simulates market hours and option prices for testing on weekends."
)
st.session_state.demo_mode = demo_toggle

# Override time for demo mode
if st.session_state.demo_mode:
    now = pytz.timezone('Asia/Kolkata').localize(datetime.datetime(2025, 4, 10, 10, 0))
else:
    now = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))

# Check market hours
market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

if not st.session_state.demo_mode and not (market_open <= now <= market_close):
    st.warning("⏸ Market is closed. App will resume at 9:15 AM IST.")
    st.stop()

st_autorefresh(interval=5000, limit=None, key="data_refresh")
st.title("🔴 NIFTY Live Paper Trading - Intraday Strategy")

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, data=payload)
    except Exception as e:
        st.error(f"Telegram alert failed: {e}")

def apply_strategy(df):
    df['EMA_5'] = df['Close'].ewm(span=5).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['ATR'] = df['Close'].rolling(window=14).std()
    df['Signal'] = 0
    df.loc[(df['EMA_5'] > df['EMA_20']) & (df['RSI'] > 50), 'Signal'] = 1
    df.loc[(df['EMA_5'] < df['EMA_20']) & (df['RSI'] < 50), 'Signal'] = -1
    return df

def fetch_data():
    if st.session_state.demo_mode:
        st.info("🧪 Demo Mode: Simulated data shown.")
        np.random.seed(int(now.strftime("%H%M%S")))
        prices = 22800 + np.cumsum(np.random.normal(0, 5, 20))
        df = pd.DataFrame({'Datetime': pd.date_range(end=now, periods=20, freq='1min'), 'Close': prices})
        df.set_index('Datetime', inplace=True)
        return df
    else:
        try:
            start = now - datetime.timedelta(minutes=30)
            return yf.download("^NSEI", interval="1m", start=start, end=now)
        except:
            return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_nse_option_chain():
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9"
    }
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        response = session.get(url, headers=headers, timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def fetch_option_price(strike, call_or_put):
    if st.session_state.demo_mode:
        return 200 + np.random.randint(-5, 5)
    data = fetch_nse_option_chain()
    if not data: return None
    for item in data['records']['data']:
        if item['strikePrice'] == strike:
            if call_or_put == "CE" and 'CE' in item:
                return item['CE']['lastPrice']
            elif call_or_put == "PE" and 'PE' in item:
                return item['PE']['lastPrice']
    return None

def get_option_recommendation(ltp, signal):
    strike = int(round(ltp / 50.0) * 50)
    if signal == 1:
        return f"NIFTY {strike} CE", strike, "CE"
    elif signal == -1:
        return f"NIFTY {strike} PE", strike, "PE"
    return "No Trade", None, None

# Start processing
df = fetch_data()
if df.empty:
    st.error("No NIFTY data.")
    st.stop()

df = apply_strategy(df)
latest = df.iloc[-1]
signal = int(latest['Signal'])
ltp = float(latest['Close'])
atr = float(latest['ATR']) if not np.isnan(latest['ATR']) else 10.0
tp_points = max(atr * 1.5, 10)
sl_points = max(atr * 0.5, 3)

st.metric("Current NIFTY", f"{ltp:.2f}")
st.metric("Signal", "Buy" if signal == 1 else "Sell" if signal == -1 else "Hold")
st.metric("Capital", f"₹{st.session_state.capital:,.2f}")

name, strike, cp = get_option_recommendation(ltp, signal)
st.markdown(f"### 📌 Option Suggestion: `{name}`")

LOT_SIZE = 50

if signal in [1, -1] and not st.session_state.hold:
    price = fetch_option_price(strike, cp)
    if price:
        entry_price = price
        sl = entry_price - sl_points
        tp = entry_price + tp_points
        lots = int(st.session_state.capital // (entry_price * LOT_SIZE))
        st.session_state.hold = True
        st.session_state.entry_price = entry_price
        st.session_state.position = {
            "Buy Time": latest.name,
            "Buy Price": entry_price,
            "SL": sl,
            "TP": tp,
            "Option": name,
            "Lots": lots
        }
        msg = f"🔼 BUY {name} @ ₹{entry_price:.2f}\n🎯 Target: ₹{tp:.2f} | 🛑 SL: ₹{sl:.2f} | 📦 {lots} lot(s)"
        send_telegram_alert(msg)
        st.success(msg)

elif st.session_state.hold:
    price = fetch_option_price(strike, cp)
    if price:
        current_price = price
        entry = st.session_state.entry_price
        pnl = (current_price - entry) * LOT_SIZE * st.session_state.position['Lots']
        exit = False
        if current_price <= st.session_state.position['SL']:
            reason = "🛑 STOP LOSS"
            exit = True
        elif current_price >= st.session_state.position['TP']:
            reason = "✅ TARGET HIT"
            exit = True
        if exit:
            st.session_state.capital += pnl
            trade = {
                **st.session_state.position,
                "Sell Time": latest.name,
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

if st.session_state.trades:
    st.subheader("📋 Trade Log")
    st.dataframe(pd.DataFrame(st.session_state.trades)[[
        "Buy Time", "Option", "Buy Price", "Sell Time", "Sell Price", "PnL", "Capital", "Reason"
    ]])
