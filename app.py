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

# Constants
LOT_SIZE = 50
TP_POINTS = 10    # 🎯 Fixed 10-point target
SL_POINTS = 3     # 🛑 Fixed 3-point stop loss

# Session state defaults
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False

# Demo toggle
demo_toggle = st.sidebar.checkbox(
    "🧪 Enable Demo Mode", 
    value=st.session_state.demo_mode, 
    key="demo_mode_checkbox", 
    help="Simulates market hours and option prices for testing on weekends."
)
st.session_state.demo_mode = demo_toggle

# Time config
if st.session_state.demo_mode:
    now = pytz.timezone('Asia/Kolkata').localize(datetime.datetime(2025, 4, 10, 10, 0))
else:
    now = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))

# Market hours check
market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

if not st.session_state.demo_mode and not (market_open <= now <= market_close):
    st.warning("⏸ Market is closed. App will resume at 9:15 AM IST.")
    start_of_day = now.replace(hour=9, minute=15)
    end_of_day = now.replace(hour=15, minute=30)
    df = yf.download("^NSEI", interval="1m", start=start_of_day, end=end_of_day)
    if not df.empty:
        closing_price = df['Close'].iloc[-1]
        st.metric("🔒 Closing Price", f"{closing_price:.2f}")
    else:
        st.info("No intraday data available to show closing price.")

    st.markdown("---")
    st.subheader("📋 Daily Trade Summary")
    if 'trades' in st.session_state and st.session_state.trades:
        trades_df = pd.DataFrame(st.session_state.trades)
        trades_df['Date'] = pd.to_datetime(trades_df['Buy Time']).dt.date
        today_trades = trades_df[trades_df['Date'] == now.date()]
        if not today_trades.empty:
            st.metric("🔄 Daily P&L", f"₹{today_trades['PnL'].sum():,.2f}")
        else:
            st.info("💬 0 trades were taken today.")
    else:
        st.info("💬 0 trades were taken today.")
    st.stop()

st_autorefresh(interval=1000, limit=None, key="data_refresh")

st.title("🔴 NIFTY Live Paper Trading - Intraday Strategy")

# Init session state
if 'capital' not in st.session_state:
    st.session_state.capital = 100000
    st.session_state.hold = False
    st.session_state.entry_price = 0.0
    st.session_state.position = None
    st.session_state.trades = []

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

def fetch_data():
    now = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    if st.session_state.demo_mode:
        np.random.seed(int(now.strftime("%H%M%S")))
        base_price = 22800
        noise = np.random.normal(loc=0, scale=10, size=5).cumsum()
        prices = base_price + noise
        timestamps = pd.date_range(end=now, periods=5, freq='1min')
        df = pd.DataFrame({'Datetime': timestamps, 'Close': prices}).set_index('Datetime')
        return df
    try:
        start = now - datetime.timedelta(minutes=20)
        df = yf.download("^NSEI", interval="1m", start=start, end=now)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        st.error(f"Live data fetch error: {e}")
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
        return 250 if call_or_put == "CE" else 180
    data = fetch_nse_option_chain()
    if data:
        for item in data['records']['data']:
            if item['strikePrice'] == strike:
                if call_or_put == "CE" and 'CE' in item:
                    return item['CE']['lastPrice']
                elif call_or_put == "PE" and 'PE' in item:
                    return item['PE']['lastPrice']
    return None

def get_option_recommendation(ltp, signal):
    strike = int(round(float(ltp) / 50.0) * 50)
    if signal == 1:
        return f"NIFTY {strike} CE", strike, "CE"
    elif signal == -1:
        return f"NIFTY {strike} PE", strike, "PE"
    return "No Trade", None, None

# Start logic
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
                tp = option_price + TP_POINTS
                sl = option_price - SL_POINTS
                lots = int(st.session_state.capital // (option_price * LOT_SIZE))

                st.session_state.hold = True
                st.session_state.entry_price = option_price
                st.session_state.position = {
                    "Buy Time": latest.name.tz_convert('Asia/Kolkata'),
                    "Buy Price": option_price,
                    "SL": sl,
                    "TP": tp,
                    "Option": name,
                    "Lots": lots
                }

                msg = f"🔼 BUY {name} @ ₹{option_price:.2f}\n🎯 Target: ₹{tp:.2f} | 🛑 SL: ₹{sl:.2f} | 📦 {lots} lot(s)"
                send_telegram_alert(msg)
                st.success(msg)
        except Exception as e:
            st.error(f"Option data error: {e}")

    elif st.session_state.hold:
        try:
            option_price = fetch_option_price(strike, cp)

            if option_price is None:
                st.warning("⚠️ Option price not available. Skipping exit check.")
            else:
                current_price = option_price
                entry_price = st.session_state.entry_price
                pnl = 0
                exit = False

                # ✅ Add sanity check to ignore unrealistic drops
                max_deviation = 20  # prevent SL trigger on fake price spikes
                if abs(current_price - entry_price) > max_deviation:
                    st.warning(f"⚠️ Ignoring suspicious price deviation: ₹{current_price:.2f} vs Entry ₹{entry_price:.2f}")
                else:
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
