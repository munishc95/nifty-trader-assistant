import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import datetime
import pytz
import requests
import os
import time
from dotenv import load_dotenv
from upstox_manager import UpstoxDataManager

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(layout="wide", page_title="NIFTY Trader Assistant")

# Initialize Upstox data manager in session state
if 'upstox_manager' not in st.session_state:
    try:
        st.session_state.upstox_manager = UpstoxDataManager()
        st.session_state.upstox_manager.start_streaming()
    except Exception as e:
        st.error(f"Failed to initialize Upstox: {str(e)}")
        st.session_state.upstox_manager = None

# Initialize session state variables
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
if "last_update" not in st.session_state:
    st.session_state.last_update = datetime.datetime.now()

# Sidebar controls
with st.sidebar:
    st.title("⚙️ Settings")
    demo_toggle = st.checkbox(
        "🧪 Enable Demo Mode",
        value=st.session_state.demo_mode,
        key="demo_mode_checkbox",
        help="Simulates market data for testing outside market hours"
    )
    st.session_state.demo_mode = demo_toggle
    
    st.subheader("Strategy Parameters")
    ema_short = st.slider("EMA Short Period", min_value=3, max_value=15, value=5)
    ema_long = st.slider("EMA Long Period", min_value=10, max_value=50, value=20)
    rsi_period = st.slider("RSI Period", min_value=7, max_value=21, value=14)
    rsi_threshold = st.slider("RSI Threshold", min_value=30, max_value=70, value=50)
    
    st.subheader("Risk Management")
    tp_multiplier = st.slider("Target Profit (ATR multiplier)", 1.0, 3.0, 1.5, 0.1)
    sl_multiplier = st.slider("Stop Loss (ATR multiplier)", 0.3, 1.0, 0.5, 0.1)
    
    st.subheader("Autotrading")
    auto_trade = st.checkbox("Enable Auto Trading", value=False)

# Time and market hours
ist_tz = pytz.timezone('Asia/Kolkata')
if st.session_state.demo_mode:
    now = ist_tz.localize(datetime.datetime(2025, 4, 21, 10, 0))
else:
    now = datetime.datetime.now(ist_tz)

# Check market hours
market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
is_market_open = (market_open <= now <= market_close)

# If market is closed and not in demo mode, show warning
if not st.session_state.demo_mode and not is_market_open:
    st.warning("⏸ Market is closed. App will resume at 9:15 AM IST.")
    next_open = market_open
    if now > market_close:
        next_open = market_open + datetime.timedelta(days=1)
        # Skip to next weekday if weekend
        while next_open.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            next_open += datetime.timedelta(days=1)
    
    st.info(f"Next market open: {next_open.strftime('%A, %d %B %Y, %I:%M %p IST')}")
    
    # Still allow demo mode to be used
    if not st.session_state.demo_mode:
        st.button("Enable Demo Mode", on_click=lambda: setattr(st.session_state, 'demo_mode', True))
        st_autorefresh(interval=60000, limit=None, key="refresh_check")
        st.stop()

# Main app - enable auto-refresh for real-time updates
st_autorefresh(interval=5000, limit=None, key="data_refresh")

# App title and header
st.title("🔴 NIFTY Live Trading Assistant")

# Add this after your title
col1, col2, col3 = st.columns(3)
with col1:
    if st.session_state.upstox_manager and st.session_state.upstox_manager.running:
        st.success("✅ Upstox Connected")
    else:
        st.error("❌ Upstox Disconnected")
        
with col2:
    st.info(f"Market status: {'🟢 Open' if is_market_open else '🔴 Closed'}")
    
with col3:
    st.info(f"Mode: {'🧪 Demo' if st.session_state.demo_mode else '🔴 Live'}")

if st.session_state.demo_mode:
    st.info("🧪 Demo Mode: Using simulated data for testing")

# Telegram notification setup
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(message):
    """Send a Telegram message with trade alerts"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        st.warning("Telegram notifications not configured. Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID environment variables.")
        return
        
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            st.error(f"Failed to send Telegram alert: {response.text}")
    except Exception as e:
        st.error(f"Telegram alert failed: {e}")

def apply_strategy(df, ema_short=5, ema_long=20, rsi_period=14, rsi_threshold=50):
    """Apply the trading strategy to historical data"""
    # Skip if dataframe is empty or has insufficient data
    if df.empty or len(df) < ema_long + 1:
        return df
    
    # Calculate EMAs
    df['EMA_SHORT'] = df['Close'].ewm(span=ema_short, adjust=False).mean()
    df['EMA_LONG'] = df['Close'].ewm(span=ema_long, adjust=False).mean()

    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # Generate signals
    df['Signal'] = 0
    df.loc[(df['EMA_SHORT'] > df['EMA_LONG']) & (df['RSI'] > rsi_threshold), 'Signal'] = 1  # Buy (Call option)
    df.loc[(df['EMA_SHORT'] < df['EMA_LONG']) & (df['RSI'] < rsi_threshold), 'Signal'] = -1  # Sell (Put option)
    
    return df

def fetch_data():
    """Fetch market data from Upstox or use simulated data in demo mode"""
    if st.session_state.demo_mode:
        # Generate simulated data
        np.random.seed(int(now.strftime("%H%M%S")))
        prices = 22800 + np.cumsum(np.random.normal(0, 5, 30))
        highs = prices + np.random.uniform(5, 20, 30)
        lows = prices - np.random.uniform(5, 20, 30)
        opens = prices - np.random.uniform(-10, 10, 30)
        volumes = np.random.randint(10000, 50000, 30)
        
        # Create DataFrame with simulated data
        df = pd.DataFrame({
            'Datetime': pd.date_range(end=now, periods=30, freq='1min'),
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        })
        df.set_index('Datetime', inplace=True)
        return df
    else:
        # Use Upstox data manager
        if st.session_state.upstox_manager:
            try:
                # Fetch historical candles
                df = st.session_state.upstox_manager.get_historical_data(
                    instrument="Nifty 50",
                    interval="1minute",
                    days=1
                )
                
                # If we got data, return the last 30 candles
                if not df.empty:
                    return df.tail(30)
                    
            except Exception as e:
                st.error(f"Failed to fetch Upstox data: {e}")
        
        # Return empty DataFrame if we couldn't get data
        return pd.DataFrame()

def get_option_recommendation(ltp, signal):
    """Generate option recommendation based on price and signal"""
    # Round to the nearest 50 for strike price
    strike = int(round(ltp / 50.0) * 50)
    
    if signal == 1:
        return f"NIFTY {strike} CE", strike, "CE"
    elif signal == -1:
        return f"NIFTY {strike} PE", strike, "PE"
    return "No Trade", None, None

def fetch_option_price(strike, call_or_put):
    """Get option prices from Upstox or generate demo prices"""
    if st.session_state.demo_mode:
        # Generate random demo price
        base_price = 200
        if call_or_put == "CE":
            return base_price + np.random.randint(-10, 10)
        else:
            return base_price + np.random.randint(-8, 12)
    
    # TODO: Implement real option price fetching from Upstox
    # For now, return simulated prices since this requires additional setup
    return 200 + np.random.randint(-5, 5)

# Main processing logic
try:
    # Fetch and process data
    df = fetch_data()
    if df.empty:
        st.error("❌ No market data available. Check connection or wait for market to open.")
        st.stop()
    
    # Apply strategy with user-defined parameters
    df = apply_strategy(
        df, 
        ema_short=ema_short, 
        ema_long=ema_long, 
        rsi_period=rsi_period,
        rsi_threshold=rsi_threshold
    )
    
    # Get latest data points
    latest = df.iloc[-1]
    signal = int(latest['Signal']) if 'Signal' in latest else 0
    ltp = float(latest['Close'])
    
    # Handle ATR for risk management
    atr = latest.get('ATR', 10.0)
    if isinstance(atr, pd.Series):
        atr = atr.iloc[0]
    if pd.isna(atr) or not np.isfinite(atr):
        atr = 10.0
    
    # Calculate target profit and stop loss points
    tp_points = max(atr * tp_multiplier, 10)
    sl_points = max(atr * sl_multiplier, 3)
    
    # Display market information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NIFTY Current Price", f"₹{ltp:.2f}")
    with col2:
        signal_text = "🟢 BUY" if signal == 1 else "🔴 SELL" if signal == -1 else "⚪ HOLD"
        st.metric("Signal", signal_text)
    with col3:
        st.metric("Available Capital", f"₹{st.session_state.capital:,.2f}")
    
    # Get option recommendation
    name, strike, cp = get_option_recommendation(ltp, signal)
    
    # Display option recommendation
    st.markdown(f"### 📌 Option Recommendation:")
    st.markdown(f"### `{name}`")
    
    # Show current position details if any
    if st.session_state.get("last_trade_price_details"):
        entry = st.session_state["last_trade_price_details"]["entry"]
        target = st.session_state["last_trade_price_details"]["target"]
        sl = st.session_state["last_trade_price_details"]["sl"]
        
        st.markdown("### Current Position")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entry Price", f"₹{entry:.2f}")
        with col2:
            st.metric("Target", f"₹{target:.2f}")
        with col3:
            st.metric("Stop Loss", f"₹{sl:.2f}")
    
    # Show last trade message if any
    if st.session_state.get("last_trade_message"):
        st.success(st.session_state["last_trade_message"])
    
    # Define lot size for NIFTY options
    LOT_SIZE = 50
    
    # Trading logic
    if signal in [1, -1] and not st.session_state.hold:
        # Get option price
        price = fetch_option_price(strike, cp)
        
        if price:
            entry_price = price
            sl = entry_price - sl_points if signal == 1 else entry_price + sl_points
            tp = entry_price + tp_points if signal == 1 else entry_price - tp_points
            
            # Calculate position size
            lots = int(st.session_state.capital // (entry_price * LOT_SIZE))
            
            # Create position entry card
            st.markdown("### 🎯 New Trade Signal")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Option:** {name}")
                st.markdown(f"**Signal:** {'BUY' if signal == 1 else 'SELL'}")
                st.markdown(f"**Entry:** ₹{entry_price:.2f}")
            
            with col2:
                st.markdown(f"**Target:** ₹{tp:.2f}")
                st.markdown(f"**Stop Loss:** ₹{sl:.2f}")
                st.markdown(f"**Lots:** {lots}")
            
            # Trade execution button or automatic trading
            if auto_trade or st.button("Execute Trade"):
                st.session_state.hold = True
                st.session_state.entry_price = entry_price
                st.session_state.position = {
                    "Buy Time": latest.name,
                    "Buy Price": entry_price,
                    "SL": sl,
                    "TP": tp,
                    "Option": name,
                    "Lots": lots,
                    "Signal": "BUY" if signal == 1 else "SELL"
                }
                
                # Prepare and send alert message
                msg = f"🔔 *TRADE ALERT*\n\n{'🟢 BUY' if signal == 1 else '🔴 SELL'} {name} @ ₹{entry_price:.2f}\n🎯 Target: ₹{tp:.2f}\n🛑 Stop Loss: ₹{sl:.2f}\n📊 {lots} lot(s)"
                send_telegram_alert(msg)
                
                # Update session state
                st.session_state.last_trade_message = f"Trade executed at {datetime.datetime.now(ist_tz).strftime('%H:%M:%S')}"
                st.session_state.last_trade_price_details = {
                    "entry": entry_price,
                    "target": tp,
                    "sl": sl
                }
                
                # Rerun to update UI immediately
                st.experimental_rerun()
    
    # Position management if we have an open position
    elif st.session_state.hold:
        price = fetch_option_price(strike, cp)
        
        if price:
            current_price = price
            entry = st.session_state.entry_price
            signal_type = st.session_state.position['Signal']
            
            # Calculate P&L based on position direction
            if signal_type == "BUY":
                pnl = (current_price - entry) * LOT_SIZE * st.session_state.position['Lots']
                exit_reason = None
                if current_price <= st.session_state.position['SL']:
                    exit_reason = "🛑 STOP LOSS"
                elif current_price >= st.session_state.position['TP']:
                    exit_reason = "✅ TARGET HIT"
            else:  # SELL position
                pnl = (entry - current_price) * LOT_SIZE * st.session_state.position['Lots']
                exit_reason = None
                if current_price >= st.session_state.position['SL']:
                    exit_reason = "🛑 STOP LOSS"
                elif current_price <= st.session_state.position['TP']:
                    exit_reason = "✅ TARGET HIT"
            
            # Position monitoring
            st.markdown("### 📊 Position Monitor")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"₹{current_price:.2f}", 
                        delta=f"{current_price - entry:.2f}")
            with col2:
                pnl_formatted = f"₹{pnl:.2f}"
                st.metric("Current P&L", pnl_formatted)
            with col3:
                st.metric("Target Distance", 
                        f"{abs(current_price - st.session_state.position['TP']):.2f}")
            
            # Manual exit button
            if st.button("Close Position"):
                exit_reason = "👤 MANUAL EXIT"
            
            # Handle position exit
            if exit_reason:
                st.session_state.capital += pnl
                
                # Record the trade
                trade = {
                    **st.session_state.position,
                    "Sell Time": datetime.datetime.now(ist_tz),
                    "Sell Price": current_price,
                    "PnL": pnl,
                    "Capital": st.session_state.capital,
                    "Reason": exit_reason
                }
                st.session_state.trades.append(trade)
                
                # Reset position flags
                st.session_state.hold = False
                st.session_state.position = None
                
                # Send alert
                msg = f"🔔 *POSITION CLOSED*\n\n{exit_reason}\nExit Price: ₹{current_price:.2f}\nP&L: ₹{pnl:.2f}\nRemaining Capital: ₹{st.session_state.capital:,.2f}"
                send_telegram_alert(msg)
                
                # Update UI
                st.success(f"{exit_reason} | Exit ₹{current_price:.2f} | P&L ₹{pnl:.2f}")
                st.session_state.last_trade_message = ""
                st.session_state.last_trade_price_details = None
                
                # Rerun to update UI
                st.experimental_rerun()
    
    # Display trade history
    if st.session_state.trades:
        st.subheader("📋 Trade Log")
        trades_df = pd.DataFrame(st.session_state.trades)
        
        # Format the dataframe for display
        if not trades_df.empty:
            display_cols = [
                "Buy Time", "Option", "Signal", "Buy Price", 
                "Sell Time", "Sell Price", "PnL", "Reason"
            ]
            
            # Ensure all columns exist
            for col in display_cols:
                if col not in trades_df.columns:
                    trades_df[col] = ""
                    
            st.dataframe(trades_df[display_cols], use_container_width=True)
            
            # Calculate and show summary statistics
            total_pnl = trades_df["PnL"].sum()
            win_trades = trades_df[trades_df["PnL"] > 0]
            loss_trades = trades_df[trades_df["PnL"] < 0]
            win_rate = len(win_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
            
            st.subheader("📈 Performance Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total P&L", f"₹{total_pnl:.2f}")
            with col2:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                st.metric("# Trades", len(trades_df))
            with col4:
                st.metric("Avg P&L per Trade", 
                        f"₹{(total_pnl / len(trades_df)):.2f}" if len(trades_df) > 0 else "₹0.00")

    # Add technical analysis charts
    if not df.empty and len(df) > 5:
        st.subheader("📊 Technical Analysis")
        
        # Import matplotlib for charts
        import matplotlib.pyplot as plt
        
        # Create figure and axis for price chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price and indicators on first axis
        ax1.plot(df.index, df['Close'], label='NIFTY', color='black', linewidth=1.5)
        if 'EMA_SHORT' in df.columns:
            ax1.plot(df.index, df['EMA_SHORT'], label=f'EMA {ema_short}', color='blue', linewidth=1)
        if 'EMA_LONG' in df.columns:
            ax1.plot(df.index, df['EMA_LONG'], label=f'EMA {ema_long}', color='red', linewidth=1)
            
        # Add buy/sell signals
        if 'Signal' in df.columns:
            buys = df[df['Signal'] == 1]
            sells = df[df['Signal'] == -1]
            ax1.scatter(buys.index, buys['Close'], color='green', s=100, marker='^', label='Buy Signal')
            ax1.scatter(sells.index, sells['Close'], color='red', s=100, marker='v', label='Sell Signal')
        
        ax1.set_title(f'NIFTY Price with Indicators ({now.strftime("%Y-%m-%d")})')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Plot RSI on second axis
        if 'RSI' in df.columns:
            ax2.plot(df.index, df['RSI'], color='purple', linewidth=1)
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax2.axhline(y=50, color='k', linestyle='--', alpha=0.3)
            ax2.fill_between(df.index, df['RSI'], 70, where=(df['RSI'] >= 70), color='r', alpha=0.3)
            ax2.fill_between(df.index, df['RSI'], 30, where=(df['RSI'] <= 30), color='g', alpha=0.3)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('RSI')
        
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout and display chart
        plt.tight_layout()
        st.pyplot(fig)
    
    # Update last refresh time
    st.session_state.last_update = datetime.datetime.now()
    st.caption(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
