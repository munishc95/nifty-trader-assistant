import pandas as pd
import numpy as np

class SignalEngine:
    """Generate real-time trading signals based on technical indicators."""

    def __init__(self, ema_short=5, ema_long=20, rsi_period=14, rsi_threshold=50):
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold

    def compute_signal(self, df: pd.DataFrame) -> int:
        """Return 1 for buy, -1 for sell, 0 for hold."""
        if df is None or df.empty:
            return 0

        df = df.copy()

        # Calculate EMAs
        df['EMA_SHORT'] = df['Close'].ewm(span=self.ema_short, adjust=False).mean()
        df['EMA_LONG'] = df['Close'].ewm(span=self.ema_long, adjust=False).mean()

        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=self.rsi_period).mean()
        loss = -delta.clip(upper=0).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        latest = df.iloc[-1]
        if latest['EMA_SHORT'] > latest['EMA_LONG'] and latest['RSI'] > self.rsi_threshold:
            return 1
        if latest['EMA_SHORT'] < latest['EMA_LONG'] and latest['RSI'] < self.rsi_threshold:
            return -1
        return 0
