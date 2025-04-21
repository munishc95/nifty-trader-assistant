import os
import threading
import time
import queue
import pandas as pd
import numpy as np
import upstox_client
from datetime import datetime, timedelta
import pytz
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UpstoxDataManager:
    def __init__(self):
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        if not self.access_token:
            raise ValueError("UPSTOX_ACCESS_TOKEN environment variable not set")
        
        # Validate token format
        if len(self.access_token) < 20:  # Tokens are typically long
            logger.warning(f"Access token seems unusually short ({len(self.access_token)} chars)")
        
        # Configure client
        try:
            self.configuration = upstox_client.Configuration()
            self.configuration.access_token = self.access_token
            self.api_client = upstox_client.ApiClient(self.configuration)
            
            # Test API connection here if needed
            
        except Exception as e:
            logger.error(f"Failed to initialize Upstox client: {e}")
            raise
        
        # Data storage
        self.current_data = {}
        self.hist_data = {}
        self.data_queue = queue.Queue()
        self.running = False
        self.stream_thread = None
        self.streamer = None
        
        # For storing candles temporarily (will build dataframes from this)
        self._candle_data = {
            "Nifty 50": [],
            "Nifty Bank": []
        }
        
        logger.info("UpstoxDataManager initialized")
    
    def start_streaming(self):
        """Start the data streaming in a background thread"""
        if self.running:
            return
        
        self.running = True
        self.stream_thread = threading.Thread(target=self._stream_data)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        logger.info("Market data streaming started")
    
    def stop_streaming(self):
        """Stop the streaming thread"""
        if not self.running:
            return
            
        self.running = False
        if self.streamer:
            try:
                self.streamer.disconnect()
            except:
                pass
                
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        logger.info("Market data streaming stopped")
    
    def _stream_data(self):
        """Background thread to stream market data"""
        try:
            # Create instrument keys
            instrument_keys = ["NSE_INDEX|Nifty 50", "NSE_INDEX|Nifty Bank"]
            
            # Initialize streamer
            self.streamer = upstox_client.MarketDataStreamerV3(
                self.api_client,
                instrument_keys,
                "full"  # Get full market data
            )
            
            # Set up callbacks
            self.streamer.on("message", self._on_message)
            self.streamer.on("error", self._on_error)
            self.streamer.on("close", self._on_close)
            
            # Connect to websocket
            self.streamer.connect()
            logger.info("Connected to Upstox WebSocket")
            
            # Keep thread alive while running
            while self.running:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            self.running = False
    
    def _on_message(self, message):
        """Handle incoming market data"""
        try:
            # Process and store the data
            instrument_key = message.get("instrument_key")
            if not instrument_key:
                return
            
            # Extract instrument name (Nifty 50 or Nifty Bank)
            parts = instrument_key.split("|")
            instrument_name = parts[1] if len(parts) > 1 else instrument_key
                
            # Update current price data
            ltp = message.get("ltp")
            if ltp:
                # Store the current tick data
                self.current_data[instrument_name] = {
                    "ltp": ltp,
                    "volume": message.get("volume", 0),
                    "timestamp": datetime.now(pytz.timezone('Asia/Kolkata')),
                    "open": message.get("open", ltp),
                    "high": message.get("high", ltp),
                    "low": message.get("low", ltp),
                    "close": message.get("close", ltp),
                }
                
                # If it's a complete candle, add to our candle data
                if message.get("candle_data"):
                    self._add_candle(instrument_name, message)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _add_candle(self, instrument_name, tick_data):
        """Add a new candle to our temporary storage"""
        try:
            candle = tick_data.get("candle_data")
            if not candle:
                return
                
            # Format timestamp
            timestamp = datetime.fromtimestamp(
                candle.get("timestamp", time.time()), 
                tz=pytz.timezone('Asia/Kolkata')
            )
                
            # Create candle data structure
            candle_entry = {
                'Datetime': timestamp,
                'Open': candle.get('open', tick_data.get('ltp')),
                'High': candle.get('high', tick_data.get('ltp')),
                'Low': candle.get('low', tick_data.get('ltp')),
                'Close': candle.get('close', tick_data.get('ltp')),
                'Volume': candle.get('volume', tick_data.get('volume', 0))
            }
            
            # Add to list for this instrument
            if instrument_name in self._candle_data:
                self._candle_data[instrument_name].append(candle_entry)
                
                # Keep only the last 500 candles
                if len(self._candle_data[instrument_name]) > 500:
                    self._candle_data[instrument_name] = self._candle_data[instrument_name][-500:]
        except Exception as e:
            logger.error(f"Error adding candle: {e}")
    
    def _on_error(self, error):
        """Handle streaming errors"""
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, status_code=None, reason=None):
        """Handle stream closure"""
        logger.warning(f"WebSocket closed. Status: {status_code}, Reason: {reason}")
        self.running = False
    
    def get_latest_price(self, instrument="Nifty 50"):
        """Get the latest price for an instrument"""
        if instrument in self.current_data:
            return self.current_data[instrument]
        return None
    
    def get_historical_data(self, instrument="Nifty 50", interval="1minute", days=1):
        """Fetch historical OHLC data"""
        # First check if we have data in our candle storage
        if instrument in self._candle_data and self._candle_data[instrument]:
            try:
                # Create a DataFrame from our stored candles
                df = pd.DataFrame(self._candle_data[instrument])
                
                if not df.empty:
                    df.set_index('Datetime', inplace=True)
                    # Sort by datetime index
                    df.sort_index(inplace=True)
                    self.hist_data[instrument] = df
                    return df
            except Exception as e:
                logger.error(f"Error creating DataFrame: {e}")
        
        # If we have no data, generate some simulated data for demo purposes
        # This will be useful until we implement the historical API properly
        try:
            # Generate some simulated data
            ist = pytz.timezone('Asia/Kolkata')
            end_date = datetime.now(ist)
            start_date = end_date - timedelta(days=days)
            
            # Create date range for the timeframe
            date_range = pd.date_range(start=start_date, end=end_date, freq='1min')
            
            # Generate random prices with a trend
            base_price = 22000 if instrument == "Nifty 50" else 45000  # Different base for Nifty vs Bank Nifty
            np.random.seed(42)  # For reproducibility
            
            # Create a slight trend
            trend = np.cumsum(np.random.normal(0, 10, len(date_range)))
            
            # Generate price data
            prices = base_price + trend
            highs = prices + np.random.uniform(5, 50, len(date_range))
            lows = prices - np.random.uniform(5, 50, len(date_range))
            opens = prices - np.random.uniform(-30, 30, len(date_range))
            volumes = np.random.randint(1000, 50000, len(date_range))
            
            # Create DataFrame
            df = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': prices,
                'Volume': volumes
            }, index=date_range)
            
            # Store and return the data
            self.hist_data[instrument] = df
            return df
            
        except Exception as e:
            logger.error(f"Error generating simulated data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error