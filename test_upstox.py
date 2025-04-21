from dotenv import load_dotenv
import os
import time
import pandas as pd
from upstox_manager import UpstoxDataManager

# Load environment variables
load_dotenv()

def main():
    print("Testing Upstox connection...")
    
    try:
        # Create manager
        manager = UpstoxDataManager()
        
        # Start streaming
        print("Starting data stream...")
        manager.start_streaming()
        
        # Wait for some data
        print("Waiting for data (10 seconds)...")
        time.sleep(10)
        
        # Check latest prices
        nifty_data = manager.get_latest_price("Nifty 50")
        if nifty_data:
            print(f"Latest Nifty 50 price: {nifty_data['ltp']}")
        else:
            print("No Nifty data received yet")
            
        # Get historical data
        print("Retrieving historical data...")
        hist_df = manager.get_historical_data("Nifty 50")
        if not hist_df.empty:
            print(f"Retrieved {len(hist_df)} candles")
            print("\nLast 5 candles:")
            print(hist_df.tail(5))
        else:
            print("No historical data available")
            
        # Clean shutdown
        print("Shutting down...")
        manager.stop_streaming()
        print("Test complete")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    main()