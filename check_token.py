from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()

def check_token():
    """Check if the Upstox access token is valid"""
    print("Checking Upstox access token...")
    
    token = os.getenv("UPSTOX_ACCESS_TOKEN")
    if not token:
        print("ERROR: UPSTOX_ACCESS_TOKEN environment variable not set!")
        return False
        
    # Print token length and first/last few characters for diagnostics
    print(f"Token length: {len(token)}")
    print(f"Token prefix: {token[:10]}...")
    print(f"Token suffix: ...{token[-10:]}")
    
    # Make a test API call to verify the token
    try:
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        
        # Try to access user profile API
        response = requests.get(
            'https://api.upstox.com/v2/user/profile',
            headers=headers
        )
        
        print(f"API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Token is valid!")
            user_data = response.json().get('data', {})
            print(f"Logged in as: {user_data.get('user_name', 'Unknown')}")
            return True
            
        else:
            print("❌ Token validation failed!")
            print(f"Error details: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error during token check: {str(e)}")
        return False

if __name__ == "__main__":
    check_token()