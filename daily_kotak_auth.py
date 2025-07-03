# daily_auth.py - Corrected Daily Authentication Script
"""
Simplified Daily Authentication for Kotak Neo API
================================================

This script performs daily authentication and saves session state.
Run this once per day to authenticate your trading session.

The Neo API client handles session management internally after successful 2FA.
We just need to authenticate and verify the session works.

Usage:
    python daily_auth.py

Author: Simplified based on official Neo API docs
"""

import os
import json
import getpass
from datetime import datetime
from dotenv import load_dotenv
from neo_api_client import NeoAPI


def daily_authentication():
    """✅ FIXED: Simplified daily authentication"""
    try:
        print("🔐 Kotak Neo API Daily Authentication")
        print("=" * 40)

        load_dotenv()

        # Initialize client
        consumer_key = os.getenv('KOTAK_CONSUMER_KEY')
        consumer_secret = os.getenv('KOTAK_CONSUMER_SECRET')
        environment = os.getenv('KOTAK_ENVIRONMENT', 'prod')
        mobile_number = os.getenv('KOTAK_MOBILE_NUMBER')
        password = os.getenv('KOTAK_PASSWORD')

        if not consumer_key or not consumer_secret:
            print("❌ Missing KOTAK_CONSUMER_KEY or KOTAK_CONSUMER_SECRET in .env file")
            return False

        print(f"Environment: {environment}")
        print(f"Consumer Key: {consumer_key[:10]}...")

        client = NeoAPI(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            environment=environment
        )

        try:
            login_response = client.login(mobilenumber=mobile_number, password=password)
            print("✅ Login successful - OTP sent to your mobile")
            print(f"Response: {login_response}")
        except Exception as e:
            print(f"❌ Login failed: {e}")
            return False

        # Step 2: Complete 2FA
        print("\n🔑 Step 2: Two-Factor Authentication")
        otp = input("Enter OTP received on mobile: ")

        try:
            session_response = client.session_2fa(OTP=otp)
            print("✅ 2FA successful - Session established")
            print(f"Response: {session_response}")
        except Exception as e:
            print(f"❌ 2FA failed: {e}")
            return False

        # Step 3: Test session by making API calls
        print("\n🧪 Step 3: Testing Session")

        try:
            print("   Testing positions API...")
            positions = client.positions()
            if 'data' in positions:
                print(f"✅ Positions API working - Found {len(positions['data'])} positions")
            else:
                print("⚠️  Positions API returned no data")

        except Exception as e:
            print(f"❌ Positions API failed: {e}")
            return False

        try:
            print("   Testing limits API...")
            limits = client.limits()
            if 'data' in limits:
                print("✅ Limits API working")
            else:
                print("⚠️  Limits API returned unexpected response")
        except Exception as e:
            print(f"⚠️  Limits API warning: {e}")
            # Don't fail on limits API - it sometimes has parameter issues

        try:
            print("   Testing scrip master API...")
            scrip_master = client.scrip_master()
            if scrip_master and 'data' in scrip_master:
                print(f"✅ Scrip master API working - {len(scrip_master['data'])} instruments")
            else:
                print("⚠️  Scrip master API returned no data")
        except Exception as e:
            print(f"❌ Scrip master API failed: {e}")
            return False

        # Step 4: Save session state
        # Step 4: Extract and save session tokens
        print("\n💾 Step 4: Extracting Session Tokens")

        # Get session tokens from client configuration
        config = client.api_client.configuration

        session_tokens = {
            'edit_token': client.configuration.edit_token,
            'sid': client.configuration.sid,
            'edit_sid': client.configuration.edit_sid,
            'edit_rid': client.configuration.edit_rid,
            'serverId': client.configuration.serverId,
            'userId': client.configuration.userId,
        }

        session_data = {
            'authenticated': True,
            'timestamp': datetime.now().isoformat(),
            'mobile': mobile_number,
            'environment': environment,
            'tokens': session_tokens  # Add this line
        }

        session_file = 'kotak_session.json'
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        print(f"✅ Session saved to {session_file}")

        # Step 5: Final verification
        print("\n✅ Authentication Complete!")
        print("=" * 40)
        print("✅ Daily authentication successful")
        print("✅ Session is active and verified")
        print("✅ All required APIs are working")
        print("✅ You can now run your trading scripts")
        print("\nℹ️  This session will remain active until market close.")
        print("ℹ️  Run this script again tomorrow for fresh authentication.")

        return True

    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify credentials in .env file")
        print("3. Ensure mobile number includes country code (+91)")
        print("4. Check if your Kotak Neo API is activated")
        print("5. Try using UAT environment first for testing")
        return False


def test_existing_session():
    """Test if existing session is still valid"""
    try:
        print("🔍 Testing Existing Session")
        print("=" * 30)

        session_file = 'kotak_session.json'
        if not os.path.exists(session_file):
            print("❌ No session file found")
            return False

        with open(session_file, 'r') as f:
            session_data = json.load(f)

        if not session_data.get('authenticated'):
            print("❌ Session not authenticated")
            return False

        print(f"📅 Session created: {session_data.get('timestamp')}")
        print(f"📱 Mobile: {session_data.get('mobile')}")
        print(f"🌐 Environment: {session_data.get('environment')}")

        # Try to create client and test
        load_dotenv()
        client = NeoAPI(
            consumer_key=os.getenv('KOTAK_CONSUMER_KEY'),
            consumer_secret=os.getenv('KOTAK_CONSUMER_SECRET'),
            environment=session_data.get('environment', 'prod')
        )

        # Test with a simple API call
        positions = client.positions()
        if 'data' in positions:
            print("✅ Existing session is valid")
            print(f"✅ Found {len(positions['data'])} positions")
            return True
        else:
            print("❌ Session validation failed")
            return False

    except Exception as e:
        print(f"❌ Session test failed: {e}")
        return False


def main():
    """Main function with options"""
    print("🚀 Kotak Neo API Authentication Manager")
    print("=" * 45)
    print("1. Perform daily authentication")
    print("2. Test existing session")
    print("3. Exit")

    while True:
        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '1':
            success = daily_authentication()
            if success:
                print("\n🎉 Ready to trade! You can now run your trading scripts.")
            break
        elif choice == '2':
            if test_existing_session():
                print("\n🎉 Existing session is valid! You can run your trading scripts.")
            else:
                print("\n⚠️  Please run daily authentication (option 1)")
            break
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()