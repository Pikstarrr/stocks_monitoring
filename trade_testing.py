# test_trade_execution.py
"""
Dummy script to test BUY and SELL operations
Tests the complete trade cycle: BUY NIFTY CE -> Hold 1 min -> SELL NIFTY CE
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Import your main trading class (make sure trade_placement.py is in same directory)
from trade_placement import KotakOptionsTrader, is_market_open


def test_complete_trade_cycle(test_mode=True):
    """
    Test complete trade cycle: BUY -> Hold -> SELL

    Args:
        test_mode: If True, runs in simulation mode. If False, places real orders
    """
    print("\n" + "=" * 80)
    print("🧪 TESTING COMPLETE TRADE CYCLE")
    print("=" * 80)

    # Load environment variables
    load_dotenv()
    access_token = os.getenv('KOTAK_ACCESS_TOKEN')
    if not access_token:
        print("❌ ERROR: Please set KOTAK_ACCESS_TOKEN environment variable")
        return False

    try:
        # Initialize trader
        print(f"\n📊 Initializing trader (Test Mode: {test_mode})...")
        trader = KotakOptionsTrader(test_mode=test_mode)

        # Check account status
        status = trader.get_account_status()
        if not status['logged_in']:
            print(f"❌ Login failed: {status.get('error', 'Unknown error')}")
            return False

        print(f"✅ Login successful!")
        print(f"   Market Open: {status.get('market_open', 'Unknown')}")
        print(f"   Scrip Master: {status.get('scrip_master_loaded', False)}")

        # Check market hours (skip in test mode)
        if not test_mode and not is_market_open():
            print("❌ Market is closed. Cannot place real orders.")
            print("   💡 Tip: Run with test_mode=True for simulation")
            return False

        # Test parameters
        index_name = "NIFTY"
        hold_time = 60  # 1 minute

        print(f"\n📈 Starting trade test for {index_name}")
        print(f"   Hold time: {hold_time} seconds")

        # Step 1: BUY NIFTY CE (ATM)
        print(f"\n🔥 STEP 1: BUYING {index_name} CE...")
        print("-" * 50)

        buy_result = trader.execute_single_trade(
            timestamp=datetime.now(),
            index_name=index_name,
            signal_type="BUY"
        )

        print(f"📊 BUY Result:")
        print(f"   Status: {buy_result['status']}")
        print(f"   Index: {buy_result['index_name']}")
        print(f"   Price: {buy_result.get('price', 'N/A')}")
        print(f"   Strike: {buy_result.get('strike', 'N/A')}")
        print(f"   Option: {buy_result.get('option_type', 'N/A')}")
        print(f"   Action: {buy_result.get('action', 'N/A')}")

        if buy_result['status'] == 'error':
            print(f"❌ BUY failed: {buy_result.get('error')}")
            return False
        elif buy_result['status'] == 'skipped':
            print(f"⚠️  BUY skipped: {buy_result.get('reason')}")
            return False
        elif buy_result['status'] == 'success':
            print(f"✅ BUY executed successfully!")
            if 'order_response' in buy_result and buy_result['order_response']:
                order_resp = buy_result['order_response']
                if not test_mode and 'data' in order_resp:
                    print(f"   Order ID: {order_resp['data'].get('orderId', 'N/A')}")

        # Step 2: Hold for specified time
        print(f"\n⏰ STEP 2: HOLDING POSITION...")
        print("-" * 50)
        print(f"   Holding for {hold_time} seconds...")

        # Countdown timer
        for remaining in range(hold_time, 0, -10):
            print(f"   ⏳ {remaining} seconds remaining...")
            time.sleep(10)

        print("   ✅ Hold period completed!")

        # Step 3: SELL NIFTY CE
        print(f"\n💰 STEP 3: SELLING {index_name} CE...")
        print("-" * 50)

        sell_result = trader.execute_single_trade(
            timestamp=datetime.now(),
            index_name=index_name,
            signal_type="SELL"
        )

        print(f"📊 SELL Result:")
        print(f"   Status: {sell_result['status']}")
        print(f"   Index: {sell_result['index_name']}")
        print(f"   Price: {sell_result.get('price', 'N/A')}")
        print(f"   Action: {sell_result.get('action', 'N/A')}")

        if sell_result['status'] == 'error':
            print(f"❌ SELL failed: {sell_result.get('error')}")
            return False
        elif sell_result['status'] == 'skipped':
            print(f"⚠️  SELL skipped: {sell_result.get('reason')}")
            # This might be expected if no position exists
            pass
        elif sell_result['status'] == 'success':
            print(f"✅ SELL executed successfully!")
            if 'order_response' in sell_result and sell_result['order_response']:
                order_resp = sell_result['order_response']
                if not test_mode and 'data' in order_resp:
                    print(f"   Order ID: {order_resp['data'].get('orderId', 'N/A')}")

        # Final status
        print(f"\n🎉 TRADE CYCLE COMPLETED!")
        print("=" * 80)

        # Show final positions
        print(f"\n📋 Final Position Check:")
        positions = trader.get_existing_positions()
        nifty_positions = [pos for pos in positions if
                           'NIFTY' in pos.get('trdSym', '') and int(pos.get('flQty', 0)) != 0]

        if nifty_positions:
            print(f"   ⚠️  Remaining NIFTY positions:")
            for pos in nifty_positions:
                print(f"      {pos.get('trdSym')}: {pos.get('flQty')} qty")
        else:
            print(f"   ✅ No remaining NIFTY positions (as expected)")

        return True

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_operations(test_mode=True):
    """
    Test individual operations separately for debugging
    """
    print("\n" + "=" * 80)
    print("🔧 TESTING INDIVIDUAL OPERATIONS")
    print("=" * 80)

    load_dotenv()
    access_token = os.getenv('KOTAK_ACCESS_TOKEN')
    if not access_token:
        print("❌ ERROR: Please set KOTAK_ACCESS_TOKEN environment variable")
        return False

    trader = KotakOptionsTrader(test_mode=test_mode)

    # Test 1: Price fetching
    print(f"\n🔍 Test 1: Price Fetching")
    try:
        price = trader.get_current_price("NIFTY")
        print(f"   ✅ NIFTY Price: {price}")
    except Exception as e:
        print(f"   ❌ Price fetch failed: {e}")

    # Test 2: Instrument finding
    print(f"\n🔍 Test 2: Instrument Finding")
    try:
        from trade_placement import get_expiry_date
        expiry = get_expiry_date("NIFTY")
        price = trader.get_current_price("NIFTY")
        strike = trader.get_itm_strike("NIFTY", price, "CE")

        instrument = trader.find_option_instrument("NIFTY", strike, "CE", expiry)
        if instrument:
            print(f"   ✅ Found instrument: {instrument['trading_symbol']}")
            print(f"      Token: {instrument['instrument_token']}")
            print(f"      Lot Size: {instrument['lot_size']}")
        else:
            print(f"   ❌ Instrument not found")
    except Exception as e:
        print(f"   ❌ Instrument finding failed: {e}")

    # Test 3: Position checking
    print(f"\n🔍 Test 3: Position Checking")
    try:
        positions = trader.get_existing_positions()
        print(f"   ✅ Found {len(positions)} total positions")

        nifty_pos = trader.find_position_by_index_and_type("NIFTY", "CE")
        if nifty_pos:
            print(f"   📊 Existing NIFTY CE: {nifty_pos.get('trdSym')} - {nifty_pos.get('flQty')} qty")
        else:
            print(f"   ℹ️  No existing NIFTY CE positions")
    except Exception as e:
        print(f"   ❌ Position checking failed: {e}")


if __name__ == "__main__":
    print("🚀 KOTAK OPTIONS TRADER - TESTING SUITE")
    print("=" * 80)

    # Menu
    print("\nSelect test type:")
    print("1. 🧪 Complete Trade Cycle (TEST MODE)")
    print("2. 🔧 Individual Operations Test")
    print("3. 🔥 Complete Trade Cycle (LIVE MODE - REAL MONEY!)")
    print("4. ❌ Exit")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        print("\n🧪 Running complete trade cycle in TEST MODE...")
        success = test_complete_trade_cycle(test_mode=True)
        if success:
            print("\n✅ Test completed successfully!")
            print("💡 If everything looks good, you can try option 3 for live testing")
        else:
            print("\n❌ Test failed. Please check the errors above.")

    elif choice == "2":
        print("\n🔧 Running individual operations test...")
        test_individual_operations(test_mode=True)

    elif choice == "3":
        print("\n⚠️  WARNING: This will place REAL orders with REAL money!")
        confirm = input("Are you sure? Type 'YES' to continue: ").strip()
        if confirm == "YES":
            print("\n🔥 Running complete trade cycle in LIVE MODE...")
            success = test_complete_trade_cycle(test_mode=False)
            if success:
                print("\n🎉 Live test completed!")
            else:
                print("\n❌ Live test failed. Check your positions!")
        else:
            print("❌ Live test cancelled")

    elif choice == "4":
        print("👋 Goodbye!")

    else:
        print("❌ Invalid choice")

    print("\n" + "=" * 80)
    print("🏁 TEST SUITE COMPLETED")
    print("=" * 80)