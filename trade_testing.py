# test_all_indices.py
"""
Simple test script to test BUY CE -> Hold 1 min -> SELL CE for all indices
Tests: NIFTY, BANKNIFTY, SENSEX, FINNIFTY, MIDCPNIFTY
"""

import time
from datetime import datetime
from trade_placement import KotakOptionsTrader


def test_all_indices(test_mode=True):
    """
    Test complete trade cycle for all indices

    Args:
        test_mode: If True, simulates orders. If False, places real orders
    """
    print("\n" + "=" * 80)
    print(f"🚀 TESTING ALL INDICES - {'TEST MODE' if test_mode else 'LIVE MODE'}")
    print("=" * 80)

    # Initialize trader
    print("\n📊 Initializing trader...")
    trader = KotakOptionsTrader(test_mode=test_mode)

    if not trader.is_logged_in:
        print("❌ Trader initialization failed!")
        print("💡 Please run 'python daily_auth.py' first to authenticate")
        return False

    print("✅ Trader initialized successfully!")

    # List of indices to test
    # indices = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY"]
    indices = ["SENSEX"]
    hold_time = 60  # 1 minute

    # Store results
    results = {}

    # Test each index
    for index_name in indices:
        print(f"\n{'=' * 60}")
        print(f"📈 Testing {index_name}")
        print(f"{'=' * 60}")

        results[index_name] = {
            'buy_status': None,
            'sell_status': None,
            'error': None
        }

        try:

            print(f"\n🔥 BUYING {index_name} CE...")
            buy_result = trader.execute_single_trade(
                timestamp=datetime.now(),
                index_name=index_name,
                signal_type="BUY"
            )

            results[index_name]['buy_status'] = buy_result['status']

            print(f"Status: {buy_result['status']}")
            if buy_result['status'] == 'success':
                print(f"✅ BUY successful!")
                print(f"   Strike: {buy_result.get('strike', 'N/A')}")
                print(f"   Price: {buy_result.get('price', 'N/A')}")
                if 'order_response' in buy_result and buy_result['order_response']:
                    if test_mode:
                        print(
                            f"   Test Order ID: {buy_result['order_response'].get('data', {}).get('orderId', 'TEST')}")
                    else:
                        print(f"   Order ID: {buy_result['order_response'].get('data', {}).get('orderId', 'N/A')}")
            elif buy_result['status'] == 'skipped':
                print(f"⚠️  BUY skipped: {buy_result.get('reason')}")
            else:
                print(f"❌ BUY failed: {buy_result.get('error')}")
                results[index_name]['error'] = buy_result.get('error')
                continue  # Skip to next index

            # Step 2: Hold
            print(f"\n⏰ Holding for {hold_time} seconds...")
            for remaining in range(hold_time, 0, -20):
                print(f"   {remaining} seconds remaining...")
                time.sleep(min(20, remaining))

            # Step 3: SELL CE
            print(f"\n💰 SELLING {index_name} CE...")
            sell_result = trader.execute_single_trade(
                timestamp=datetime.now(),
                index_name=index_name,
                signal_type="SELL"
            )

            results[index_name]['sell_status'] = sell_result['status']

            print(f"Status: {sell_result['status']}")
            if sell_result['status'] == 'success':
                print(f"✅ SELL successful!")
                if 'order_response' in sell_result and sell_result['order_response']:
                    if test_mode:
                        print(
                            f"   Test Order ID: {sell_result['order_response'].get('data', {}).get('orderId', 'TEST')}")
                    else:
                        print(f"   Order ID: {sell_result['order_response'].get('data', {}).get('orderId', 'N/A')}")
            elif sell_result['status'] == 'skipped':
                print(f"⚠️  SELL skipped: {sell_result.get('reason')}")
            else:
                print(f"❌ SELL failed: {sell_result.get('error')}")
                results[index_name]['error'] = sell_result.get('error')

            print(f"\n🔥 SELLLINNGGGGGG NOWWWWWWWW...")

            # Step 1: BUY PE
            print(f"\n🔥 BUYING {index_name} CE...")
            buy_result = trader.execute_single_trade(
                timestamp=datetime.now(),
                index_name=index_name,
                signal_type="SHORT"
            )

            results[index_name]['buy_status'] = buy_result['status']

            print(f"Status: {buy_result['status']}")
            if buy_result['status'] == 'success':
                print(f"✅ BUY successful!")
                print(f"   Strike: {buy_result.get('strike', 'N/A')}")
                print(f"   Price: {buy_result.get('price', 'N/A')}")
                if 'order_response' in buy_result and buy_result['order_response']:
                    if test_mode:
                        print(
                            f"   Test Order ID: {buy_result['order_response'].get('data', {}).get('orderId', 'TEST')}")
                    else:
                        print(f"   Order ID: {buy_result['order_response'].get('data', {}).get('orderId', 'N/A')}")
            elif buy_result['status'] == 'skipped':
                print(f"⚠️  BUY skipped: {buy_result.get('reason')}")
            else:
                print(f"❌ BUY failed: {buy_result.get('error')}")
                results[index_name]['error'] = buy_result.get('error')
                continue  # Skip to next index

            # Step 2: Hold
            print(f"\n⏰ Holding for {hold_time} seconds...")
            for remaining in range(hold_time, 0, -20):
                print(f"   {remaining} seconds remaining...")
                time.sleep(min(20, remaining))

            # Step 3: SELL PE
            print(f"\n💰 SELLING {index_name} CE...")
            sell_result = trader.execute_single_trade(
                timestamp=datetime.now(),
                index_name=index_name,
                signal_type="COVER"
            )

            results[index_name]['sell_status'] = sell_result['status']

            print(f"Status: {sell_result['status']}")
            if sell_result['status'] == 'success':
                print(f"✅ SELL successful!")
                if 'order_response' in sell_result and sell_result['order_response']:
                    if test_mode:
                        print(
                            f"   Test Order ID: {sell_result['order_response'].get('data', {}).get('orderId', 'TEST')}")
                    else:
                        print(f"   Order ID: {sell_result['order_response'].get('data', {}).get('orderId', 'N/A')}")
            elif sell_result['status'] == 'skipped':
                print(f"⚠️  SELL skipped: {sell_result.get('reason')}")
            else:
                print(f"❌ SELL failed: {sell_result.get('error')}")
                results[index_name]['error'] = sell_result.get('error')

        except Exception as e:
            print(f"❌ Error testing {index_name}: {e}")
            results[index_name]['error'] = str(e)

    # Summary
    print(f"\n{'=' * 80}")
    print("📊 TEST SUMMARY")
    print(f"{'=' * 80}")

    for index, result in results.items():
        buy_icon = "✅" if result['buy_status'] == 'success' else "❌"
        sell_icon = "✅" if result['sell_status'] == 'success' else "❌"

        print(f"\n{index}:")
        print(f"  BUY:  {buy_icon} {result['buy_status'] or 'Not attempted'}")
        print(f"  SELL: {sell_icon} {result['sell_status'] or 'Not attempted'}")
        if result['error']:
            print(f"  Error: {result['error']}")

    # Check final positions
    print(f"\n📋 Final Position Check:")
    try:
        positions = trader.get_existing_positions()
        active_positions = [pos for pos in positions if int(pos.get('flBuyQty', 0)) != 0]

        if active_positions:
            print(f"⚠️  Found {len(active_positions)} active positions:")
            for pos in active_positions:
                print(f"   {pos.get('trdSym')}: {pos.get('flBuyQty')} qty")
        else:
            print("✅ No active positions (clean exit)")
    except Exception as e:
        print(f"⚠️  Could not check positions: {e}")

    print(f"\n{'=' * 80}")
    print("🏁 TEST COMPLETED")
    print(f"{'=' * 80}\n")


def quick_test_single_index(index_name="NIFTY", test_mode=True):
    """Quick test for a single index"""
    print(f"\n🚀 Quick test for {index_name} ({'TEST' if test_mode else 'LIVE'} MODE)")

    trader = KotakOptionsTrader(test_mode=test_mode)

    if not trader.is_logged_in:
        print("❌ Not logged in!")
        return

    # BUY
    print(f"\n🔥 BUYING {index_name} CE...")
    buy_result = trader.execute_single_trade(
        timestamp=datetime.now(),
        index_name=index_name,
        signal_type="BUY"
    )
    print(f"Result: {buy_result['status']}")

    if buy_result['status'] != 'success':
        print(f"Buy failed: {buy_result.get('error') or buy_result.get('reason')}")
        return

    # Hold
    print("\n⏰ Holding for 60 seconds...")
    time.sleep(60)

    # SELL
    print(f"\n💰 SELLING {index_name} CE...")
    sell_result = trader.execute_single_trade(
        timestamp=datetime.now(),
        index_name=index_name,
        signal_type="SELL"
    )
    print(f"Result: {sell_result['status']}")

    print("\n✅ Test complete!")


if __name__ == "__main__":
    # Simple choice
    print("Select test type:")
    print("1. Test all indices (TEST MODE)")
    print("2. Test all indices (LIVE MODE - REAL MONEY!)")
    print("3. Quick test NIFTY only (TEST MODE)")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        test_all_indices(test_mode=True)
    elif choice == "2":
        confirm = input("⚠️  This will place REAL orders! Type 'YES' to confirm: ")
        if confirm == "YES":
            test_all_indices(test_mode=False)
        else:
            print("Cancelled.")
    elif choice == "3":
        quick_test_single_index("NIFTY", test_mode=True)
    else:
        print("Invalid choice")