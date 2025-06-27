# trade_placement_enhanced.py - Phase 1: Imports and Class Init
import pandas as pd
from datetime import datetime, timedelta
import calendar

from dotenv import load_dotenv
from neo_api_client import NeoAPI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_last_thursday(year, month):
    """Get the last Thursday of the given month"""
    # Get the last day of the month
    last_day = calendar.monthrange(year, month)[1]
    last_date = datetime(year, month, last_day).date()

    # Find the last Thursday
    days_to_subtract = (last_date.weekday() - 3) % 7
    if days_to_subtract == 0 and last_date.weekday() != 3:
        days_to_subtract = 7
    last_thursday = last_date - timedelta(days=days_to_subtract)

    return last_thursday


def get_expiry_date(index_name):
    """
    Get the appropriate expiry date based on current date
    Returns current month expiry if >7 days remaining, else next month
    """
    today = datetime.now().date()

    # Get current month's last Thursday
    current_month_expiry = _get_last_thursday(today.year, today.month)

    # Calculate days remaining
    days_remaining = (current_month_expiry - today).days

    if days_remaining > 7:  # Changed from 10 to 7 days
        logger.info(f"Using current month expiry: {current_month_expiry} ({days_remaining} days remaining)")
        return current_month_expiry
    else:
        # Get next month's expiry
        next_month = today.month + 1 if today.month < 12 else 1
        next_year = today.year if today.month < 12 else today.year + 1
        next_expiry = _get_last_thursday(next_year, next_month)
        days_to_next = (next_expiry - today).days
        logger.info(f"Using next month expiry: {next_expiry} ({days_to_next} days remaining)")
        return next_expiry


def is_market_open():
    """
    Check if market is open for trading
    Market hours: 9:15 AM to 3:30 PM on weekdays
    """
    now = datetime.now()

    # Check if it's a weekday (Monday = 0, Sunday = 6)
    if now.weekday() > 4:  # Saturday or Sunday
        return False

    # Check time
    market_open = now.replace(hour=9, minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)

    return market_open <= now <= market_close


def _get_expiry_for_testing(month_offset):
    """Get expiry date for testing purposes"""
    today = datetime.now().date()
    target_month = today.month + month_offset
    target_year = today.year

    if target_month > 12:
        target_month -= 12
        target_year += 1

    return _get_last_thursday(target_year, target_month)


def test_trader():
    """
    Test the trader without placing real orders
    """
    # Get access token from environment
    import os
    load_dotenv()
    access_token = os.getenv('KOTAK_ACCESS_TOKEN')
    if not access_token:
        print("Please set KOTAK_ACCESS_TOKEN environment variable")
        return

    # Initialize in test mode
    trader = KotakOptionsTrader(access_token=access_token, test_mode=True)

    # Run setup test
    trader.test_strategy_setup()

    # Test some trades
    print("\n\nTESTING TRADE EXECUTION (Test Mode):")
    print("=" * 60)

    # Test BUY signal
    result = trader.execute_single_trade(
        timestamp=datetime.now(),
        index_name="NIFTY",
        signal_type="BUY"
    )
    print(f"\nBUY Test: {result['status']}")
    if result['status'] == 'error':
        print(f"Error: {result['error']}")

    # Test SELL signal
    result = trader.execute_single_trade(
        timestamp=datetime.now(),
        index_name="NIFTY",
        signal_type="SELL"
    )
    print(f"\nSELL Test: {result['status']}")
    if result['status'] == 'skipped':
        print(f"Reason: {result['reason']}")

    # Test SHORT signal
    result = trader.execute_single_trade(
        timestamp=datetime.now(),
        index_name="BANKNIFTY",
        signal_type="SHORT"
    )
    print(f"\nSHORT Test: {result['status']}")
    if result['status'] == 'error':
        print(f"Error: {result['error']}")

    # Test COVER signal
    result = trader.execute_single_trade(
        timestamp=datetime.now(),
        index_name="BANKNIFTY",
        signal_type="COVER"
    )
    print(f"\nCOVER Test: {result['status']}")
    if result['status'] == 'skipped':
        print(f"Reason: {result['reason']}")


def live_trader_example():
    """
    Example of using the trader in live mode
    """
    import os
    load_dotenv()
    access_token = os.getenv('KOTAK_ACCESS_TOKEN')
    if not access_token:
        raise ValueError("Please set KOTAK_ACCESS_TOKEN environment variable")

    # Initialize in live mode
    trader = KotakOptionsTrader(access_token=access_token, test_mode=False)

    # Check account status
    status = trader.get_account_status()
    print(f"Account Status: {status}")

    if not status['logged_in']:
        print("Failed to login. Please check your access token.")
        return

    # Check if market is open
    if not is_market_open():
        print("Market is closed. Trading hours are 9:15 AM to 3:30 PM on weekdays.")
        return

    # Example: Execute a trade based on signal
    try:
        result = trader.execute_single_trade(
            timestamp=datetime.now(),
            index_name="NIFTY",
            signal_type="BUY"
        )

        if result['status'] == 'success':
            print(f"Trade executed successfully!")
            print(f"Order ID: {result['order_response'].get('order_id')}")
            print(f"Option: {result.get('option_type')} Strike: {result.get('strike')}")
        elif result['status'] == 'skipped':
            print(f"Trade skipped: {result['reason']}")
        else:
            print(f"Trade failed: {result.get('error')}")

    except Exception as e:
        print(f"Error: {e}")


class KotakOptionsTrader:
    def __init__(self, access_token, environment='prod', test_mode=False):
        """
        Initialize Kotak Neo API client with access token only

        Args:
            access_token: Pre-generated access token for automated trading
            environment: 'prod' or 'uat'
            test_mode: If True, simulates orders without placing them
        """
        self.client = NeoAPI(access_token=access_token)
        self.access_token = access_token
        self.is_logged_in = False
        self.scrip_master = None
        self.scrip_master_df = None
        self.test_mode = test_mode

        if self.test_mode:
            logger.info("*** RUNNING IN TEST MODE - NO REAL ORDERS WILL BE PLACED ***")

        # Index symbol mappings
        self.index_symbols = {
            'NIFTY': 'NIFTY',
            'BANKNIFTY': 'BANKNIFTY',
            'FINNIFTY': 'FINNIFTY',
            'MIDCPNIFTY': 'MIDCPNIFTY',
            'SENSEX': 'SENSEX'
        }

        # Strike price intervals for each index
        self.strike_intervals = {
            'NIFTY': 50,
            'BANKNIFTY': 100,
            'FINNIFTY': 50,
            'MIDCPNIFTY': 25,
            'SENSEX': 100
        }

        # Index instrument tokens for getting LTP
        self.index_tokens = {
            'NIFTY': '26000',
            'BANKNIFTY': '26009',
            'FINNIFTY': '26037',
            'MIDCPNIFTY': '26074',
            'SENSEX': '1'
        }

        # ITM strikes to go (1 = first ITM, 2 = second ITM)
        self.itm_strikes = 1  # Start with 1 strike ITM for safety

        # Automatically login and load scrip master on initialization
        self.auto_login()

    def auto_login(self):
        """
        Automated login using access token and load scrip master
        """
        try:
            # The NeoAPI client is already initialized with access token
            # Just verify it's working by making a test call
            if not self.test_mode:
                self.client.limits()
            self.is_logged_in = True
            logger.info("Successfully logged in using access token")

            # Load scrip master after successful login
            self.load_scrip_master()
            return True

        except Exception as e:
            logger.error(f"Auto login failed: {e}")
            self.is_logged_in = False
            return False

    def load_scrip_master(self):
        """
        Load and parse scrip master data from Kotak Neo API
        """
        try:
            logger.info("Loading scrip master data...")

            if self.test_mode:
                # In test mode, create minimal scrip master for testing
                logger.info("Test mode: Creating mock scrip master")
                self.scrip_master_df = self._create_mock_scrip_master()
                return True

            # Get scrip master data from Kotak Neo API
            scrip_master = self.client.scrip_master()

            if scrip_master and 'data' in scrip_master:
                self.scrip_master = scrip_master['data']

                # Convert to DataFrame for easier filtering
                self.scrip_master_df = pd.DataFrame(self.scrip_master)

                logger.info(f"Scrip master loaded successfully with {len(self.scrip_master)} instruments")
                return True
            else:
                logger.error("Failed to load scrip master - no data received")
                return False

        except Exception as e:
            logger.error(f"Error loading scrip master: {e}")
            return False

    def _create_mock_scrip_master(self):
        """Create mock scrip master for testing"""
        mock_data = []
        indices = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX']

        for index in indices:
            base_price = {'NIFTY': 24000, 'BANKNIFTY': 52000, 'FINNIFTY': 24500,
                          'MIDCPNIFTY': 12000, 'SENSEX': 80000}[index]
            interval = self.strike_intervals[index]

            # Create options for current and next month
            for month_offset in [0, 1]:
                expiry_date = _get_expiry_for_testing(month_offset)
                expiry_str = expiry_date.strftime("%d%b%Y").upper()

                # Create strikes around ATM
                for i in range(-5, 6):
                    strike = base_price + (i * interval)
                    for opt_type in ['CE', 'PE']:
                        mock_data.append({
                            'pSymbol': index,
                            'pExchSeg': 'nse_fo',
                            'pInstType': 'OPTIDX',
                            'pExpiry': expiry_str,
                            'pStrikePrice': str(strike),
                            'pOptionType': opt_type,
                            'pToken': f'TEST_{index}_{strike}_{opt_type}_{expiry_str}',
                            'pTrdSymbol': f'{index}{expiry_str}{strike}{opt_type}',
                            'pLotSize': {'NIFTY': 25, 'BANKNIFTY': 15, 'FINNIFTY': 25,
                                         'MIDCPNIFTY': 50, 'SENSEX': 10}[index]
                        })

        return pd.DataFrame(mock_data)

    def get_itm_strike(self, index_name, current_price, option_type):
        """
        Get the ITM strike price for the given index and option type

        For CE (Call): ITM strikes are BELOW current price
        For PE (Put): ITM strikes are ABOVE current price
        """
        interval = self.strike_intervals[index_name]
        atm_strike = round(current_price / interval) * interval

        if option_type == 'CE':
            # For calls, ITM is below current price
            itm_strike = atm_strike - (interval * self.itm_strikes)
            logger.info(f"CE ITM Strike: {itm_strike} (Current: {current_price}, ATM: {atm_strike})")
        else:  # PE
            # For puts, ITM is above current price
            itm_strike = atm_strike + (interval * self.itm_strikes)
            logger.info(f"PE ITM Strike: {itm_strike} (Current: {current_price}, ATM: {atm_strike})")

        return itm_strike

    def get_current_price(self, index_name):
        """Get current market price for the index"""
        try:
            if self.test_mode:
                # Return mock prices for testing
                mock_prices = {
                    'NIFTY': 24000,
                    'BANKNIFTY': 52000,
                    'FINNIFTY': 24500,
                    'MIDCPNIFTY': 12000,
                    'SENSEX': 80000
                }
                return mock_prices.get(index_name, 24000)

            # Use index token to get current price
            token = self.index_tokens.get(index_name)
            if not token:
                raise ValueError(f"Index token not found for {index_name}")

            quotes = self.client.quotes(
                instrument_tokens=[token],
                quote_type="ltp",
                isIndex=True
            )

            if quotes and 'data' in quotes and len(quotes['data']) > 0:
                return float(quotes['data'][0]['ltp'])
            else:
                raise Exception(f"No price data received for {index_name}")

        except Exception as e:
            logger.error(f"Error getting current price for {index_name}: {e}")
            raise

    def find_option_instrument(self, index_name, strike_price, option_type, expiry_date):
        """
        Find option instrument details from scrip master

        Args:
            index_name: Index name (NIFTY, BANKNIFTY, etc.)
            strike_price: Strike price
            option_type: 'CE' or 'PE'
            expiry_date: Expiry date (datetime.date object)

        Returns:
            dict: Instrument details with token, symbol, etc.
        """
        if self.scrip_master_df is None:
            raise Exception("Scrip master not loaded. Please check connection.")

        try:
            # Format expiry date for matching
            expiry_str = expiry_date.strftime("%d%b%Y").upper()

            # Filter criteria
            criteria = (
                    (self.scrip_master_df['pSymbol'] == index_name) &
                    (self.scrip_master_df['pExchSeg'] == 'nse_fo') &
                    (self.scrip_master_df['pInstType'] == 'OPTIDX') &
                    (self.scrip_master_df['pExpiry'] == expiry_str) &
                    (self.scrip_master_df['pStrikePrice'].astype(float) == float(strike_price)) &
                    (self.scrip_master_df['pOptionType'] == option_type)
            )

            matching_instruments = self.scrip_master_df[criteria]

            if len(matching_instruments) == 0:
                logger.error(f"No instrument found for {index_name} {strike_price} {option_type} {expiry_str}")
                # Log available strikes for debugging
                available = self.scrip_master_df[
                    (self.scrip_master_df['pSymbol'] == index_name) &
                    (self.scrip_master_df['pOptionType'] == option_type) &
                    (self.scrip_master_df['pExpiry'] == expiry_str)
                    ]['pStrikePrice'].unique()
                logger.error(f"Available strikes: {sorted(available.astype(float))}")
                return None

            if len(matching_instruments) > 1:
                logger.warning(f"Multiple instruments found, using first one: {len(matching_instruments)}")

            instrument = matching_instruments.iloc[0]

            return {
                'instrument_token': instrument['pToken'],
                'trading_symbol': instrument['pTrdSymbol'],
                'symbol': instrument['pSymbol'],
                'strike_price': instrument['pStrikePrice'],
                'option_type': instrument['pOptionType'],
                'expiry': instrument['pExpiry'],
                'lot_size': int(instrument['pLotSize'])
            }

        except Exception as e:
            logger.error(f"Error finding option instrument: {e}")
            return None

    def get_existing_positions(self):
        """Get all existing positions from Kotak"""
        try:
            if self.test_mode:
                # Return mock positions for testing
                return []

            if not self.is_logged_in:
                raise Exception("Not logged in. Please check access token.")

            positions = self.client.positions()
            return positions.get('data', [])
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def find_position_by_index_and_type(self, index_name, option_type):
        """
        Find existing position that matches the index and option type
        """
        positions = self.get_existing_positions()

        for position in positions:
            symbol = position.get('trdSym', '')
            quantity = int(position.get('flQty', 0))

            # Check if this is an option of the specified index and type
            if (index_name in symbol and
                    option_type in symbol and
                    quantity != 0):
                logger.info(f"Found existing position: {symbol}, Qty: {quantity}")
                return position

        logger.info(f"No existing {option_type} position found for {index_name}")
        return None

    def place_order(self, instrument_token, trading_symbol, quantity, order_type, transaction_type, price=None):
        """
        Place order using Kotak Neo API with proper instrument token

        Args:
            instrument_token: Instrument token from scrip master
            trading_symbol: Trading symbol from scrip master
            quantity: Quantity to trade
            order_type: 'MKT' for market, 'LMT' for limit
            transaction_type: 'B' for buy, 'S' for sell
            price: Price for limit orders
        """
        try:
            # Check market hours
            if not is_market_open() and not self.test_mode:
                raise Exception("Market is closed. Orders can only be placed between 9:15 AM and 3:30 PM on weekdays.")

            order_params = {
                "instrument_token": instrument_token,
                "exchange_segment": "nse_fo",
                "product": "NRML",
                "price": str(price) if price else "0",
                "order_type": order_type,
                "quantity": str(quantity),
                "validity": "DAY",
                "trading_symbol": trading_symbol,
                "transaction_type": transaction_type,
                "amo": "NO",
                "disclosed_quantity": "0",
                "market_protection": "0",
                "pf": "N",
                "trigger_price": "0",
                "tag": "automated_trade"
            }

            if self.test_mode:
                logger.info(f"TEST MODE - Would place order: {order_params}")
                return {
                    'status': 'success',
                    'order_id': f'TEST_ORDER_{datetime.now().timestamp()}',
                    'message': 'Test order placed successfully',
                    'params': order_params
                }

            response = self.client.place_order(**order_params)
            logger.info(f"Order placed successfully: {response}")
            return response

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise

    def validate_trade(self, index_name, signal_type):
        """
        Validate if a trade should be placed
        Returns: (bool, str) - (can_trade, reason)
        """
        # Check if we already have a position for this signal type
        if signal_type in ['BUY', 'SELL']:
            existing = self.find_position_by_index_and_type(index_name, 'CE')
            if signal_type == 'BUY' and existing:
                return False, f"Already have CE position for {index_name}"
            elif signal_type == 'SELL' and not existing:
                return False, f"No CE position to sell for {index_name}"

        elif signal_type in ['SHORT', 'COVER']:
            existing = self.find_position_by_index_and_type(index_name, 'PE')
            if signal_type == 'SHORT' and existing:
                return False, f"Already have PE position for {index_name}"
            elif signal_type == 'COVER' and not existing:
                return False, f"No PE position to cover for {index_name}"

        return True, "Trade validation passed"

    def execute_single_trade(self, timestamp, index_name, signal_type, price=None):
        """
        Execute a single trade based on timestamp, signal type
        Gets current price automatically if not provided

        Args:
            timestamp: Trading signal timestamp (pandas Timestamp or datetime)
            index_name: Index to trade ('NIFTY', 'BANKNIFTY', etc.)
            signal_type: 'BUY', 'SELL', 'SHORT', 'COVER'
            price: Current index price (optional, will fetch if not provided)

        Returns:
            dict: Trade execution result
        """

        # Ensure we're logged in
        if not self.is_logged_in:
            raise Exception("Not logged in. Please check access token.")

        try:
            # Convert timestamp if needed
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)

            # Get current price if not provided
            if price is None:
                price = self.get_current_price(index_name)

            logger.info(f"Processing trade: {timestamp} - {signal_type} - {index_name} at {price}")

            # Validate trade
            can_trade, reason = self.validate_trade(index_name, signal_type)
            if not can_trade:
                logger.warning(f"Trade validation failed: {reason}")
                return {
                    'timestamp': timestamp,
                    'signal_type': signal_type,
                    'index_name': index_name,
                    'price': price,
                    'status': 'skipped',
                    'reason': reason,
                    'order_response': None
                }

            # Get expiry date based on 7-day logic
            expiry_date = get_expiry_date(index_name)

            result = {
                'timestamp': timestamp,
                'signal_type': signal_type,
                'index_name': index_name,
                'price': price,
                'expiry_date': expiry_date,
                'status': 'pending',
                'order_response': None,
                'error': None
            }

            if signal_type == 'BUY':
                # BUY = CE (Call Option) - ITM
                itm_strike = self.get_itm_strike(index_name, price, 'CE')
                order_response = self._handle_buy_signal(index_name, itm_strike, expiry_date, 'CE')
                result['option_type'] = 'CE'
                result['strike'] = itm_strike
                result['action'] = 'BUY_NEW'

            elif signal_type == 'SELL':
                # SELL existing CE position
                order_response = self._handle_sell_signal(index_name, 'CE')
                result['option_type'] = 'CE'
                result['action'] = 'SELL_EXISTING'

            elif signal_type == 'SHORT':
                # SHORT = PE (Put Option) - ITM
                itm_strike = self.get_itm_strike(index_name, price, 'PE')
                order_response = self._handle_short_signal(index_name, itm_strike, expiry_date, 'PE')
                result['option_type'] = 'PE'
                result['strike'] = itm_strike
                result['action'] = 'SHORT_NEW'

            elif signal_type == 'COVER':
                # COVER existing PE position
                order_response = self._handle_cover_signal(index_name, 'PE')
                result['option_type'] = 'PE'
                result['action'] = 'COVER_EXISTING'

            else:
                raise ValueError(f"Invalid signal type: {signal_type}")

            result['order_response'] = order_response
            result['status'] = 'success' if order_response else 'no_action'

            logger.info(f"Trade executed successfully: {signal_type} - {result['status']}")
            return result

        except Exception as e:
            error_msg = f"Error executing trade: {str(e)}"
            logger.error(error_msg)

            result = {
                'timestamp': timestamp,
                'signal_type': signal_type,
                'index_name': index_name,
                'price': price,
                'status': 'error',
                'error': error_msg,
                'order_response': None
            }

            return result

    def _handle_buy_signal(self, index_name, strike_price, expiry_date, option_type):
        """Handle BUY signal - Buy CE option"""
        # Get instrument details from scrip master
        instrument = self.find_option_instrument(index_name, strike_price, option_type, expiry_date)

        if not instrument:
            raise Exception(f"Instrument not found for {index_name} {strike_price} {option_type}")

        response = self.place_order(
            instrument_token=instrument['instrument_token'],
            trading_symbol=instrument['trading_symbol'],
            quantity=instrument['lot_size'],
            order_type="MKT",
            transaction_type="B"
        )

        logger.info(f"BUY order placed for {instrument['trading_symbol']}, Quantity: {instrument['lot_size']}")
        return response

    def _handle_sell_signal(self, index_name, option_type):
        """Handle SELL signal - Sell existing CE position"""
        position = self.find_position_by_index_and_type(index_name, option_type)

        if not position:
            logger.warning(f"No existing {option_type} position found for {index_name}")
            return None

        # Get instrument token from position
        instrument_token = position.get('tok', '')
        trading_symbol = position.get('trdSym', '')
        quantity = abs(int(position.get('flQty', 0)))

        if quantity > 0:
            response = self.place_order(
                instrument_token=instrument_token,
                trading_symbol=trading_symbol,
                quantity=quantity,
                order_type="MKT",
                transaction_type="S"
            )

            logger.info(f"SELL order placed for {trading_symbol}, Quantity: {quantity}")
            return response
        else:
            logger.warning(f"No quantity to sell for {trading_symbol}")
            return None

    def _handle_short_signal(self, index_name, strike_price, expiry_date, option_type):
        """Handle SHORT signal - Buy PE option"""
        # Get instrument details from scrip master
        instrument = self.find_option_instrument(index_name, strike_price, option_type, expiry_date)

        if not instrument:
            raise Exception(f"Instrument not found for {index_name} {strike_price} {option_type}")

        response = self.place_order(
            instrument_token=instrument['instrument_token'],
            trading_symbol=instrument['trading_symbol'],
            quantity=instrument['lot_size'],
            order_type="MKT",
            transaction_type="B"  # BUY PE for SHORT signal
        )

        logger.info(
            f"SHORT order (Buy PE) placed for {instrument['trading_symbol']}, Quantity: {instrument['lot_size']}")
        return response

    def _handle_cover_signal(self, index_name, option_type):
        """Handle COVER signal - Sell existing PE position"""
        position = self.find_position_by_index_and_type(index_name, option_type)

        if not position:
            logger.warning(f"No existing {option_type} position found for {index_name}")
            return None

        # Get instrument token from position
        instrument_token = position.get('tok', '')
        trading_symbol = position.get('trdSym', '')
        quantity = abs(int(position.get('flQty', 0)))

        if quantity > 0:
            response = self.place_order(
                instrument_token=instrument_token,
                trading_symbol=trading_symbol,
                quantity=quantity,
                order_type="MKT",
                transaction_type="S"  # SELL PE for COVER signal
            )

            logger.info(f"COVER order (Sell PE) placed for {trading_symbol}, Quantity: {quantity}")
            return response
        else:
            logger.warning(f"No quantity to cover for {trading_symbol}")
            return None

    def get_account_status(self):
        """Get account and login status"""
        try:
            if not self.is_logged_in:
                return {"logged_in": False, "message": "Not logged in"}

            if self.test_mode:
                return {
                    "logged_in": True,
                    "test_mode": True,
                    "message": "Running in test mode"
                }

            # Get account details
            limits = self.client.limits()
            positions = self.client.positions()

            return {
                "logged_in": True,
                "limits": limits,
                "positions_count": len(positions.get('data', [])),
                "scrip_master_loaded": self.scrip_master_df is not None,
                "total_instruments": len(self.scrip_master_df) if self.scrip_master_df is not None else 0,
                "message": "Account active",
                "market_open": is_market_open()
            }
        except Exception as e:
            return {"logged_in": False, "error": str(e)}

    def test_strategy_setup(self):
        """
        Test the complete setup without placing actual orders
        """
        print("\n" + "=" * 60)
        print("TESTING KOTAK OPTIONS TRADER SETUP")
        print("=" * 60 + "\n")

        # 1. Check login status
        status = self.get_account_status()
        print(f"1. Login Status: {'✅ Connected' if status['logged_in'] else '❌ Not Connected'}")
        if 'error' in status:
            print(f"   Error: {status['error']}")

        # 2. Check market hours
        market_open = is_market_open()
        print(f"2. Market Status: {'✅ Open' if market_open else '❌ Closed'}")
        print(f"   Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 3. Test each index
        indices = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX']

        for index in indices:
            print(f"\n3. Testing {index}:")
            try:
                # Get current price
                price = self.get_current_price(index)
                print(f"   Current Price: {price}")

                # Get expiry
                expiry = get_expiry_date(index)
                print(f"   Selected Expiry: {expiry}")

                # Get ITM strikes
                ce_strike = self.get_itm_strike(index, price, 'CE')
                pe_strike = self.get_itm_strike(index, price, 'PE')

                # Find instruments
                ce_instrument = self.find_option_instrument(index, ce_strike, 'CE', expiry)
                pe_instrument = self.find_option_instrument(index, pe_strike, 'PE', expiry)

                if ce_instrument:
                    print(f"   CE Option: {ce_instrument['trading_symbol']} (Lot: {ce_instrument['lot_size']})")
                else:
                    print(f"   CE Option: Not found")

                if pe_instrument:
                    print(f"   PE Option: {pe_instrument['trading_symbol']} (Lot: {pe_instrument['lot_size']})")
                else:
                    print(f"   PE Option: Not found")

            except Exception as e:
                print(f"   Error: {str(e)}")

        # 4. Check existing positions
        print(f"\n4. Existing Positions:")
        positions = self.get_existing_positions()
        if positions:
            for pos in positions:
                if int(pos.get('flQty', 0)) != 0:
                    print(f"   {pos.get('trdSym')}: {pos.get('flQty')} qty")
        else:
            print("   No open positions")

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)

    # Example usage for testing


if __name__ == "__main__":
    # Run test by default
    test_trader()
