# trade_placement_fixed.py - Corrected Kotak Options Trader
"""
Corrected Kotak Options Trading Script
=====================================

This script provides reliable automated options trading with:
- Corrected API parameter usage (trading_symbol vs instrument_token)
- Simplified session management (no manual session parameter passing)
- Proper index name handling for quotes
- Comprehensive retry mechanisms with exponential backoff
- Circuit breaker protection
- Email alerts for critical failures
- Order verification and status checking
- Fallback strike selection
- Price validation

Prerequisites:
1. Run daily_auth.py once per day to authenticate
2. Configure email settings in .env file
3. Ensure all environment variables are set

Author: Fixed implementation based on official Neo API docs
"""

import pandas as pd
import json
import os
import time
import smtplib
from datetime import datetime, timedelta
import calendar
from functools import wraps
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
from neo_api_client import NeoAPI
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Circuit breaker configuration
MAX_RETRIES = 3
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_TIMEOUT = 300  # 5 minutes
circuit_breaker_failures = 0
circuit_breaker_last_failure = None

# Email configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': '',  # Will be loaded from environment
    'password': '',  # Will be loaded from environment
    'recipient': ''  # Will be loaded from environment
}


def load_email_config():
    """Load email configuration from environment"""
    load_dotenv()
    EMAIL_CONFIG['email'] = os.getenv('EMAIL_USERNAME', '')
    EMAIL_CONFIG['password'] = os.getenv('EMAIL_PASSWORD', '')
    EMAIL_CONFIG['recipient'] = os.getenv('EMAIL_RECIPIENT', EMAIL_CONFIG['email'])


def send_alert_email(subject, message):
    """Send email alert for critical failures"""
    try:
        if not all([EMAIL_CONFIG['email'], EMAIL_CONFIG['password'], EMAIL_CONFIG['recipient']]):
            logger.warning("Email configuration incomplete, skipping alert")
            return

        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['email']
        msg['To'] = EMAIL_CONFIG['recipient']
        msg['Subject'] = f"Trading Alert: {subject}"

        body = f"""
Trading Alert
=============
Time: {datetime.now()}
Subject: {subject}

Details:
{message}

Please check your trading system immediately.
        """

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
        server.send_message(msg)
        server.quit()

        logger.info(f"Alert email sent: {subject}")
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")


def check_circuit_breaker():
    """Check if circuit breaker is active"""
    global circuit_breaker_failures, circuit_breaker_last_failure

    if circuit_breaker_failures >= CIRCUIT_BREAKER_THRESHOLD:
        if circuit_breaker_last_failure:
            time_since_failure = time.time() - circuit_breaker_last_failure
            if time_since_failure < CIRCUIT_BREAKER_TIMEOUT:
                remaining = CIRCUIT_BREAKER_TIMEOUT - time_since_failure
                raise Exception(f"Circuit breaker active. Retry in {remaining:.0f} seconds")
            else:
                # Reset circuit breaker
                circuit_breaker_failures = 0
                circuit_breaker_last_failure = None
                logger.info("Circuit breaker reset")


def record_failure():
    """Record a failure for circuit breaker"""
    global circuit_breaker_failures, circuit_breaker_last_failure
    circuit_breaker_failures += 1
    circuit_breaker_last_failure = time.time()

    if circuit_breaker_failures >= CIRCUIT_BREAKER_THRESHOLD:
        send_alert_email("Circuit Breaker Activated",
                         f"Too many failures ({circuit_breaker_failures}). Trading halted for {CIRCUIT_BREAKER_TIMEOUT / 60} minutes.")


def retry_with_backoff(max_retries=MAX_RETRIES, base_delay=0.1):
    """Decorator for retrying functions with exponential backoff"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            check_circuit_breaker()

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        record_failure()
                        logger.error(f"Final attempt failed for {func.__name__}: {e}")
                        raise

                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s")
                    time.sleep(delay)

        return wrapper

    return decorator


def validate_price(price, index_name):
    """Validate if price is reasonable"""
    price_ranges = {
        'NIFTY': (15000, 35000),
        'BANKNIFTY': (35000, 70000),
        'FINNIFTY': (15000, 35000),
        'MIDCPNIFTY': (8000, 20000),
        'SENSEX': (50000, 100000)
    }

    min_price, max_price = price_ranges.get(index_name, (0, float('inf')))
    if not (min_price <= price <= max_price):
        raise ValueError(f"Price {price} for {index_name} outside expected range [{min_price}, {max_price}]")

    return True


def _get_last_thursday(year, month):
    """Get the last Thursday of the given month"""
    last_day = calendar.monthrange(year, month)[1]
    last_date = datetime(year, month, last_day).date()

    days_to_subtract = (last_date.weekday() - 3) % 7
    if days_to_subtract == 0 and last_date.weekday() != 3:
        days_to_subtract = 7
    last_thursday = last_date - timedelta(days=days_to_subtract)

    return last_thursday


def get_expiry_date(index_name):
    """Get the appropriate expiry date based on current date"""
    today = datetime.now().date()
    current_month_expiry = _get_last_thursday(today.year, today.month)
    days_remaining = (current_month_expiry - today).days

    if days_remaining > 7:
        logger.info(f"Using current month expiry: {current_month_expiry} ({days_remaining} days remaining)")
        return current_month_expiry
    else:
        next_month = today.month + 1 if today.month < 12 else 1
        next_year = today.year if today.month < 12 else today.year + 1
        next_expiry = _get_last_thursday(next_year, next_month)
        days_to_next = (next_expiry - today).days
        logger.info(f"Using next month expiry: {next_expiry} ({days_to_next} days remaining)")
        return next_expiry


def is_market_open():
    """Check if market is open for trading"""
    now = datetime.now()
    if now.weekday() > 4:  # Saturday or Sunday
        return False

    market_open = now.replace(hour=9, minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)
    return market_open <= now <= market_close


class KotakOptionsTrader:
    def __init__(self, test_mode=False):
        """
        Initialize Kotak Options Trader with corrected API implementation

        Args:
            test_mode: If True, simulates orders without placing them
        """
        load_dotenv()
        load_email_config()

        self.test_mode = test_mode
        self.is_logged_in = False
        self.client = None
        self.scrip_master = None
        self.scrip_master_df = None
        self.session_file = 'kotak_session.json'

        if self.test_mode:
            logger.info("*** RUNNING IN TEST MODE - NO REAL ORDERS WILL BE PLACED ***")

        # ✅ FIXED: Proper index symbol mapping for quotes API
        self.index_symbols = {
            'NIFTY': 'Nifty 50',
            'BANKNIFTY': 'Nifty Bank',
            'FINNIFTY': 'Nifty Fin Services',
            'MIDCPNIFTY': 'Nifty MidCap Select',
            'SENSEX': 'BSE Sensex'
        }

        self.strike_intervals = {
            'NIFTY': 50,
            'BANKNIFTY': 100,
            'FINNIFTY': 50,
            'MIDCPNIFTY': 25,
            'SENSEX': 100
        }

        self.itm_strikes = 1  # ITM strikes to go

        # Initialize client and login
        self.initialize_client()

    def initialize_client(self):
        """✅ FIXED: Simplified client initialization"""
        try:
            if self.test_mode:
                self.is_logged_in = True
                self.load_scrip_master()
                return True

            # Initialize client
            consumer_key = os.getenv('KOTAK_CONSUMER_KEY')
            consumer_secret = os.getenv('KOTAK_CONSUMER_SECRET')
            environment = os.getenv('KOTAK_ENVIRONMENT', 'prod')

            if not consumer_key or not consumer_secret:
                raise Exception("Missing KOTAK_CONSUMER_KEY or KOTAK_CONSUMER_SECRET")

            self.client = NeoAPI(
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                environment=environment
            )

            # ✅ FIXED: Simple session validation - let client handle session internally
            if self.validate_session():
                self.is_logged_in = True
                logger.info("Session validated successfully")
                self.load_scrip_master()
                return True
            else:
                logger.error("No valid session found. Please run daily_auth.py first")
                return False

        except Exception as e:
            logger.error(f"Client initialization failed: {e}")
            send_alert_email("Authentication Failed",
                             f"Failed to initialize client: {e}\nPlease run daily_auth.py")
            return False

    def validate_session(self):
        """✅ FIXED: Handle 'No Data' response as valid"""
        try:
            if not os.path.exists(self.session_file):
                logger.error("Session file not found")
                return False

            with open(self.session_file, 'r') as f:
                session_data = json.load(f)

            if not session_data.get('authenticated'):
                logger.error("Session not authenticated")
                return False

            # Restore session tokens
            if 'tokens' in session_data:
                config = self.client.api_client.configuration
                tokens = session_data['tokens']

                for key, value in tokens.items():
                    if value:
                        setattr(config, key, value)

            # Test session with positions API
            positions = self.client.positions()

            # ✅ FIX: Handle "No Data" as valid response (empty positions)
            if ('data' in positions) or (positions.get('stCode') == 5203 and positions.get('errMsg') == 'No Data'):
                logger.info("Session is valid (positions API accessible)")
                return True
            else:
                logger.error(f"Session validation failed: {positions}")
                return False

        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            return False

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def load_scrip_master(self):
        """✅ FIXED: Handle scrip master URLs and download CSV files"""
        try:
            logger.info("Loading scrip master data...")

            if self.test_mode:
                logger.info("Test mode: Creating mock scrip master")
                self.scrip_master_df = self._create_mock_scrip_master()
                return True

            # Get scrip master URLs
            scrip_response = self.client.scrip_master()

            if not scrip_response or 'filesPaths' not in scrip_response:
                raise Exception("Failed to get scrip master file paths")

            # Find NSE F&O file (needed for options trading)
            file_paths = scrip_response['filesPaths']
            nse_fo_url = None
            for file_path in file_paths:
                if 'nse_fo.csv' in file_path:
                    nse_fo_url = file_path
                    break

            if not nse_fo_url:
                raise Exception("NSE F&O scrip master file not found in response")

            logger.info(f"Downloading scrip master from: {nse_fo_url}")

            # Download and parse CSV
            import requests
            response = requests.get(nse_fo_url, timeout=30)
            response.raise_for_status()

            # Parse CSV content
            from io import StringIO
            csv_content = StringIO(response.text)
            self.scrip_master_df = pd.read_csv(csv_content)

            # ✅ FIX: Actually apply column standardization
            available_columns = self.scrip_master_df.columns.tolist()
            logger.info(f"Original columns: {available_columns}")

            # Standardize column names if needed
            column_mappings = {
                'symbol': 'pSymbol',
                'exchange_segment': 'pExchSeg',
                'instrument_type': 'pInstType',
                'expiry': 'pExpiry',
                'strike_price': 'pStrikePrice',
                'option_type': 'pOptionType',
                'token': 'pToken',
                'trading_symbol': 'pTrdSymbol',
                'lot_size': 'pLotSize'
            }

            # Apply column renaming
            self.scrip_master_df.rename(columns=column_mappings, inplace=True)
            logger.info(f"Standardized columns: {self.scrip_master_df.columns.tolist()}")

            if len(self.scrip_master_df) < 100:
                raise Exception(f"Scrip master seems incomplete: only {len(self.scrip_master_df)} instruments")

            logger.info(f"Scrip master loaded successfully with {len(self.scrip_master_df)} instruments")
            logger.info(f"Sample data:\n{self.scrip_master_df.head(2).to_string()}")

            return True

        except Exception as e:
            logger.error(f"Error loading scrip master: {e}")
            if not self.test_mode:
                send_alert_email("Scrip Master Load Failed", f"Failed to load scrip master: {e}")
            raise


    def _create_mock_scrip_master(self):
        """Create mock scrip master for testing"""
        mock_data = []
        indices = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX']

        for index in indices:
            base_price = {'NIFTY': 24000, 'BANKNIFTY': 52000, 'FINNIFTY': 24500,
                          'MIDCPNIFTY': 12000, 'SENSEX': 80000}[index]
            interval = self.strike_intervals[index]

            for month_offset in [0, 1]:
                expiry_date = self._get_expiry_for_testing(month_offset)
                expiry_str = expiry_date.strftime("%d%b%Y").upper()

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

    def _get_expiry_for_testing(self, month_offset):
        """Get expiry date for testing purposes"""
        today = datetime.now().date()
        target_month = today.month + month_offset
        target_year = today.year

        if target_month > 12:
            target_month -= 12
            target_year += 1

        return _get_last_thursday(target_year, target_month)

    @retry_with_backoff(max_retries=3, base_delay=0.2)
    def get_current_price(self, index_name):
        """✅ FIXED: Get current market price with corrected quotes API usage"""
        try:
            if self.test_mode:
                mock_prices = {
                    'NIFTY': 24000,
                    'BANKNIFTY': 52000,
                    'FINNIFTY': 24500,
                    'MIDCPNIFTY': 12000,
                    'SENSEX': 80000
                }
                price = mock_prices.get(index_name, 24000)
                validate_price(price, index_name)
                return price

            # ✅ FIXED: Use proper index symbol mapping
            index_symbol = self.index_symbols.get(index_name)
            if not index_symbol:
                raise ValueError(f"Index symbol not found for {index_name}")

            # ✅ FIXED: Proper instrument tokens format for indices
            instrument_tokens = [{
                "instrument_token": index_symbol,
                "exchange_segment": "nse_cm"
            }]

            # ✅ FIXED: No manual session parameter passing - client handles internally
            quotes = self.client.quotes(
                instrument_tokens=instrument_tokens,
                quote_type="ltp",
                isIndex=True
            )

            if quotes and 'data' in quotes and len(quotes['data']) > 0:
                price = float(quotes['data'][0]['ltp'])
                validate_price(price, index_name)
                logger.info(f"Current price for {index_name}: {price}")
                return price
            else:
                raise Exception(f"No price data received for {index_name}")

        except Exception as e:
            logger.error(f"Error getting current price for {index_name}: {e}")
            raise

    def get_itm_strike(self, index_name, current_price, option_type):
        """Get the ITM strike price for the given index and option type"""
        interval = self.strike_intervals[index_name]
        atm_strike = round(current_price / interval) * interval

        if option_type == 'CE':
            itm_strike = atm_strike - (interval * self.itm_strikes)
            logger.info(f"CE ITM Strike: {itm_strike} (Current: {current_price}, ATM: {atm_strike})")
        else:  # PE
            itm_strike = atm_strike + (interval * self.itm_strikes)
            logger.info(f"PE ITM Strike: {itm_strike} (Current: {current_price}, ATM: {atm_strike})")

        return itm_strike

    def find_option_instrument(self, index_name, strike_price, option_type, expiry_date):
        """✅ FIXED: Find option instrument with proper return format"""
        if self.scrip_master_df is None:
            raise Exception("Scrip master not loaded. Please check connection.")

        try:
            expiry_str = expiry_date.strftime("%d%b%Y").upper()

            # Try exact strike first
            instrument = self._find_exact_instrument(index_name, strike_price, option_type, expiry_str)
            if instrument:
                return instrument

            # Fallback: Find nearest available strikes
            logger.warning(f"Exact strike {strike_price} not found, searching for nearest")
            interval = self.strike_intervals[index_name]

            for offset in [1, -1, 2, -2]:
                fallback_strike = strike_price + (offset * interval)
                instrument = self._find_exact_instrument(index_name, fallback_strike, option_type, expiry_str)
                if instrument:
                    logger.info(f"Using fallback strike {fallback_strike} instead of {strike_price}")
                    send_alert_email("Strike Fallback Used",
                                     f"Used {fallback_strike} instead of {strike_price} for {index_name} {option_type}")
                    return instrument

            # If still not found, log available strikes and fail
            available = self.scrip_master_df[
                (self.scrip_master_df['pSymbol'] == index_name) &
                (self.scrip_master_df['pOptionType'] == option_type) &
                (self.scrip_master_df['pExpiry'] == expiry_str)
                ]['pStrikePrice'].unique()

            error_msg = f"No suitable instrument found for {index_name} {strike_price} {option_type}. Available: {sorted(available.astype(float))}"
            logger.error(error_msg)
            send_alert_email("Instrument Not Found", error_msg)
            return None

        except Exception as e:
            logger.error(f"Error finding option instrument: {e}")
            return None

    def _find_exact_instrument(self, index_name, strike_price, option_type, expiry_str):
        """✅ FIXED: Return trading_symbol as primary identifier"""
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
            return None

        if len(matching_instruments) > 1:
            logger.warning(f"Multiple instruments found, using first one")

        instrument = matching_instruments.iloc[0]

        return {
            'trading_symbol': instrument['pTrdSymbol'],  # ✅ Primary identifier for place_order
            'instrument_token': instrument['pToken'],
            'symbol': instrument['pSymbol'],
            'strike_price': instrument['pStrikePrice'],
            'option_type': instrument['pOptionType'],
            'expiry': instrument['pExpiry'],
            'lot_size': int(instrument['pLotSize'])
        }

    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def verify_order_status(self, order_id):
        """Verify order status after placement"""
        try:
            if self.test_mode:
                return {'status': 'COMPLETE', 'qty': 25, 'message': 'Test order verified'}

            order_book = self.client.order_report()

            if 'data' in order_book:
                for order in order_book['data']:
                    if order.get('nOrdNo') == order_id:
                        status = order.get('ordSt', 'UNKNOWN')
                        qty = order.get('qty', 0)
                        message = order.get('rejRsn', 'No message')

                        return {
                            'status': status,
                            'qty': qty,
                            'message': message,
                            'order_details': order
                        }

            return {'status': 'NOT_FOUND', 'qty': 0, 'message': 'Order not found in order book'}

        except Exception as e:
            logger.error(f"Error verifying order status: {e}")
            raise

    @retry_with_backoff(max_retries=2, base_delay=0.3)
    def place_order(self, trading_symbol, quantity, order_type, transaction_type, price=None):
        """✅ FIXED: Place order with correct parameters - using trading_symbol"""
        try:
            if not is_market_open() and not self.test_mode:
                raise Exception("Market is closed. Orders can only be placed between 9:15 AM and 3:30 PM on weekdays.")

            # ✅ FIXED: Correct parameter structure as per Neo API docs
            order_params = {
                "trading_symbol": trading_symbol,  # ✅ Primary identifier (was instrument_token)
                "exchange_segment": "nse_fo",
                "product": "NRML",
                "price": str(price) if price else "0",
                "order_type": order_type,
                "quantity": str(quantity),
                "validity": "DAY",
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
                    'data': {'orderId': f'TEST_ORDER_{datetime.now().timestamp()}'},
                    'message': 'Test order placed successfully',
                    'params': order_params
                }

            # ✅ FIXED: Direct API call without manual session management
            response = self.client.place_order(**order_params)

            if 'data' in response and 'orderId' in response['data']:
                order_id = response['data']['orderId']
                logger.info(f"Order placed successfully: {response}")

                time.sleep(1)
                verification = self.verify_order_status(order_id)

                if verification['status'] in ['COMPLETE', 'PENDING']:
                    logger.info(f"Order verified: {verification['status']}")
                    response['verification'] = verification
                    return response
                else:
                    error_msg = f"Order verification failed: {verification['status']} - {verification['message']}"
                    logger.error(error_msg)
                    send_alert_email("Order Verification Failed",
                                     f"Order {order_id} for {trading_symbol}: {error_msg}")
                    raise Exception(error_msg)
            else:
                raise Exception(f"Order placement failed: {response}")

        except Exception as e:
            error_msg = f"Error placing order for {trading_symbol}: {e}"
            logger.error(error_msg)
            send_alert_email("Order Placement Failed", error_msg)
            raise

    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def get_existing_positions(self):
        """✅ FIXED: Handle 'No Data' response"""
        try:
            if self.test_mode:
                return []

            if not self.is_logged_in:
                raise Exception("Not logged in. Please run daily_auth.py")

            positions = self.client.positions()

            # ✅ FIX: Handle both success cases
            if 'data' in positions:
                return positions['data']
            elif positions.get('stCode') == 5203 and positions.get('errMsg') == 'No Data':
                logger.info("No positions found (empty portfolio)")
                return []  # Return empty list for no positions
            else:
                logger.warning(f"Unexpected positions response: {positions}")
                return []

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise

    def find_position_by_index_and_type(self, index_name, option_type):
        """Find existing position that matches the index and option type"""
        positions = self.get_existing_positions()

        for position in positions:
            symbol = position.get('trdSym', '')
            quantity = int(position.get('flQty', 0))

            if (index_name in symbol and
                    option_type in symbol and
                    quantity != 0):
                logger.info(f"Found existing position: {symbol}, Qty: {quantity}")
                return position

        logger.info(f"No existing {option_type} position found for {index_name}")
        return None

    def validate_trade(self, index_name, signal_type):
        """Validate if a trade should be placed"""
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
        """Execute a single trade based on timestamp, signal type"""
        if not self.is_logged_in and not self.test_mode:
            raise Exception("Not logged in. Please run daily_auth.py first")

        try:
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)

            if price is None:
                price = self.get_current_price(index_name)

            logger.info(f"Processing trade: {timestamp} - {signal_type} - {index_name} at {price}")

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
                itm_strike = self.get_itm_strike(index_name, price, 'CE')
                order_response = self._handle_buy_signal(index_name, itm_strike, expiry_date, 'CE')
                result.update({
                    'option_type': 'CE',
                    'strike': itm_strike,
                    'action': 'BUY_NEW'
                })

            elif signal_type == 'SELL':
                order_response = self._handle_sell_signal(index_name, 'CE')
                result.update({
                    'option_type': 'CE',
                    'action': 'SELL_EXISTING'
                })

            elif signal_type == 'SHORT':
                itm_strike = self.get_itm_strike(index_name, price, 'PE')
                order_response = self._handle_short_signal(index_name, itm_strike, expiry_date, 'PE')
                result.update({
                    'option_type': 'PE',
                    'strike': itm_strike,
                    'action': 'SHORT_NEW'
                })

            elif signal_type == 'COVER':
                order_response = self._handle_cover_signal(index_name, 'PE')
                result.update({
                    'option_type': 'PE',
                    'action': 'COVER_EXISTING'
                })

            else:
                raise ValueError(f"Invalid signal type: {signal_type}")

            result['order_response'] = order_response
            result['status'] = 'success' if order_response else 'no_action'

            logger.info(f"Trade executed successfully: {signal_type} - {result['status']}")
            return result

        except Exception as e:
            error_msg = f"Error executing trade: {str(e)}"
            logger.error(error_msg)

            return {
                'timestamp': timestamp,
                'signal_type': signal_type,
                'index_name': index_name,
                'price': price,
                'status': 'error',
                'error': error_msg,
                'order_response': None
            }

    def _handle_buy_signal(self, index_name, strike_price, expiry_date, option_type):
        """✅ FIXED: Handle BUY signal with correct parameters"""
        instrument = self.find_option_instrument(index_name, strike_price, option_type, expiry_date)

        if not instrument:
            raise Exception(f"Instrument not found for {index_name} {strike_price} {option_type}")

        # ✅ FIXED: Use trading_symbol for order placement
        response = self.place_order(
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

        trading_symbol = position.get('trdSym', '')
        quantity = abs(int(position.get('flQty', 0)))

        if quantity > 0:
            response = self.place_order(
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
        """✅ FIXED: Handle SHORT signal with correct parameters"""
        instrument = self.find_option_instrument(index_name, strike_price, option_type, expiry_date)

        if not instrument:
            raise Exception(f"Instrument not found for {index_name} {strike_price} {option_type}")

        # ✅ FIXED: Use trading_symbol for order placement
        response = self.place_order(
            trading_symbol=instrument['trading_symbol'],
            quantity=instrument['lot_size'],
            order_type="MKT",
            transaction_type="B"
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

        trading_symbol = position.get('trdSym', '')
        quantity = abs(int(position.get('flQty', 0)))

        if quantity > 0:
            response = self.place_order(
                trading_symbol=trading_symbol,
                quantity=quantity,
                order_type="MKT",
                transaction_type="S"
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

            logger.info("Testing API connectivity...")

            try:
                limits = self.client.limits()
                logger.info("✅ Limits API working")
            except Exception as e:
                logger.warning(f"Limits API issue: {e}")
                limits = {"error": str(e)}

            try:
                positions = self.client.positions()
                logger.info("✅ Positions API working")
                positions_count = len(positions.get('data', []))
            except Exception as e:
                logger.warning(f"Positions API issue: {e}")
                positions_count = -1

            return {
                "logged_in": True,
                "limits": limits,
                "positions_count": positions_count,
                "scrip_master_loaded": self.scrip_master_df is not None,
                "total_instruments": len(self.scrip_master_df) if self.scrip_master_df is not None else 0,
                "message": "Account active",
                "market_open": is_market_open()
            }
        except Exception as e:
            return {"logged_in": False, "error": str(e)}


# Example usage functions
def test_trader():
    """Test the trader without placing real orders"""
    trader = KotakOptionsTrader(test_mode=True)

    if not trader.is_logged_in:
        print("❌ Trader initialization failed")
        return

    print("\n🧪 TESTING TRADER FUNCTIONALITY")
    print("=" * 50)

    # Test different trade types
    test_signals = [
        ("BUY", "NIFTY"),
        ("SELL", "NIFTY"),
        ("SHORT", "BANKNIFTY"),
        ("COVER", "BANKNIFTY")
    ]

    for signal_type, index_name in test_signals:
        print(f"\n📊 Testing {signal_type} signal for {index_name}...")

        result = trader.execute_single_trade(
            timestamp=datetime.now(),
            index_name=index_name,
            signal_type=signal_type
        )

        print(f"   Status: {result['status']}")
        if result['status'] == 'error':
            print(f"   Error: {result.get('error')}")
        elif result['status'] == 'skipped':
            print(f"   Reason: {result.get('reason')}")
        else:
            print(f"   Price: {result.get('price')}")
            print(f"   Strike: {result.get('strike', 'N/A')}")


def live_trader_example():
    """Example of using the trader in live mode"""
    trader = KotakOptionsTrader(test_mode=False)

    if not trader.is_logged_in:
        print("❌ Not logged in. Please run daily_auth.py first")
        return

    status = trader.get_account_status()
    print(f"Account Status: {status}")

    if not is_market_open():
        print("Market is closed. Trading hours are 9:15 AM to 3:30 PM on weekdays.")
        return

    # Example trade execution
    try:
        result = trader.execute_single_trade(
            timestamp=datetime.now(),
            index_name="NIFTY",
            signal_type="BUY"
        )

        if result['status'] == 'success':
            print(f"Trade executed successfully!")
            print(f"Order ID: {result['order_response'].get('data', {}).get('orderId')}")
            print(f"Option: {result.get('option_type')} Strike: {result.get('strike')}")
        elif result['status'] == 'skipped':
            print(f"Trade skipped: {result['reason']}")
        else:
            print(f"Trade failed: {result.get('error')}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Default to test mode for safety
    test_trader()