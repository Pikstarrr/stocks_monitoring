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
import dhanhq

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


def _get_expiry_for_testing(month_offset):
    """Get expiry date for testing purposes"""
    today = datetime.now().date()
    target_month = today.month + month_offset
    target_year = today.year

    if target_month > 12:
        target_month -= 12
        target_year += 1

    return _get_last_thursday(target_year, target_month)


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
        """Load scrip master with proper data handling"""
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

            # Find NSE F&O file
            file_paths = scrip_response['filesPaths']
            nse_fo_url = None
            bse_fo_url = None

            for file_path in file_paths:
                if 'nse_fo.csv' in file_path:
                    nse_fo_url = file_path
                elif 'bse_fo.csv' in file_path:
                    bse_fo_url = file_path

            if not nse_fo_url:
                raise Exception("NSE F&O scrip master file not found")

            logger.info(f"Downloading scrip master from: {nse_fo_url}")

            # Download and parse CSV
            import requests
            response = requests.get(nse_fo_url, timeout=30)
            response.raise_for_status()

            # Parse CSV content
            from io import StringIO
            csv_content = StringIO(response.text)
            self.scrip_master_df = pd.read_csv(csv_content)

            # Rename columns
            column_rename = {
                'dStrikePrice;': 'pStrikePrice',
                'pExpiryDate': 'pExpiry',
                'lLotSize': 'pLotSize'
            }

            self.scrip_master_df.rename(columns=column_rename, inplace=True)

            # ✅ FIX: Handle strike price format (divide by 100)
            self.scrip_master_df['pStrikePrice'] = pd.to_numeric(self.scrip_master_df['pStrikePrice'],
                                                                 errors='coerce') / 100

            # ✅ FIX: Convert Unix timestamp to datetime
            self.scrip_master_df['pExpiry_dt'] = pd.to_datetime(self.scrip_master_df['pExpiry'], unit='s')
            self.scrip_master_df['pExpiry_str'] = self.scrip_master_df['pExpiry_dt'].dt.strftime('%d%b%Y').str.upper()

            # Clean lot size
            self.scrip_master_df['pLotSize'] = pd.to_numeric(self.scrip_master_df['pLotSize'], errors='coerce')

            # Filter for options only
            if 'pInstType' in self.scrip_master_df.columns:
                options_df = self.scrip_master_df[
                    self.scrip_master_df['pInstType'].str.upper().isin(['OPTIDX', 'OPTSTK'])
                ]
                logger.info(f"Found {len(options_df)} option instruments out of {len(self.scrip_master_df)} total")
                self.scrip_master_df = options_df

            if bse_fo_url:
                try:
                    logger.info(f"Downloading BSE scrip master from: {bse_fo_url}")

                    # Download and parse BSE CSV
                    response = requests.get(bse_fo_url, timeout=30)
                    response.raise_for_status()

                    # Parse BSE CSV content
                    from io import StringIO
                    csv_content = StringIO(response.text)
                    bse_df = pd.read_csv(csv_content)

                    # Apply same transformations as NSE data
                    column_rename = {
                        'dStrikePrice;': 'pStrikePrice',
                        'pExpiryDate': 'pExpiry',
                        'lLotSize': 'pLotSize'
                    }
                    bse_df.rename(columns=column_rename, inplace=True)

                    # Fix strike price and expiry
                    bse_df['pStrikePrice'] = pd.to_numeric(bse_df['pStrikePrice'], errors='coerce') / 100
                    bse_df['pExpiry_dt'] = pd.to_datetime(bse_df['pExpiry'], unit='s')
                    bse_df['pExpiry_str'] = bse_df['pExpiry_dt'].dt.strftime('%d%b%Y').str.upper()
                    bse_df['pLotSize'] = pd.to_numeric(bse_df['pLotSize'], errors='coerce')

                    # Filter BSE options
                    if 'pInstType' in bse_df.columns:
                        bse_options = bse_df[bse_df['pInstType'].str.upper().isin(['OPTIDX', 'OPTSTK'])]
                        logger.info(f"Found {len(bse_options)} BSE option instruments")

                        # Merge with NSE data
                        self.scrip_master_df = pd.concat([self.scrip_master_df, bse_options], ignore_index=True)
                        logger.info(f"Total instruments after merging: {len(self.scrip_master_df)}")

                except Exception as e:
                    logger.warning(f"Failed to load BSE scrip master: {e}")

            # ✅ Create index name mapping from trading symbols
            self._create_index_mapping()

            logger.info(f"Scrip master loaded successfully with {len(self.scrip_master_df)} option instruments")
            return True

        except Exception as e:
            logger.error(f"Error loading scrip master: {e}")
            if not self.test_mode:
                send_alert_email("Scrip Master Load Failed", f"Failed to load scrip master: {e}")
            raise

    def _create_index_mapping(self):
        """Create mapping of index names to their symbols"""
        self.index_symbol_map = {}

        # Extract index names from trading symbols
        for idx, row in self.scrip_master_df.iterrows():
            trd_symbol = row['pTrdSymbol']

            # Match index patterns
            if trd_symbol.startswith('NIFTY') and not any(
                    prefix in trd_symbol for prefix in ['BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']):
                self.index_symbol_map['NIFTY'] = row['pSymbol']
            elif trd_symbol.startswith('BANKNIFTY'):
                self.index_symbol_map['BANKNIFTY'] = row['pSymbol']
            elif trd_symbol.startswith('FINNIFTY'):
                self.index_symbol_map['FINNIFTY'] = row['pSymbol']
            elif trd_symbol.startswith('MIDCPNIFTY'):
                self.index_symbol_map['MIDCPNIFTY'] = row['pSymbol']
            elif trd_symbol.startswith('SENSEX'):
                self.index_symbol_map['SENSEX'] = row['pSymbol']

        logger.info(f"Index symbol mapping: {self.index_symbol_map}")

    def _create_mock_scrip_master(self):
        """Create mock scrip master with dynamic pricing"""
        mock_data = []
        indices = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX']

        # Get current and next month info
        from datetime import datetime
        current_date = datetime.now()

        # Try to get actual prices, fallback to defaults
        actual_prices = {}
        default_prices = {
            'NIFTY': 25500,
            'BANKNIFTY': 57000,
            'FINNIFTY': 24500,
            'MIDCPNIFTY': 12500,
            'SENSEX': 82000
        }

        # In test mode, we can't fetch real prices, so use the defaults
        # but you could enhance this to read from a config file
        for index in indices:
            actual_prices[index] = default_prices[index]

        for index in indices:
            base_price = actual_prices[index]
            interval = self.strike_intervals[index]
            lot_size = {
                'NIFTY': 25,
                'BANKNIFTY': 15,
                'FINNIFTY': 25,
                'MIDCPNIFTY': 50,
                'SENSEX': 10
            }[index]

            # Generate wider range of strikes
            strike_range = 25  # Generate 50 strikes total (25 above and below)

            for month_offset in [0, 1]:  # Current and next month
                target_month = current_date.month + month_offset
                target_year = current_date.year

                if target_month > 12:
                    target_month -= 12
                    target_year += 1

                expiry_date = _get_last_thursday(target_year, target_month)
                expiry_datetime = datetime.combine(expiry_date, datetime.min.time())

                year_short = expiry_date.strftime("%y")
                month_short = expiry_date.strftime("%b").upper()
                expiry_str = expiry_date.strftime("%d%b%Y").upper()
                expiry_timestamp = int(expiry_datetime.timestamp())

                # Round base price to nearest strike
                rounded_base = round(base_price / interval) * interval

                for i in range(-strike_range, strike_range + 1):
                    strike = rounded_base + (i * interval)

                    for opt_type in ['CE', 'PE']:
                        trading_symbol = f"{index}{year_short}{month_short}{int(strike)}{opt_type}"
                        symbol_id = f"{abs(hash(trading_symbol)) % 100000}"

                        mock_data.append({
                            'pSymbol': symbol_id,
                            'pGroup': 'EQ',
                            'pExchSeg': 'bse_fo' if index == 'SENSEX' else 'nse_fo',
                            'pInstType': 'OPTIDX',
                            'pSymbolName': index,
                            'pTrdSymbol': trading_symbol,
                            'pOptionType': opt_type,
                            'pScripRefKey': f'TEST_{symbol_id}',
                            'pISIN': '',
                            'pAssetCode': index,
                            'pExpiry': expiry_timestamp,
                            'pExpiry_dt': expiry_datetime,
                            'pExpiry_str': expiry_str,
                            'pStrikePrice': float(strike),
                            'pLotSize': lot_size,
                            'pExchange': 'NSE',
                            'pInstName': 'OPTIDX',
                            'pExpiryDate': expiry_str,
                            'pSegment': 'D',
                            'iPermittedToTrade': 1
                        })

        df = pd.DataFrame(mock_data)

        logger.info(f"Mock scrip master created with {len(df)} instruments")

        # Show strike ranges
        for index in indices:
            index_options = df[(df['pTrdSymbol'].str.startswith(index)) & (df['pOptionType'] == 'CE')]
            if not index_options.empty:
                strikes = sorted(index_options['pStrikePrice'].unique())
                logger.info(f"{index} CE strikes: {int(strikes[0])} to {int(strikes[-1])} ({len(strikes)} strikes)")

        return df

    @retry_with_backoff(max_retries=3, base_delay=0.2)
    # def get_current_price(self, index_name):
    #     """✅ FIXED: Get current market price with corrected quotes API usage"""
    #     try:
    #         if self.test_mode:
    #             mock_prices = {
    #                 'NIFTY': 24000,
    #                 'BANKNIFTY': 52000,
    #                 'FINNIFTY': 24500,
    #                 'MIDCPNIFTY': 12000,
    #                 'SENSEX': 80000
    #             }
    #             price = mock_prices.get(index_name, 24000)
    #             validate_price(price, index_name)
    #             return price
    #
    #         # ✅ FIXED: Use proper index symbol mapping
    #         index_symbol = self.index_symbols.get(index_name)
    #         if not index_symbol:
    #             raise ValueError(f"Index symbol not found for {index_name}")
    #
    #         # ✅ FIXED: Proper instrument tokens format for indices
    #         instrument_tokens = [{
    #             "instrument_token": index_symbol,
    #             "exchange_segment": "nse_cm"
    #         }]
    #
    #         # ✅ FIXED: No manual session parameter passing - client handles internally
    #         quotes = self.client.quotes(
    #             instrument_tokens=instrument_tokens,
    #             quote_type="ltp",
    #             isIndex=True
    #         )
    #
    #         if quotes and 'data' in quotes and len(quotes['data']) > 0:
    #             price = float(quotes['data'][0]['ltp'])
    #             validate_price(price, index_name)
    #             logger.info(f"Current price for {index_name}: {price}")
    #             return price
    #         else:
    #             raise Exception(f"No price data received for {index_name}")
    #
    #     except Exception as e:
    #         logger.error(f"Error getting current price for {index_name}: {e}")
    #         raise

    def get_current_price(self, index_name):
        symbol = ""

        if index_name == 'NIFTY':
            symbol = "13"
        elif index_name == 'BANKNIFTY':
            symbol = "25"
        elif index_name == 'FINNIFTY':
            symbol = "27"
        elif index_name == 'MIDCPNIFTY':
            symbol = "442"
        elif index_name == 'SENSEX':
            symbol = "51"

        try:

            load_dotenv()

            CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
            ACCESS_TOKEN = os.getenv("DHAN_API_KEY")

            # Initialize Dhan
            dhan_object = dhanhq.dhanhq(CLIENT_ID, ACCESS_TOKEN)

            quote_response = dhan_object.quote_data({"IDX_I": [int(symbol)]})
            if 'data' in quote_response and 'data' in quote_response['data'] and 'IDX_I' in quote_response['data'][
                'data']:
                quotes = quote_response['data']['data']['IDX_I']
                price = quotes.get(symbol).get('last_price')
                return price
            else:
                raise Exception(f"No price data received for {index_name}")
        except Exception as e:
            print(f"Error fetching quotes: {e}")
            raise Exception(f"No price data received for {index_name}")

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
        """Find option instrument - Updated to match trading symbol format"""
        if self.scrip_master_df is None:
            raise Exception("Scrip master not loaded. Please check connection.")

        try:
            # Format expiry for trading symbol: YYMMMDD format
            expiry_str_for_symbol = expiry_date.strftime("%y%b%d").upper()  # e.g., 25JUL31
            expiry_str_short = expiry_date.strftime("%y%b").upper()  # e.g., 25JUL

            logger.info(
                f"Searching for {index_name} {strike_price} {option_type} with expiry pattern {expiry_str_short}")

            # Try exact strike first
            instrument = self._find_exact_instrument(index_name, strike_price, option_type, expiry_str_short)
            if instrument:
                return instrument

            # If no exact match, try nearest strike
            logger.warning(f"Exact strike {strike_price} not found, searching for nearest")

            # Find nearest strikes
            interval = self.strike_intervals[index_name]
            for offset in [1, -1, 2, -2, 3, -3]:
                fallback_strike = strike_price + (offset * interval)
                instrument = self._find_exact_instrument(index_name, fallback_strike, option_type, expiry_str_short)
                if instrument:
                    logger.info(f"Using fallback strike {fallback_strike} instead of {strike_price}")
                    return instrument

            # If still nothing found, find the closest available
            instrument = self._find_closest_available_strike(index_name, strike_price, option_type, expiry_str_short)
            if instrument:
                return instrument

            error_msg = f"No suitable instrument found for {index_name} {strike_price} {option_type}"
            logger.error(error_msg)
            return None

        except Exception as e:
            logger.error(f"Error finding option instrument: {e}")
            logger.error(f"Stack trace: ", exc_info=True)
            return None

    def _get_index_symbol(self, index_name):
        """Find the actual symbol used in scrip master for the index"""
        # Check exact match first
        exact_match = self.scrip_master_df[
            (self.scrip_master_df['pSymbol'] == index_name) &
            (self.scrip_master_df['pInstType'] == 'OPTIDX')
            ]

        if not exact_match.empty:
            return index_name

        # Check for contains match
        contains_match = self.scrip_master_df[
            (self.scrip_master_df['pSymbol'].str.contains(index_name, case=False, na=False)) &
            (self.scrip_master_df['pInstType'] == 'OPTIDX')
            ]

        if not contains_match.empty:
            symbols = contains_match['pSymbol'].unique()
            # Return the shortest symbol (usually the main index)
            return min(symbols, key=len)

        # Check in trading symbol
        trd_match = self.scrip_master_df[
            (self.scrip_master_df['pTrdSymbol'].str.contains(index_name, case=False, na=False)) &
            (self.scrip_master_df['pInstType'] == 'OPTIDX')
            ]

        if not trd_match.empty:
            return trd_match.iloc[0]['pSymbol']

        logger.error(f"Could not find symbol for {index_name}")
        return None

    def _debug_available_options(self, index_name, option_type, expiry_str):
        """Debug available options with proper filtering"""
        logger.info(f"\n=== AVAILABLE OPTIONS DEBUG ===")

        # Filter for the specific index
        if index_name == 'NIFTY':
            index_options = self.scrip_master_df[
                (self.scrip_master_df['pTrdSymbol'].str.match(r'^NIFTY\d+', na=False)) &
                (~self.scrip_master_df['pTrdSymbol'].str.contains('BANKNIFTY|FINNIFTY|MIDCPNIFTY', na=False)) &
                (self.scrip_master_df['pInstType'] == 'OPTIDX')
                ]
        else:
            index_options = self.scrip_master_df[
                (self.scrip_master_df['pTrdSymbol'].str.startswith(index_name, na=False)) &
                (self.scrip_master_df['pInstType'] == 'OPTIDX')
                ]

        logger.info(f"Total {index_name} options: {len(index_options)}")

        # Filter by expiry
        expiry_options = index_options[index_options['pExpiry_str'] == expiry_str]
        logger.info(f"Options for expiry {expiry_str}: {len(expiry_options)}")

        # Filter by option type
        type_options = expiry_options[expiry_options['pOptionType'] == option_type]
        logger.info(f"{option_type} options: {len(type_options)}")

        if len(type_options) > 0:
            strikes = sorted(type_options['pStrikePrice'].unique())
            logger.info(f"Available strikes: {strikes[:20]}")

            # Show sample trading symbols
            logger.info("Sample trading symbols:")
            for _, row in type_options.head(5).iterrows():
                logger.info(f"  {row['pTrdSymbol']} - Strike: {row['pStrikePrice']}")

    def _find_closest_available_strike(self, index_name, target_strike, option_type, expiry_str_short):
        """Find the closest available strike when exact/nearby strikes not found"""
        # Get all options for this index/expiry/type
        if index_name == 'NIFTY':
            options = self.scrip_master_df[
                (self.scrip_master_df['pTrdSymbol'].str.contains(f'^NIFTY{expiry_str_short}', regex=True, na=False)) &
                (~self.scrip_master_df['pTrdSymbol'].str.contains('BANKNIFTY|FINNIFTY|MIDCPNIFTY', na=False)) &
                (self.scrip_master_df['pOptionType'] == option_type) &
                (self.scrip_master_df['pInstType'] == 'OPTIDX')
                ]
        else:
            options = self.scrip_master_df[
                (self.scrip_master_df['pTrdSymbol'].str.contains(f'^{index_name}{expiry_str_short}', regex=True,
                                                                 na=False)) &
                (self.scrip_master_df['pOptionType'] == option_type) &
                (self.scrip_master_df['pInstType'] == 'OPTIDX')
                ]

        if len(options) == 0:
            logger.error(f"No options found for {index_name} {expiry_str_short} {option_type}")
            return None

        # Find the closest strike
        options = options.copy()
        options['strike_diff'] = abs(options['pStrikePrice'] - float(target_strike))
        closest = options.nsmallest(1, 'strike_diff')

        if closest.empty:
            return None

        instrument = closest.iloc[0]
        closest_strike = instrument['pStrikePrice']

        logger.warning(
            f"Using closest available strike {closest_strike} instead of {target_strike} (diff: {abs(closest_strike - target_strike)})")

        return {
            'trading_symbol': instrument['pTrdSymbol'],
            'instrument_token': instrument.get('pScripRefKey', instrument.get('pSymbol', '')),
            'symbol': instrument['pSymbol'],
            'strike_price': instrument['pStrikePrice'],
            'option_type': instrument['pOptionType'],
            'expiry': instrument['pExpiry_str'],
            'lot_size': int(instrument['pLotSize'])
        }

    def debug_index_data(self, index_name):
        """Debug data for a specific index - FIXED version"""
        print(f"\n=== DEBUG: {index_name} OPTIONS ===")

        # For NIFTY, we need to exclude MIDCPNIFTY, FINNIFTY, etc.
        # Look for exact index name at the start of trading symbol
        if index_name == 'NIFTY':
            # Match NIFTY but not MIDCPNIFTY, FINNIFTY, BANKNIFTY
            mask = (
                    self.scrip_master_df['pTrdSymbol'].str.match(r'^NIFTY\d+', na=False) &
                    (self.scrip_master_df['pInstType'] == 'OPTIDX')
            )
        else:
            # For other indices, match at the start
            mask = (
                    self.scrip_master_df['pTrdSymbol'].str.startswith(index_name, na=False) &
                    (self.scrip_master_df['pInstType'] == 'OPTIDX')
            )

        index_data = self.scrip_master_df[mask]

        print(f"Found {len(index_data)} {index_name} options")

        if len(index_data) > 0:
            # Show unique symbols
            print(f"\nUnique pSymbol values (first 10): {index_data['pSymbol'].unique()[:10]}")

            # Show sample data
            sample = index_data.iloc[0]
            print(f"\nSample option:")
            print(f"  pSymbol: {sample['pSymbol']}")
            print(f"  pTrdSymbol: {sample['pTrdSymbol']}")
            print(f"  pExpiry (timestamp): {sample['pExpiry']}")

            # Convert timestamp to readable date
            from datetime import datetime
            expiry_date = datetime.fromtimestamp(sample['pExpiry'])
            print(f"  pExpiry (date): {expiry_date}")

            print(f"  pStrikePrice: {sample['pStrikePrice']} (actual: {sample['pStrikePrice'] / 100})")
            print(f"  pOptionType: {sample['pOptionType']}")
            print(f"  pExchSeg: {sample['pExchSeg']}")

            # Show more trading symbols
            print(f"\nSample trading symbols:")
            for trd_sym in index_data['pTrdSymbol'].head(10):
                print(f"  {trd_sym}")

    def test_current_month_options(self):
        """Test what options are available for current month"""
        from datetime import datetime

        current_date = datetime.now()
        current_month_str = current_date.strftime("%y%b").upper()  # e.g., "25JUL"

        print(f"\n=== CURRENT MONTH OPTIONS TEST ===")
        print(f"Looking for options with: {current_month_str}")

        # Filter for current month NIFTY options
        current_options = self.scrip_master_df[
            (self.scrip_master_df['pTrdSymbol'].str.contains(f'NIFTY{current_month_str}', na=False)) &
            (~self.scrip_master_df['pTrdSymbol'].str.contains('BANKNIFTY|FINNIFTY|MIDCPNIFTY', na=False)) &
            (self.scrip_master_df['pInstType'] == 'OPTIDX')
            ]

        print(f"Found {len(current_options)} options for {current_month_str}")

        if len(current_options) > 0:
            # Show strikes
            ce_options = current_options[current_options['pOptionType'] == 'CE']
            strikes = sorted(ce_options['pStrikePrice'].unique())
            print(f"\nAvailable CE strikes: {strikes[:20]}")

            # Show sample
            print(f"\nSample current month options:")
            for _, row in current_options.head(5).iterrows():
                print(f"  {row['pTrdSymbol']} - Strike: {row['pStrikePrice']} - Expiry: {row['pExpiry_str']}")

        # Also check next month
        next_month = (current_date.month % 12) + 1
        next_year = current_date.year + (1 if current_date.month == 12 else 0)
        next_month_date = datetime(next_year, next_month, 1)
        next_month_str = next_month_date.strftime("%y%b").upper()

        print(f"\nChecking next month: {next_month_str}")
        next_options = self.scrip_master_df[
            (self.scrip_master_df['pTrdSymbol'].str.contains(f'NIFTY{next_month_str}', na=False)) &
            (~self.scrip_master_df['pTrdSymbol'].str.contains('BANKNIFTY|FINNIFTY|MIDCPNIFTY', na=False))
            ]
        print(f"Found {len(next_options)} options for {next_month_str}")

    def _find_exact_instrument(self, index_name, strike_price, option_type, expiry_str_short):
        """Find exact instrument using trading symbol pattern"""
        # Build search pattern - the trading symbol format is NIFTY25JUL22250CE
        search_pattern = f'{index_name}{expiry_str_short}{int(strike_price)}{option_type}'

        logger.info(f"Looking for trading symbol pattern: {search_pattern}")

        # Search by exact trading symbol
        matching_instruments = self.scrip_master_df[
            (self.scrip_master_df['pTrdSymbol'] == search_pattern) &
            (self.scrip_master_df['pInstType'] == 'OPTIDX') &
            (self.scrip_master_df['pExchSeg'].isin(['nse_fo', 'NSE_FO', 'bse_fo', 'BSE_FO']))
            ]

        if len(matching_instruments) == 0:
            # Log what we have for debugging
            if self.test_mode:
                available = self.scrip_master_df[
                    (self.scrip_master_df['pTrdSymbol'].str.contains(index_name, na=False)) &
                    (self.scrip_master_df['pTrdSymbol'].str.contains(expiry_str_short, na=False)) &
                    (self.scrip_master_df['pOptionType'] == option_type)
                    ]
                if not available.empty:
                    logger.info(f"Available similar options:")
                    for _, row in available.head(3).iterrows():
                        logger.info(f"  {row['pTrdSymbol']} - Strike: {row['pStrikePrice']}")

            # Try with contains for more flexible matching
            matching_instruments = self.scrip_master_df[
                (self.scrip_master_df['pTrdSymbol'].str.contains(
                    f'^{index_name}{expiry_str_short}.*{int(strike_price)}{option_type}$', regex=True, na=False)) &
                (self.scrip_master_df['pStrikePrice'] == float(strike_price)) &
                (self.scrip_master_df['pOptionType'] == option_type) &
                (self.scrip_master_df['pInstType'] == 'OPTIDX')
                ]

            # For NIFTY, exclude variants
            if index_name == 'NIFTY':
                matching_instruments = matching_instruments[
                    ~matching_instruments['pTrdSymbol'].str.contains('BANKNIFTY|FINNIFTY|MIDCPNIFTY', na=False)
                ]

        if len(matching_instruments) == 0:
            return None

        if len(matching_instruments) > 1:
            logger.warning(f"Multiple instruments found ({len(matching_instruments)}), using first one")

        instrument = matching_instruments.iloc[0]

        return {
            'trading_symbol': instrument['pTrdSymbol'],
            'instrument_token': instrument.get('pScripRefKey', instrument.get('pSymbol', '')),
            'symbol': instrument['pSymbol'],
            'strike_price': instrument['pStrikePrice'],
            'option_type': instrument['pOptionType'],
            'expiry': instrument['pExpiry_str'],
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
        """Place order with proper error handling and verification"""
        try:
            if not is_market_open() and not self.test_mode:
                raise Exception("Market is closed. Orders can only be placed between 9:15 AM and 3:30 PM on weekdays.")

            # ✅ NEW: Check if order already exists before placing
            if self._is_order_already_placed(trading_symbol, transaction_type):
                logger.warning(f"Order already exists for {trading_symbol} {transaction_type}, skipping")
                # Return a success response since the order is already there
                return {
                    'status': 'success',
                    'data': {'orderId': 'EXISTING_ORDER'},
                    'message': 'Order already exists',
                    'duplicate': True
                }

            import random
            unique_id = f"{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}"

            exchange_segment = "bse_fo" if "SENSEX" in trading_symbol else "nse_fo"

            order_params = {
                "trading_symbol": trading_symbol,
                "exchange_segment": exchange_segment,
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
                "tag": f"auto_{unique_id}"  # ✅ Unique tag for each order
            }

            if self.test_mode:
                logger.info(f"TEST MODE - Would place order: {order_params}")
                return {
                    'status': 'success',
                    'data': {'orderId': f'TEST_ORDER_{datetime.now().timestamp()}'},
                    'message': 'Test order placed successfully',
                    'params': order_params
                }

            # Place the order
            response = self.client.place_order(**order_params)

            # Check initial response
            if response and ('nOrdNo' in response or (response.get('stat') == 'Ok' and response.get('stCode') == 200)):
                order_id = response.get('nOrdNo', response.get('data', {}).get('orderId'))

                if order_id:
                    logger.info(f"Order submitted with ID: {order_id}")

                    # Wait and verify order status
                    time.sleep(2)
                    verification = self.verify_order_status(order_id)

                    # Check if order was rejected
                    if verification['status'] in ['REJECTED', 'CANCELLED', 'REJ']:
                        reject_reason = verification.get('message', 'Unknown reason')

                        # Check for common rejection reasons
                        if any(term in reject_reason.lower() for term in
                               ['insufficient', 'margin', 'funds', 'balance']):
                            raise Exception(f"Order rejected due to insufficient funds: {reject_reason}")
                        else:
                            raise Exception(f"Order rejected: {reject_reason}")

                    elif verification['status'] in ['COMPLETE', 'EXECUTED', 'TRADED']:
                        logger.info(f"Order {order_id} executed successfully")
                        return {
                            'status': 'success',
                            'data': {'orderId': order_id},
                            'message': 'Order executed successfully',
                            'response': response,
                            'verification': verification
                        }

                    elif verification['status'] in ['PENDING', 'OPEN', 'TRIGGER_PENDING']:
                        logger.info(f"Order {order_id} is pending execution")
                        return {
                            'status': 'success',
                            'data': {'orderId': order_id},
                            'message': 'Order placed and pending execution',
                            'response': response,
                            'verification': verification
                        }
                    else:
                        logger.warning(f"Unknown order status: {verification['status']}")
                        return {
                            'status': 'success',
                            'data': {'orderId': order_id},
                            'message': f'Order placed with status: {verification["status"]}',
                            'response': response,
                            'verification': verification
                        }

            # Order placement failed
            error_msg = response.get('errMsg', str(response))

            # Check for duplicate order error
            if 'already exists' in error_msg.lower() or 'orderid' in error_msg.lower():
                logger.error(f"Order ID conflict: {error_msg}")
                # Wait a bit and generate new ID on retry
                time.sleep(1)
                # Don't raise exception, return success
                return {
                    'status': 'success',
                    'data': {'orderId': 'DUPLICATE'},
                    'message': 'Order already exists',
                    'duplicate': True,
                    'response': response
                }

            raise Exception(f"Order placement failed: {error_msg}")

        except Exception as e:
            error_msg = str(e)

            # Don't retry for certain errors
            if any(term in error_msg.lower() for term in ['already exists', 'insufficient', 'margin', 'funds']):
                # These errors shouldn't be retried
                record_failure()
                # Re-raise without going through retry mechanism
                raise e from None

            # For other errors, let retry mechanism handle it
            logger.error(f"Error placing order for {trading_symbol}: {e}")
            send_alert_email("Order Placement Failed", f"Error placing order for {trading_symbol}: {e}")
            raise

    def _is_order_already_placed(self, trading_symbol, transaction_type):
        """Check if an order for this symbol is already placed"""
        try:
            if self.test_mode:
                return False

            order_book = self.client.order_report()
            orders = order_book if isinstance(order_book, list) else order_book.get('data', [])

            for order in orders:
                order_symbol = order.get('trdSym', '')
                order_trans_type = order.get('trnsTp', '')
                order_status = order.get('stat', order.get('ordSt', '')).upper()

                # Check for rejection reason if needed
                rejection_reason = order.get('rejRsn', '')

                if (order_symbol == trading_symbol and
                        order_trans_type == transaction_type and
                        order_status not in ['REJECTED', 'CANCELLED', 'REJ']):

                    order_id = order.get('nOrdNo', order.get('GuiOrdId', 'Unknown'))
                    logger.info(
                        f"Found existing order: {order_id} for {trading_symbol} {transaction_type} with status {order_status}")
                    return True

                # Optional: Log rejected orders for debugging
                elif order_status == 'REJECTED' and rejection_reason:
                    logger.debug(f"Found rejected order for {trading_symbol}: {rejection_reason}")

            return False

        except Exception as e:
            logger.error(f"Error checking existing orders: {e}")
            return False

    def check_available_margin(self):
        """Check available margin before placing orders"""
        try:
            if self.test_mode:
                return {'available_margin': 1000000, 'used_margin': 0}

            limits = self.client.limits()

            # ✅ FIXED: Handle the actual response structure
            if limits and limits.get('stat') == 'Ok':
                # Extract margin info from flat structure
                available = float(limits.get('Net', '0'))
                used = float(limits.get('MarginUsed', '0'))
                collateral = float(limits.get('CollateralValue', '0'))

                logger.info(
                    f"Margin check - Available: ₹{available:,.2f}, Used: ₹{used:,.2f}, Collateral: ₹{collateral:,.2f}")

                return {
                    'available_margin': available,
                    'used_margin': used,
                    'collateral_value': collateral,
                    'raw_response': limits
                }
            else:
                logger.error(f"Failed to get margin info: {limits}")
                return {'available_margin': 0, 'used_margin': 0}

        except Exception as e:
            logger.error(f"Error checking margin: {e}")
            return {'available_margin': 0, 'used_margin': 0}

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
            symbol = position.get('sym', '')
            opt_tp = position.get('optTp', '')
            buy_quantity = int(position.get('flBuyQty', 0))
            sell_quantity = int(position.get('flSellQty', 0))

            if (index_name in symbol and
                    option_type in opt_tp and
                    buy_quantity != 0 and sell_quantity == 0):
                logger.info(f"Found existing position: {symbol}, Qty: {buy_quantity}")
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
        """Execute a single trade with proper margin check"""
        if not self.is_logged_in and not self.test_mode:
            raise Exception("Not logged in. Please run daily_auth.py first")

        try:
            # ✅ ENHANCED: Better margin check before BUY/SHORT orders
            if signal_type in ['BUY', 'SHORT'] and not self.test_mode:
                margin_info = self.check_available_margin()
                available_margin = margin_info.get('available_margin', 0)

                # Estimate required margin (rough estimate - 15-20% of contract value)
                if price:
                    lot_size = {'NIFTY': 25, 'BANKNIFTY': 15, 'FINNIFTY': 25,
                                'MIDCPNIFTY': 50, 'SENSEX': 10}.get(index_name, 25)
                    contract_value = price * lot_size
                    estimated_margin = contract_value * 0.15  # 15% margin requirement estimate

                    logger.info(f"Estimated margin required: ₹{estimated_margin:,.2f} for {index_name}")

                    if available_margin < estimated_margin:
                        error_msg = f"Insufficient margin. Available: ₹{available_margin:,.2f}, Required (est): ₹{estimated_margin:,.2f}"
                        logger.error(error_msg)
                        send_alert_email("Insufficient Margin", error_msg)

                        return {
                            'timestamp': timestamp,
                            'signal_type': signal_type,
                            'index_name': index_name,
                            'status': 'error',
                            'error': error_msg,
                            'margin_info': margin_info,
                            'order_response': None
                        }

                if available_margin < 5000:  # Minimum threshold
                    error_msg = f"Margin too low. Available: ₹{available_margin:,.2f}"
                    logger.error(error_msg)
                    return {
                        'timestamp': timestamp,
                        'signal_type': signal_type,
                        'index_name': index_name,
                        'status': 'error',
                        'error': error_msg,
                        'margin_info': margin_info,
                        'order_response': None
                    }

            # Continue with existing trade logic...
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
        quantity = abs(int(position.get('flBuyQty', 0)))

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
        quantity = abs(int(position.get('flBuyQty', 0)))

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
