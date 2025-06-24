import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
from neo_api_client import NeoAPI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KotakOptionsTrader:
    def __init__(self, consumer_key, consumer_secret, mobile_number=None, password=None,
                 access_token=None, environment='prod'):
        """
        Initialize Kotak Neo API client with multiple authentication options

        Args:
            consumer_key: Your consumer key
            consumer_secret: Your consumer secret
            mobile_number: Your mobile number (for traditional login)
            password: Your password (for traditional login)
            access_token: Pre-generated access token for automated trading
            environment: 'prod' or 'uat'
        """
        # self.client = NeoAPI(consumer_key=consumer_key,
        #                      consumer_secret=consumer_secret,
        #                      environment=environment)

        self.client = NeoAPI(access_token=access_token)
        self.mobile_number = mobile_number
        self.password = password
        self.stored_access_token = access_token
        self.is_logged_in = False

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

    def login(self, access_token=None, otp=None):
        """
        Login to Kotak Neo API with multiple methods

        Args:
            access_token: Pre-generated access token for automated login
            otp: OTP for 2FA (if using mobile/password login)
        """
        try:
            if access_token:
                # Use access token for automated login
                self.is_logged_in = True
                logger.info("Successfully logged in using access token")
            else:
                # Traditional mobile/password login
                self.client.login(mobilenumber=self.mobile_number, password=self.password)
                if otp:
                    self.client.session_2fa(otp)
                self.is_logged_in = True
                logger.info("Successfully logged in using mobile/password")
        except Exception as e:
            logger.error(f"Login failed: {e}")
            raise

    def auto_login(self):
        """
        Automated login using stored credentials or token
        This method attempts multiple login strategies for automation
        """
        try:
            # Method 1: Try using stored access token (if available)
            if hasattr(self, 'stored_access_token') and self.stored_access_token:
                self.login(access_token=self.stored_access_token)
                return True

            # Method 2: Use environment variables or config file
            import os
            access_token = os.getenv('KOTAK_ACCESS_TOKEN')
            if access_token:
                self.login(access_token=access_token)
                return True

            # Method 3: For automation, you might want to implement
            # a token refresh mechanism or use API keys if available
            logger.warning("No automated login method available. Manual login required.")
            return False

        except Exception as e:
            logger.error(f"Auto login failed: {e}")
            return False

    def get_expiry_date(self, index_name):
        """
        Get the appropriate expiry date based on current date
        Returns current month expiry if >10 days remaining, else next month
        """
        today = datetime.now().date()

        # Get current month's last Thursday
        current_month_expiry = self._get_last_thursday(today.year, today.month)

        # Calculate days remaining
        days_remaining = (current_month_expiry - today).days

        if days_remaining > 10:
            return current_month_expiry
        else:
            # Get next month's expiry
            next_month = today.month + 1 if today.month < 12 else 1
            next_year = today.year if today.month < 12 else today.year + 1
            return self._get_last_thursday(next_year, next_month)

    def _get_last_thursday(self, year, month):
        """Get the last Thursday of the given month"""
        # Get the last day of the month
        last_day = calendar.monthrange(year, month)[1]
        last_date = datetime(year, month, last_day).date()

        # Find the last Thursday
        days_to_subtract = (last_date.weekday() - 3) % 7
        last_thursday = last_date - timedelta(days=days_to_subtract)

        return last_thursday

    def get_atm_strike(self, index_name, current_price):
        """Get the ATM strike price for the given index"""
        interval = self.strike_intervals[index_name]
        atm_strike = round(current_price / interval) * interval
        return atm_strike

    def get_current_price(self, symbol):
        """Get current market price for the symbol"""
        try:
            # Get quotes for the symbol
            quotes = self.client.quotes(instrument_tokens=[symbol],
                                        quote_type="ltp",
                                        isIndex=True)
            return float(quotes['data'][0]['ltp'])
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            raise

    def get_option_symbol(self, index_name, strike_price, option_type, expiry_date):
        """
        Generate option symbol based on index, strike, type and expiry
        Format: NIFTY24JUN24000CE or NIFTY24JUN24000PE
        """
        expiry_str = expiry_date.strftime("%d%b%y").upper()
        option_symbol = f"{index_name}{expiry_str}{int(strike_price)}{option_type}"
        return option_symbol

    def get_existing_positions(self):
        """Get all existing positions from Kotak"""
        try:
            if not self.is_logged_in:
                raise Exception("Not logged in. Please login first.")

            positions = self.client.positions()
            return positions.get('data', [])
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def find_position_by_symbol_pattern(self, index_name, option_type):
        """
        Find existing position that matches the index and option type pattern
        """
        positions = self.get_existing_positions()

        for position in positions:
            symbol = position.get('trdSym', '')
            quantity = int(position.get('flQty', 0))

            # Check if this is an option of the specified index and type
            if (index_name in symbol and
                    option_type in symbol and
                    quantity != 0):
                return position

        return None

    def place_order(self, symbol, quantity, order_type, transaction_type, price=None):
        """
        Place order using Kotak Neo API

        Args:
            symbol: Trading symbol
            quantity: Quantity to trade
            order_type: 'MKT' for market, 'LMT' for limit
            transaction_type: 'B' for buy, 'S' for sell
            price: Price for limit orders
        """
        try:
            order_params = {
                "instrument_token": symbol,
                "exchange_segment": "nse_fo",  # NSE F&O
                "product": "NRML",  # Normal
                "price": str(price) if price else "0",
                "order_type": order_type,
                "quantity": str(quantity),
                "validity": "DAY",
                "trading_symbol": symbol,
                "transaction_type": transaction_type,
                "amo": "NO",
                "disclosed_quantity": "0",
                "market_protection": "0",
                "pf": "N",
                "trigger_price": "0",
                "tag": "string"
            }

            response = self.client.place_order(**order_params)
            logger.info(f"Order placed successfully: {response}")
            return response

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise

    def execute_single_trade(self, timestamp, signal_type, price, index_name='NIFTY'):
        """
        Execute a single trade based on timestamp, signal type, and price

        Args:
            timestamp: Trading signal timestamp (pandas Timestamp or datetime)
            signal_type: 'BUY', 'SELL', 'SHORT', 'COVER'
            price: Current index price
            index_name: Index to trade ('NIFTY', 'BANKNIFTY', etc.)

        Returns:
            dict: Trade execution result
        """

        # Ensure we're logged in
        if not self.is_logged_in:
            if not self.auto_login():
                raise Exception("Failed to login automatically. Please login manually.")

        try:
            # Convert timestamp if needed
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)

            logger.info(f"Processing trade: {timestamp} - {signal_type} - {index_name} at {price}")

            # Get expiry date
            expiry_date = self.get_expiry_date(index_name)

            # Get ATM strike
            atm_strike = self.get_atm_strike(index_name, price)

            result = {
                'timestamp': timestamp,
                'signal_type': signal_type,
                'index_name': index_name,
                'price': price,
                'atm_strike': atm_strike,
                'expiry_date': expiry_date,
                'status': 'pending',
                'order_response': None,
                'error': None
            }

            if signal_type == 'BUY':
                # BUY = CE (Call Option)
                order_response = self._handle_buy_signal(index_name, atm_strike, expiry_date, 'CE')
                result['option_type'] = 'CE'
                result['action'] = 'BUY_NEW'

            elif signal_type == 'SELL':
                # SELL existing CE position
                order_response = self._handle_sell_signal(index_name, 'CE')
                result['option_type'] = 'CE'
                result['action'] = 'SELL_EXISTING'

            elif signal_type == 'SHORT':
                # SHORT = PE (Put Option)
                order_response = self._handle_short_signal(index_name, atm_strike, expiry_date, 'PE')
                result['option_type'] = 'PE'
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
        option_symbol = self.get_option_symbol(index_name, strike_price, option_type, expiry_date)

        # Place buy order for 1 lot (quantity varies by index)
        lot_size = self._get_lot_size(index_name)

        response = self.place_order(
            symbol=option_symbol,
            quantity=lot_size,
            order_type="MKT",
            transaction_type="B"
        )

        logger.info(f"BUY order placed for {option_symbol}, Quantity: {lot_size}")
        return response

    def _handle_sell_signal(self, index_name, option_type):
        """Handle SELL signal - Sell existing CE position"""
        position = self.find_position_by_symbol_pattern(index_name, option_type)

        if not position:
            logger.warning(f"No existing {option_type} position found for {index_name}")
            return None

        symbol = position['trdSym']
        quantity = abs(int(position['flQty']))  # Use absolute value

        if quantity > 0:
            response = self.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type="MKT",
                transaction_type="S"
            )

            logger.info(f"SELL order placed for {symbol}, Quantity: {quantity}")
            return response
        else:
            logger.warning(f"No quantity to sell for {symbol}")
            return None

    def _handle_short_signal(self, index_name, strike_price, expiry_date, option_type):
        """Handle SHORT signal - Sell PE option"""
        option_symbol = self.get_option_symbol(index_name, strike_price, option_type, expiry_date)

        # Place sell order for 1 lot
        lot_size = self._get_lot_size(index_name)

        response = self.place_order(
            symbol=option_symbol,
            quantity=lot_size,
            order_type="MKT",
            transaction_type="S"
        )

        logger.info(f"SHORT order placed for {option_symbol}, Quantity: {lot_size}")
        return response

    def _handle_cover_signal(self, index_name, option_type):
        """Handle COVER signal - Buy back existing PE short position"""
        position = self.find_position_by_symbol_pattern(index_name, option_type)

        if not position:
            logger.warning(f"No existing {option_type} position found for {index_name}")
            return None

        symbol = position['trdSym']
        quantity = abs(int(position['flQty']))  # Use absolute value

        if quantity > 0:
            response = self.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type="MKT",
                transaction_type="B"
            )

            logger.info(f"COVER order placed for {symbol}, Quantity: {quantity}")
            return response
        else:
            logger.warning(f"No quantity to cover for {symbol}")
            return None

    def _get_lot_size(self, index_name):
        """Get lot size for each index"""
        lot_sizes = {
            'NIFTY': 25,
            'BANKNIFTY': 15,
            'FINNIFTY': 25,
            'MIDCPNIFTY': 50,
            'SENSEX': 10
        }
        return lot_sizes.get(index_name, 25)  # Default to 25 if not found

    def get_account_status(self):
        """Get account and login status"""
        try:
            if not self.is_logged_in:
                return {"logged_in": False, "message": "Not logged in"}

            # Get account details if available
            limits = self.client.limits()
            positions = self.client.positions()

            return {
                "logged_in": True,
                "limits": limits,
                "positions_count": len(positions.get('data', [])),
                "message": "Account active"
            }
        except Exception as e:
            return {"logged_in": False, "error": str(e)}


# Example usage for automated trading
def automated_trading_example():
    """
    Example of how to use the trader for automated single trade execution
    """
    # Method 1: Using access token for full automation
    trader = KotakOptionsTrader(
        consumer_key="your_consumer_key",
        consumer_secret="your_consumer_secret",
        access_token="your_access_token"  # For automation
    )

    # Method 2: Using environment variables
    # Set KOTAK_ACCESS_TOKEN in your environment
    trader_env = KotakOptionsTrader(
        consumer_key="your_consumer_key",
        consumer_secret="your_consumer_secret"
    )

    # Execute single trades
    try:
        # Example 1: BUY signal
        result1 = trader.execute_single_trade(
            timestamp="2025-06-24 09:15:00",
            signal_type="BUY",
            price=25000.50,
            index_name="NIFTY"
        )
        print(f"BUY Result: {result1['status']} - {result1.get('error', 'Success')}")

        # Example 2: SELL signal
        result2 = trader.execute_single_trade(
            timestamp="2025-06-24 14:30:00",
            signal_type="SELL",
            price=25150.75,
            index_name="NIFTY"
        )
        print(f"SELL Result: {result2['status']} - {result2.get('error', 'Success')}")

        # Example 3: SHORT signal
        result3 = trader.execute_single_trade(
            timestamp="2025-06-24 10:45:00",
            signal_type="SHORT",
            price=24980.25,
            index_name="BANKNIFTY"
        )
        print(f"SHORT Result: {result3['status']} - {result3.get('error', 'Success')}")

        # Example 4: COVER signal
        result4 = trader.execute_single_trade(
            timestamp="2025-06-24 15:20:00",
            signal_type="COVER",
            price=25020.80,
            index_name="BANKNIFTY"
        )
        print(f"COVER Result: {result4['status']} - {result4.get('error', 'Success')}")

    except Exception as e:
        print(f"Trading Error: {e}")