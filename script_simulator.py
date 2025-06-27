import pandas as pd
from datetime import datetime, timedelta
import numpy as np


class SlidingWindowSimulator:
    """
    Simulates real-time data streaming by maintaining a sliding window of historical data
    and feeding new data points one by one, just like the real fetch_ohlc function
    """

    def __init__(self, csv_file_path, window_months=6):
        """
        Initialize the simulator

        Args:
            csv_file_path (str): Path to CSV file with year data
            window_months (int): Number of months to keep in sliding window
        """
        self.csv_file_path = csv_file_path
        self.window_months = window_months
        self.full_data = None
        self.current_index = 0
        self.window_size_days = window_months * 30  # Approximate

        # Load and prepare data
        self._load_data()

    def _load_data(self):
        """Load and prepare the CSV data"""
        # Load CSV - adjust column names as needed
        df = pd.read_csv(self.csv_file_path)

        # Assuming CSV has columns: datetime, open, high, low, close
        # Adjust column names if different
        expected_columns = ['datetime', 'open', 'high', 'low', 'close']

        # Handle different possible column names
        if 'timestamp' in df.columns and 'datetime' not in df.columns:
            df = df.rename(columns={'timestamp': 'datetime'})

        # Convert datetime column
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.sort_index()

        # Keep only OHLC columns
        self.full_data = df[['open', 'high', 'low', 'close']].astype(float)

        # Calculate initial window size (6 months of data)
        total_rows = len(self.full_data)
        self.window_size_rows = min(int(total_rows * 0.5), total_rows - 100)  # Leave 100 rows for simulation
        self.current_index = self.window_size_rows

        print(f"📊 Loaded {total_rows} rows from CSV")
        print(f"📊 Window size: {self.window_size_rows} rows ({self.window_months} months)")
        print(f"📊 Available for simulation: {total_rows - self.window_size_rows} new data points")

    def get_current_data(self):
        """
        Get current sliding window data (simulates fetch_ohlc)

        Returns:
            pd.DataFrame: Current window of OHLC data with datetime index
        """
        if self.current_index >= len(self.full_data):
            print("⚠️ No more data available for simulation")
            return None

        # Get sliding window: from (current_index - window_size) to current_index
        start_idx = max(0, self.current_index - self.window_size_rows)
        end_idx = self.current_index + 1

        current_window = self.full_data.iloc[start_idx:end_idx].copy()

        # Simulate the "live candle append" behavior from your original code
        # The last row represents the "current" candle
        latest_timestamp = current_window.index[-1]
        latest_data = current_window.iloc[-1]

        print(f"📈 Sliding window: {len(current_window)} rows, Latest: {latest_timestamp}")

        return current_window

    def advance_time(self):
        """
        Advance to next data point (simulates time passing)

        Returns:
            bool: True if advanced successfully, False if no more data
        """
        if self.current_index >= len(self.full_data) - 1:
            return False

        self.current_index += 1
        return True

    def reset(self):
        """Reset simulator to beginning"""
        self.current_index = self.window_size_rows

    def get_simulation_progress(self):
        """Get current simulation progress"""
        total_simulation_points = len(self.full_data) - self.window_size_rows
        current_point = self.current_index - self.window_size_rows + 1
        progress_pct = (current_point / total_simulation_points) * 100

        return {
            'current_point': current_point,
            'total_points': total_simulation_points,
            'progress_pct': round(progress_pct, 2),
            'remaining_points': total_simulation_points - current_point
        }


def simulate_realtime_trading(csv_file_path, window_months=6, max_iterations=None):
    """
    Main simulation function that replaces your fetch_ohlc calls

    Args:
        csv_file_path (str): Path to your CSV file
        window_months (int): Window size in months
        max_iterations (int): Maximum iterations to run (None for all data)

    Yields:
        pd.DataFrame: Each iteration yields a sliding window of OHLC data
    """
    simulator = SlidingWindowSimulator(csv_file_path, window_months)

    iteration = 0
    while True:
        # Get current sliding window data
        current_data = simulator.get_current_data()

        if current_data is None:
            print("✅ Simulation complete - no more data")
            break

        # Show progress
        progress = simulator.get_simulation_progress()
        print(f"🕒 Iteration {iteration + 1}: {progress['progress_pct']}% complete "
              f"({progress['current_point']}/{progress['total_points']})")

        # Yield the current window (this replaces your fetch_ohlc call)
        yield current_data

        # Advance to next time point
        if not simulator.advance_time():
            break

        iteration += 1

        # Stop if max iterations reached
        if max_iterations and iteration >= max_iterations:
            print(f"🛑 Stopped after {max_iterations} iterations")
            break


# Example usage function to replace your fetch_ohlc calls
def fetch_ohlc_simulation(symbol, csv_file_path, interval=15, months=5):
    """
    Drop-in replacement for your fetch_ohlc function during testing

    Args:
        symbol: Symbol (ignored in simulation)
        csv_file_path: Path to CSV file
        interval: Interval (ignored in simulation)
        months: Months (used as window size)

    Returns:
        pd.DataFrame: OHLC data for current simulation step
    """
    # This would be called from a global simulator instance
    # For your testing, you'll initialize this once and call get_current_data()

    if not hasattr(fetch_ohlc_simulation, 'simulator'):
        fetch_ohlc_simulation.simulator = SlidingWindowSimulator(csv_file_path, months)

    return fetch_ohlc_simulation.simulator.get_current_data()


# Test function to verify the logic
def test_sliding_window_logic(csv_file_path, num_steps=5):
    """
    Test function to verify sliding window behavior

    Args:
        csv_file_path (str): Path to CSV file
        num_steps (int): Number of simulation steps to run
    """
    print("🧪 Testing Sliding Window Logic")
    print("=" * 50)

    simulator = SlidingWindowSimulator(csv_file_path, window_months=6)

    for step in range(num_steps):
        print(f"\n--- Step {step + 1} ---")

        # Get current data
        data = simulator.get_current_data()

        if data is None:
            break

        print(f"Window size: {len(data)} rows")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"Latest close: {data['close'].iloc[-1]:.2f}")

        # Show progress
        progress = simulator.get_simulation_progress()
        print(f"Progress: {progress['progress_pct']}% ({progress['current_point']}/{progress['total_points']})")

        # Advance time
        if not simulator.advance_time():
            print("No more data available")
            break

    print("\n✅ Test completed")


if __name__ == "__main__":
    # Example usage
    csv_path = "your_data.csv"  # Replace with your CSV path

    # Test the logic
    test_sliding_window_logic(csv_path, num_steps=3)

    # Full simulation example
    print("\n" + "=" * 50)
    print("🚀 Running Full Simulation")

    for i, data in enumerate(simulate_realtime_trading(csv_path, window_months=6, max_iterations=10)):
        print(f"Iteration {i + 1}: Got {len(data)} rows, latest close: {data['close'].iloc[-1]:.2f}")

        # Here you would call your strategy function
        # your_strategy_function_with_data(data)