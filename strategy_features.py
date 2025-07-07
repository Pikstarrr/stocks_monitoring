# strategy_features.py — fixed for pandas indexing

import pandas as pd
import numpy as np
import pandas_ta as ta


def rational_quadratic_kernel(series, h, r, x):
    """Rational Quadratic Kernel exactly as in Pine Script"""
    values = []
    series_array = series.values

    for i in range(len(series)):
        sum_weights = 0.0
        sum_values = 0.0

        start_idx = max(0, i - h + 1)

        for j in range(start_idx, i + 1):
            weight = (1 + ((i - j) ** 2) / ((r ** 2) * (h ** 2))) ** (-r)
            sum_weights += weight
            sum_values += weight * series_array[j]

        values.append(sum_values / sum_weights if sum_weights != 0 else series_array[i])

    return pd.Series(values, index=series.index)


def gaussian_kernel(series, h, x):
    """Gaussian Kernel exactly as in Pine Script"""
    values = []
    series_array = series.values

    for i in range(len(series)):
        sum_weights = 0.0
        sum_values = 0.0

        start_idx = max(0, i - h + 1)

        for j in range(start_idx, i + 1):
            weight = np.exp(-((i - j) ** 2) / (2 * (h ** 2)))
            sum_weights += weight
            sum_values += weight * series_array[j]

        values.append(sum_values / sum_weights if sum_weights != 0 else series_array[i])

    return pd.Series(values, index=series.index)

# Add your indicator wrappers (RSI, ADX, etc.) as needed
def compute_feature(df, kind, param_a, param_b):
    """
    Compute technical indicator features matching Pine Script's ml.n_* functions
    """
    if kind == "RSI":
        return compute_rsi(df, param_a, param_b)
    elif kind == "WT":
        return compute_wt(df, param_a, param_b)
    elif kind == "CCI":
        return compute_cci(df, param_a, param_b)
    elif kind == "ADX":
        return compute_adx(df, param_a)
    else:
        raise ValueError(f"Unknown feature kind: {kind}")

def lorentzian_distance(x, y):
    """
    Computes Lorentzian distance between two vectors x and y
    """
    return np.sum(np.log(1 + np.abs(x - y)))


def normalize_series(series, min_val=None, max_val=None):
    """Normalize series to [-1, 1] range like Pine Script's ml.n_* functions"""
    if min_val is None:
        min_val = series.rolling(window=200, min_periods=1).min()
    if max_val is None:
        max_val = series.rolling(window=200, min_periods=1).max()

    normalized = 2 * (series - min_val) / (max_val - min_val) - 1
    return normalized.fillna(0)

def compute_rsi(df, length, lookback):
    """RSI normalized to [-1, 1]"""
    rsi = ta.rsi(df['close'], length=length)
    # Pine Script normalizes RSI from [0, 100] to [-1, 1]
    normalized_rsi = (rsi - 50) / 50
    return normalized_rsi.fillna(0)


def compute_wt(df, channel_length, average_length):
    """Wave Trend Oscillator normalized to [-1, 1]"""
    # Wave Trend calculation
    hlc3 = (df['high'] + df['low'] + df['close']) / 3

    # EMA of HLC3
    ema1 = ta.ema(hlc3, length=channel_length)

    # EMA of absolute difference
    diff = abs(hlc3 - ema1)
    ema2 = ta.ema(diff, length=channel_length)

    # CI (Choppiness Index component)
    ci = (hlc3 - ema1) / (0.015 * ema2)

    # Wave Trend is EMA of CI
    wt1 = ta.ema(ci, length=average_length)

    # Normalize to [-1, 1] - WT typically ranges from -60 to +60
    normalized_wt = wt1 / 60
    normalized_wt = normalized_wt.clip(-1, 1)

    return normalized_wt.fillna(0)

def compute_cci(df, length, lookback):
    """CCI normalized to [-1, 1]"""
    cci = ta.cci(df['high'], df['low'], df['close'], length=length)
    # CCI typically ranges from -200 to +200, normalize to [-1, 1]
    normalized_cci = cci / 200
    normalized_cci = normalized_cci.clip(-1, 1)
    return normalized_cci.fillna(0)

def compute_adx(df, length):
    """ADX normalized to [-1, 1]"""
    adx = ta.adx(df['high'], df['low'], df['close'], length=length)
    if isinstance(adx, pd.DataFrame):
        adx = adx[f'ADX_{length}']
    # ADX ranges from 0 to 100, normalize to [-1, 1]
    # Pine Script uses a different normalization for ADX
    normalized_adx = (adx - 50) / 50
    return normalized_adx.fillna(0)