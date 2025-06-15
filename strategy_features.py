# strategy_features.py — fixed for pandas indexing

import pandas as pd
import numpy as np

def rational_quadratic_kernel(series, h, r, x):
    values = []
    for i in range(len(series)):
        num = 0
        den = 0
        for k in range(max(0, i - h), i + 1):
            dist = (i - k) ** 2
            w = (1 + dist / (r * x**2)) ** -r
            num += w * series.iloc[k]
            den += w
        values.append(num / den if den != 0 else np.nan)
    return pd.Series(values, index=series.index)

def gaussian_kernel(series, h, x):
    values = []
    for i in range(len(series)):
        num = 0
        den = 0
        for k in range(max(0, i - h), i + 1):
            dist = (i - k) ** 2
            w = np.exp(-dist / (2 * x**2))
            num += w * series.iloc[k]
            den += w
        values.append(num / den if den != 0 else np.nan)
    return pd.Series(values, index=series.index)

# Add your indicator wrappers (RSI, ADX, etc.) as needed
def compute_feature(df, kind, param_a, param_b):
    # Replace with actual computation logic, e.g. using ta-lib or pandas-ta
    return pd.Series(df['close'].rolling(param_a).mean(), index=df.index)

def lorentzian_distance(x, y):
    """
    Computes Lorentzian distance between two vectors x and y
    """
    return np.sum(np.log(1 + np.abs(x - y)))