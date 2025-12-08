


# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import klines_samples
import importlib

importlib.reload(klines_samples)

# %%
# loading data and set timestamp as index
df = pd.read_csv("DATA/organized/BTCUSDT/klines_1m.csv")
df["ts"] = pd.to_datetime(df["ts"])
df.set_index("ts", inplace=True, drop=True)
df.index.name = None



# %%
df_aggTrades = pd.read_parquet("DATA/organized/BTCUSDT/aggTrades.parquet")#

df_aggTrades.set_index("ts", inplace=True, drop=True)


df_aggTrades.index.name = None

df_aggTrades['sell_size'] = df_aggTrades['size'] * df_aggTrades['isBuyerMaker']
df_aggTrades['buy_size'] = df_aggTrades['size'] * (1 - df_aggTrades['isBuyerMaker'])

df['sell_volume'] = df_aggTrades['sell_size'].resample('1min').sum()
df['buy_volume'] = df_aggTrades['buy_size'].resample('1min').sum()


df_volume = pd.concat([df_aggTrades['sell_size'].resample('10s').sum(),
           df_aggTrades['buy_size'].resample('10s').sum()], axis=1)

# %%
# --------------------------------------------
# PRODUCT OF LOG(HIGH/LOW) AND LOG(CLOSE/OPEN)
# --------------------------------------------
# hlco_change = \sum_{i=1}^{n} log(high/low) * log(close/open)
def hlco_change(df: pd.DataFrame, n: int) -> pd.Series:
    temp = np.log(df['high'] / df['low']) * np.log(df['close'] / df['open'])
    return temp.rolling(window=n, min_periods=n).sum()


# --------------------------------------------  
# Range-Based Volatility Spike 
# --------------------------------------------
def rangebased_vol_spike(df: pd.DataFrame, window: int = 200) -> pd.Series:
    temp = np.log(df['high'] / df['low']).abs()

    rolling_median = temp.rolling(window).median()
    rolling_mad = temp.rolling(window).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=False
    )
    rolling_mad = rolling_mad.replace(0, np.nan)

    return (temp - rolling_median) / rolling_mad

# Range-Based momentum Spike
# the formula has a huge problem, if the past volatility for a long time is low
# then the rolling median and rolling_mad are both very small
# so a short period of spikes/volatility up and down will include a lot points
# so the spike is in the relative sense that compared to the past volatility

#temp = (np.log(df['high'] / df['low']) * np.log(df['close'] / df['open'])).abs()
#rolling_median = temp.rolling(200).median()
#np.log(rolling_median).hist(bins=200)
#np.log(temp+eps).hist(bins = 200)
# Going a little deeper, the above plot shows a lot of log(high/low) * log(close/open) are indeed zero
# and this happens due to C = O, so this product form is not ideal, and 
# that might be the reason people use Garman-Klass variance estimator instead.


# --------------------------------------------  
# Range-Based Momentum Spike
# --------------------------------------------
def rangebased_momentum_spike(df: pd.DataFrame, window: int = 200) -> pd.Series:
    #eps = 1e-12
    temp = (np.log(df['high'] / df['low']) * np.log(df['close'] / df['open'])).abs()

    rolling_median = temp.rolling(window).median()
    rolling_mad = temp.rolling(window).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=False
    )
    rolling_mad = rolling_mad.replace(0, np.nan)

    return (temp - rolling_median) / rolling_mad

# --------------------------------------------  
# Directional Volatility Spike
# --------------------------------------------

def mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))


def directional_gk_break(df: pd.DataFrame, window: int = 200) -> pd.Series:
    

    
    # 1. GK computation
    GK = 0.5 * (np.log(df['high'] / df['low']))**2 - \
     (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']))**2
    # numerical guard: GK shouldn't be negative, but round-off can make it slightly < 0
    GK = GK.clip(lower=0)

    df['GK'] = GK

    # 2. Directional version: sign(C - O) * sqrt(GK)
    direction = np.sign(df['close'] - df['open'])  # +1 up, -1 down, 0 flat
    GK_dir = direction * np.sqrt(GK)
    

    # 3. Robust normalization: rolling median + MAD
    rolling_median = GK_dir.rolling(window).median()

    

    rolling_mad = GK_dir.rolling(window).apply(mad, raw=False)
    rolling_mad = rolling_mad.replace(0, np.nan)

    return (GK_dir - rolling_median) / rolling_mad

# --------------------------------------------  
# True range expansion vs ATR
# --------------------------------------------

def atr_expansion(df: pd.DataFrame, atr_window: int = 14,
                  ema_alpha: float | None = None) -> pd.DataFrame:
    """
    Compute True Range (TR), ATR (EMA of TR), and ATR-based expansion ratio.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['open', 'high', 'low', 'close'].
    atr_window : int
        Window (in bars) for ATR. If ema_alpha is None, alpha = 2/(atr_window+1).
    ema_alpha : float or None
        Smoothing factor for EMA. If None, use 2/(atr_window+1).

    Returns
    -------
    df_out : pandas.DataFrame
        Original df with:
        - 'TR'       : True Range
        - 'ATR'      : ATR via EMA of TR
        - 'ATR_exp'  : TR / ATR  (range expansion index)
    """
    df = df.copy()

    H = df['high']
    L = df['low']
    C = df['close']

    # ----------------------------
    # 1. True Range (TR)
    # TR_t = max(
    #   H_t - L_t,
    #   |H_t - C_{t-1}|,
    #   |L_t - C_{t-1}|
    # )
    # ----------------------------
    prev_close = C.shift(1)

    tr1 = H - L
    tr2 = (H - prev_close).abs()
    tr3 = (L - prev_close).abs()

    TR = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ----------------------------
    # 2. ATR (EMA of TR)
    # ----------------------------
    if ema_alpha is None:
        ema_alpha = 2 / (atr_window + 1.0)

    ATR = TR.ewm(alpha=ema_alpha, adjust=False).mean()

    # ----------------------------
    # 3. Expansion ratio: TR / ATR
    # ----------------------------
    return TR / ATR

# --------------------------------------------  
# Single- bar deviation to the EMA
# --------------------------------------------


def n_bar_dev_to_ema_under(df: pd.DataFrame, n: int = 1, ema_window: int = 20) -> pd.Series:

    ema = df['close'].ewm(span=ema_window, adjust=False).mean()
    return ((ema.shift(1) - (df['low'] + df['high']) / 2) * (df['close'] <= df['open'])).rolling(n).mean()


def n_bar_dev_to_ema_above(df: pd.DataFrame, n: int = 1, ema_window: int = 20) -> pd.Series:

    ema = df['close'].ewm(span=ema_window, adjust=False).mean()
    return (((df['high'] + df['low']) / 2 - ema.shift(1))* (df['close'] >= df['open'])).rolling(n).mean()

# --------------------------------------------  
# Legendre polynomial fitting
# --------------------------------------------


def polynomial_coeff(df: pd.DataFrame, window: int = 3) -> pd.Series:

    ema20_low = df['close'].ewm(span=20, adjust=False).mean().shift(1) - df['low']
    ema20_high = df['close'].ewm(span=20, adjust=False).mean().shift(1) - df['high']
    x = np.arange(window)
    factor = 1.0 / (x.T @ x) * x.T 

    def legendre_fitting(y: np.ndarray) -> np.ndarray:
        y = y - np.ones(len(y)) * y[0]
        return factor @ y

    def past_n_greater_than_zero(y: np.ndarray) -> bool:
        return np.all(y > 0)
    
    filter_bool = ema20_high.rolling(window).apply(past_n_greater_than_zero, raw=True)
    result = ema20_low.rolling(window).apply(legendre_fitting, raw=True)
    result[~filter_bool.astype(bool)] = np.nan
    return result


# %%
# --------------------------------------------
# regression coef difference
# --------------------------------------------
def regression_coef_difference(df: pd.DataFrame, window: int = 5) -> pd.Series:

    ema20_shift1 = df['close'].ewm(span=20, adjust=False).mean().shift(1)
    ema20_high = df['close'].ewm(span=20, adjust=False).mean().shift(1) - df['high']
    x = np.arange(window)
    

    def linear_fitting(y: np.ndarray) -> np.ndarray:
        # regress y on x and return the linear coef
        A = np.vstack([x, np.ones_like(x)]).T
        coef, _ = np.linalg.lstsq(A, y, rcond=None)[0:2]
        return coef[0]

    def past_n_greater_than_zero(y: np.ndarray) -> bool:
        return np.all(y > 0)
    
    filter_bool = ema20_high.rolling(window-1).apply(lambda x: np.all(x > 0), raw=True)

    ema_fitting = ema20_shift1.rolling(window).apply(linear_fitting, raw=True)
    close_fitting = df['close'].rolling(window).apply(linear_fitting, raw=True)
    result = ema_fitting - close_fitting
    result[~filter_bool.astype(bool)] = np.nan
    return result






# %%
df['past_pk_10'] = hlco_change(df, 10)
df['range_vol_spike_200'] = rangebased_vol_spike(df, window=200)
df['range_momentum_spike_200'] = rangebased_momentum_spike(df, window=200)
df['directional_gk_break_200'] = directional_gk_break(df, window=200)
df['atr_expansion'] = atr_expansion(df, atr_window=14)
df['3_bar_dev_to_ema_under'] = n_bar_dev_to_ema_under(df, n=3,ema_window=20)
df['3_bar_dev_to_ema_above'] = n_bar_dev_to_ema_above(df, n=3,ema_window=20)

df['high_minus_ema_20'] = df['high'] - df['close'].ewm(span=20, adjust=False).mean().shift(1)
df['ema_20_minus_low'] = df['close'].ewm(span=20, adjust=False).mean().shift(1) - df['low']
df['polynomial2_coeff'] = polynomial_coeff(df, window=4)
df['linear_coeff_diff'] = regression_coef_difference(df, window=5)

df['close_over_ema_20'] = np.log(df['close'] / df['close'].ewm(span=20, adjust=False).mean().shift(1))

# %%
# Example usage (requires df['past_pk']):

agg_trades_path = "./DATA/organized/BTCUSDT/aggTrades.parquet"

layout = klines_samples.plot_minibar_with_tail(df, 
                value_column="linear_coeff_diff",
                agg_trades=agg_trades_path,
                ema_window=20,
                left_tail_p=0.00, right_tail_p=0.025)

klines_samples.show(layout)      

 

# %%
