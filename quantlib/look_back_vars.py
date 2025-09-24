

import pandas as pd
import numpy as np
import pandas_ta as pta
from scipy.stats import norm


# Moving Average (selectable mode via pandas_ta.ma)
def pta_sma(
    df: pd.DataFrame,
    length: int = 14,
) -> pd.Series:
    return pta.sma(close=df["Close"], length=length)

# Moving Average (selectable mode via pandas_ta.ma)
def pta_ema(
    df: pd.DataFrame,
    length: int = 14,
) -> pd.Series:
    return pta.ema(close=df["Close"], length=length)

# Moving Average Convergence Divergence (MACD)
def pta_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD line, signal, and histogram using pandas_ta.
    """
    return pta.macd(close=df["Close"], fast=fast, slow=slow, signal=signal)



# Moving Average Convergence Divergence (MACD) Scaled by ATR (customized)
def macd_atr_scaled(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    lag: int = 12,
) -> pd.DataFrame:
    """
    MACD line, signal, and histogram scaled by ATR with lag
    Returns a DataFrame with columns renamed to be parameter-explicit.
    The first available computation will happen at the slow+lag+1 bar (first bar no true range)
    When used with IndicatorPipeline(name="macd"), output columns become:
    - line_<fast>_<slow>_{lag}_<signal>
    - signal_<fast>_<slow>_{lag}_<signal>
    - hist_<fast>_<slow>_{lag}_<signal>
    """

    ema_fast = pta.ema(close=df["Close"], length=fast)
    ema_slow = pta.ema(close=df["Close"], length=slow)
    

    atr = pta.atr(close=df["Close"],
              high = df["High"],
              low = df["Low"], length=slow+lag)

    ema_diff = ema_fast - ema_slow.shift(lag)
    ema_normed_diff = ema_diff/atr/np.sqrt(0.5*(slow-fast)+lag)

    macd_signal = pta.ema(close = ema_normed_diff, length=signal)
    macd_hist = ema_normed_diff - macd_signal

    out = pd.concat([ema_normed_diff, macd_signal, macd_hist], axis = 1)
    cols = list(out.columns)
    rename_map = {}
    rename_map[cols[0]] = f"MACD_scaled_{fast}_{slow}_{lag}_{signal}"
    rename_map[cols[1]] = f"MACDs_scaled_{fast}_{slow}_{lag}_{signal}"
    rename_map[cols[2]] = f"MACDh_scaled_{fast}_{slow}_{lag}_{signal}"
    return out.rename(columns=rename_map)

# Percentage Price Oscillator (PPO)
def pta_ppo(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    PPO line, signal, and histogram using pandas_ta.
    """
    return pta.ppo(close=df["Close"], fast=fast, slow=slow, signal=signal)

#RSI
def pta_rsi(
    df: pd.DataFrame,
    length: int = 14
) -> pd.Series:
    return pta.rsi(close=df["Close"], length=length)




#Bollinger Bands
def pta_bbands(
    df: pd.DataFrame,
    length: int = 20,
    std: float = 2.0
) -> pd.DataFrame:
    return pta.bbands(close=df["Close"], length=length, std=std)


#Actual True Range(ATR)
def pta_atr(
    df: pd.DataFrame,
    length: int = 14
) -> pd.Series:
    return pta.atr(high=df["High"], low=df["Low"], close=df["Close"], length=length)


# Money Flow Index (MFI)
def pta_mfi(
    df: pd.DataFrame,
    length: int = 14,
) -> pd.Series:
    """
    Money Flow Index using pandas_ta.mfi.

    Returns a Series named mfi_<length>.
    """
    # Ensure float dtypes to avoid pandas FutureWarning when internal ops set floats into int series
    high = pd.to_numeric(df["High"], errors="coerce").astype("float64")
    low = pd.to_numeric(df["Low"], errors="coerce").astype("float64")
    close = pd.to_numeric(df["Close"], errors="coerce").astype("float64")
    volume = pd.to_numeric(df["Volume"], errors="coerce").astype("float64")
    return  pta.mfi(high=high, low=low, close=close, volume=volume, length=length)


# Custom MFI implementation (avoids dtype issues and external dependencies)
def mfi_custom(
    df: pd.DataFrame,
    length: int = 14,
) -> pd.Series:
    """
    Money Flow Index computed in pure pandas to avoid dtype warnings.

    Formula
    - Typical Price (TP) = (High + Low + Close) / 3
    - Money Flow (MF) = TP * Volume
    - Positive/Negative Money Flow split by change in TP
    - MFI = 100 - 100 / (1 + sum(PMF) / sum(NMF)) over `length`
    """
    high = pd.to_numeric(df["High"], errors="coerce").astype("float64")
    low = pd.to_numeric(df["Low"], errors="coerce").astype("float64")
    close = pd.to_numeric(df["Close"], errors="coerce").astype("float64")
    volume = pd.to_numeric(df["Volume"], errors="coerce").astype("float64")

    tp = (high + low + close) / 3.0
    dtp = tp.diff()
    flow = tp * volume
    pos = flow.where(dtp > 0.0, 0.0)
    neg = flow.where(dtp < 0.0, 0.0)

    pmf = pos.rolling(length, min_periods=length).sum()
    nmf = neg.rolling(length, min_periods=length).sum()
    mfr = pmf / nmf.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + mfr))
    return out.rename(f"MFI_{length}").astype("float64")


# Rate of Change (ROC)
def pta_roc(
    df: pd.DataFrame,
    length: int = 10,
) -> pd.Series:
    return pta.roc(close=df["Close"], length=length)


# Bollinger Band Width (BBW)
def pta_bbw(
    df: pd.DataFrame,
    length: int = 20,
    std: float = 2.0,
    normalize: bool = False,
) -> pd.Series:
    """
    Bollinger Band width = upper - lower (optionally normalized by middle band).
    """
    bands = pta.bbands(close=df["Close"], length=length, std=std)
    if not isinstance(bands, pd.DataFrame) or bands.shape[1] < 3:
        return pd.Series(index=df.index, name=f"bbw_{length}_{std}")
    lower = bands.iloc[:, 0]
    mid = bands.iloc[:, 1]
    upper = bands.iloc[:, 2]
    width = (upper - lower)
    if normalize:
        with pd.option_context('mode.use_inf_as_na', True):
            width = width / mid.replace(0, pd.NA)
    return width.rename(f"BBandWidth_{length}_{std}")


# On-Balance Volume (OBV)
def pta_obv(
    df: pd.DataFrame,
) -> pd.Series:
    return pta.obv(close=df["Close"], volume=df["Volume"])


# Rolling skew and kurtosis of 1-bar returns
def returns_rolling_moments(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    r = pd.to_numeric(df["Close"], errors="coerce").pct_change()
    skew = r.rolling(window).skew().rename(f"RET_skew_{window}")
    kurt = r.rolling(window).kurt().rename(f"RET_kurt_{window}")
    return pd.concat([skew, kurt], axis=1)


# Rolling autocorrelation of returns at given lag
def returns_autocorr(
    df: pd.DataFrame,
    lag: int = 1,
    window: int = 100,
) -> pd.Series:
    r = pd.to_numeric(df["Close"], errors="coerce").pct_change()
    x = r
    y = r.shift(lag)
    mean_x = x.rolling(window).mean()
    mean_y = y.rolling(window).mean()
    cov = (x * y).rolling(window).mean() - mean_x * mean_y
    std_x = x.rolling(window).std(ddof=0)
    std_y = y.rolling(window).std(ddof=0)
    corr = cov / (std_x * std_y)
    return corr.rename(f"RET_autocorr_{lag}_{window}")


#Stochastic Oscillator (STOCH)
def pta_stoch(
    df: pd.DataFrame,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3
) -> pd.DataFrame:
    return pta.stoch(high=df["High"], low=df["Low"], close=df["Close"], k=k, d=d, smooth_k=smooth_k)


# Stochastic RSI (StochRSI)
def pta_stochrsi(
    df: pd.DataFrame,
    length: int = 14,
    rsi_length: int = 14,
    k: int = 3,
    d: int = 3,
) -> pd.DataFrame:
    """
    Stochastic RSI using pandas_ta.stochrsi.
    """
    return pta.stochrsi(close=df["Close"], length=length, rsi_length=rsi_length, k=k, d=d)
    

# Average Directional Index (ADX) with +DI and -DI
def pta_adx(
    df: pd.DataFrame,
    length: int = 14,
) -> pd.DataFrame:
    """
    ADX and directional indicators using pandas_ta.

    Returns a DataFrame with normalized column names:
    - adx_<length>
    - di_plus_<length>
    - di_minus_<length>
    """
    return pta.adx(high=df["High"], low=df["Low"], close=df["Close"], length=length)


# Price intensity
def p_intensity(
    df: pd.DataFrame,
    length: int = 14,
) -> pd.Series:
    raw_out = (df["Close"] - df["Open"]) / pta.atr(high=df["High"], low=df["Low"], close=df["Close"], length=1)
    out = pta.ema(close=raw_out, length=length)
    return out.rename(f"P_intensity_{length}")



def pta_aroon(
    df: pd.DataFrame,
    length: int = 25,
) -> pd.DataFrame:
    """
    Aroon indicator using pandas_ta.

    Returns a DataFrame with columns:
    - AROON_UP_<length>
    - AROON_DOWN_<length>
    """
    out = pta.aroon(high=df["High"], low=df["Low"], length=length)
    # Rename columns for consistency
    out = out.rename(columns={
        f"AROONU_{length}": f"AROON_UP_{length}",
        f"AROOND_{length}": f"AROON_DOWN_{length}"
    })
    return out


def legendre_trend_r2_atr_scaled(
    df: pd.DataFrame,
    lookback: int = 20,
    atr_long: int = 100,
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    Linear-trend indicator based on first-order Legendre regression of log-price,
    scaled by long-term ATR and weighted by R² fit.

    Returns a DataFrame with one column:
    - legendre_trend_<lookback>_<atr_long>

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [Open, High, Low, Close] (or specify `price_col`).
    lookback : int
        Number of bars in the regression window.
    atr_long : int
        Lookback for long-term ATR used for volatility scaling.
    price_col : str
        Column to use for prices (default "Close").
    """

    # --- Precompute first-order Legendre polynomial coefficients for [-1,1] ---
    x = np.linspace(-1, 1, lookback)
    # normalized so dot gives slope*2/(lookback-1) (as regressed on [-1,1])
    legendre1 = x / np.sum(x**2) * 2  


    # --- Prepare log prices ---
    log_price = np.log(df[price_col])

    # Rolling dot product: slope in log space
    # Each value is 2 * slope / (lookback - 1)
    def legendre_slope(series):
        return np.dot(series.values, legendre1)

    raw_slope = log_price.rolling(lookback).apply(legendre_slope, raw=False)

    # --- Compute R² for each window ---
    def r2(series):
        y = series.values
        y_hat = np.dot(y, legendre1) * x / 2 + y.mean()
        ss_res = np.sum((y - y_hat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0

    r2_score = log_price.rolling(lookback).apply(r2, raw=False)

    # --- Long-term ATR for volatility scaling (normalize by current price) ---
    atr = pta.atr(
        high=df["High"], low=df["Low"], close=df["Close"], length=atr_long
    )
    atr_fraction = atr / df[price_col]

    # --- Final indicator: R² × slope / ATRfraction ---
    trend = r2_score * raw_slope / atr_fraction

    out = trend.to_frame(name=f"legendre_trend_{lookback}_{atr_long}")
    return out


def cmma(
    df: pd.DataFrame,
    n: int = 20,
    c: float = 1.0,
    atr_length: int = 14,
) -> pd.Series:
    """
    Close Minus Moving Average (CMMA) indicator.
    
    Formula: CMMA_t^(n) = 100 × Φ(C ⋅ (log C_t - Σ_{i=1}^{n} log C_{t-i}) / (ATR_t ⋅ √(n + 1))) - 50
    
    Where:
    - Φ is the cumulative distribution function of the standard normal distribution
    - C is a constant multiplier (default 1.0)
    - log C_t is the natural logarithm of the Close price at time t
    - Σ_{i=1}^{n} log C_{t-i} is the sum of log Close prices for the previous n periods
    - ATR_t is the Average True Range at time t
    - √(n + 1) is the square root of n + 1
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [Open, High, Low, Close].
    n : int
        Number of periods for the moving average calculation (default 20).
    c : float
        Constant multiplier in the formula (default 1.0).
    atr_length : int
        Length for ATR calculation (default 14).
    
    Returns
    -------
    pd.Series
        CMMA values with name "CMMA_<n>_<c>_<atr_length>".
    """
    
    # Calculate log of close prices
    log_close = np.log(df["Close"])
    
    # Calculate mean of log close prices for previous n periods
    # Shift first, then calculate rolling mean to get mean of previous n periods only
    log_close_mean = log_close.shift(1).rolling(window=n, min_periods=n).mean()
    
    # Calculate the numerator: log C_t - mean of log C_{t-i} for i=1 to n
    numerator = log_close - log_close_mean
    
    # Calculate ATR
    atr = pta.atr(high=df["High"], low=df["Low"], close=df["Close"], length=atr_length)
    
    # Calculate the denominator: ATR_t ⋅ √(n + 1)
    denominator = atr * np.sqrt(n + 1)
    
    # Calculate the argument for the normal CDF
    
    argument = c *numerator / denominator
    
    # Apply the normal CDF (Φ) and scale
    # argument is a pandas Series, norm.cdf will work elementwise and return a numpy array of same shape
    cmma_values = 100 * norm.cdf(argument.values) - 50
    cmma_values = pd.Series(cmma_values, index=argument.index)
    
    return cmma_values.rename(f"CMMA_{n}_{c}_{atr_length}")


def diff_close_legendre_fit(
    df: pd.DataFrame,
    n: int = 20,
) -> pd.Series:
    """
    Polynomial regression indicator using Legendre polynomials.
    
    Fits a polynomial regression of degree `degree` to the last `n` close prices
    and returns the normalized deviation of current price from the fitted value.
    
    Formula: (current_price - fitted_current_price) / RMSE
    
    Where:
    - Fits polynomial regression using Legendre polynomials on x ∈ [-1, 1]
    - RMSE is the root mean square error of the regression
    - Current price deviation is normalized by the regression's RMSE
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain the price column (default "Close").
    n : int
        Number of bars to look back for regression (default 20).
    
    Returns
    -------
    pd.Series
        Polynomial regression indicator values with name "poly_reg_<n>_<degree>".
    """
    
    # Prepare price data
    prices = df['Close'].astype(float)
    
    # Create x values as linear space from -1 to 1 (same as legendre_trend_r2_atr_scaled)
    x = np.linspace(-1, 1, n)
    
    # Precompute Legendre polynomial basis functions
    # For degree 1, 2, 3: linear, quadratic, cubic
    basis_1 = x
    # Quadratic: P2(x) = (3x² - 1) / 2
    basis_2 = (3 * x**2 - 1) / 2
    # Cubic: P3(x) = (5x³ - 3x) / 2
    basis_3 = (5 * x**3 - 3 * x) / 2
    
    # Create design matrix X
    X = np.column_stack([np.ones(n), basis_1, basis_2, basis_3])

    
    # Calculate (X'X)^(-1)X' for efficient regression
    XtX_inv = np.linalg.inv(X.T @ X)
    XtX_inv_Xt = XtX_inv @ X.T
    
    def polynomial_regression_rolling(price_window):
        """Apply polynomial regression to a window of prices."""
        if len(price_window) < n or price_window.isna().any():
            return np.nan
        
        y = price_window.values
        
        # Fit regression: coefficients = (X'X)^(-1)X'y
        coeffs = XtX_inv_Xt @ y
        
        # Calculate fitted values
        y_fitted = X @ coeffs
        
        # Calculate RMSE
        residuals = y - y_fitted
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Current price deviation (last element)
        current_deviation = y[-1] - y_fitted[-1]
        
        # Return normalized deviation
        if rmse == 0:
            return 0
        return current_deviation / rmse
    
    # Apply rolling regression
    result = prices.rolling(window=n, min_periods=n).apply(
        polynomial_regression_rolling, raw=False
    )
    
    return result.rename(f"poly_reg_{n}")



