import pandas as pd
import numpy as np
import pandas_ta as pta


# Moving Average (selectable mode via pandas_ta.ma)
def pta_sma(
    df: pd.DataFrame,
    length: int = 14,
) -> pd.Series:
    s = pta.sma(close=df["Close"], length=length)
    return s.rename(f"{length}")

# Moving Average (selectable mode via pandas_ta.ma)
def pta_ema(
    df: pd.DataFrame,
    length: int = 14,
) -> pd.Series:
    s = pta.ema(close=df["Close"], length=length)
    return s.rename(f"{length}")

# Moving Average Convergence Divergence (MACD)
def pta_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD line, signal, and histogram using pandas_ta.
    Returns a DataFrame with columns renamed to be parameter-explicit.
    When used with IndicatorPipeline(name="macd"), output columns become:
    - macd_line_<fast>_<slow>_<signal>
    - macd_signal_<fast>_<slow>_<signal>
    - macd_hist_<fast>_<slow>_<signal>
    """
    out = pta.macd(close=df["Close"], fast=fast, slow=slow, signal=signal)
    if isinstance(out, pd.DataFrame):
        cols = list(out.columns)
        rename_map = {}
        # pandas_ta returns columns typically like: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        # We normalize to line/signal/hist with explicit params
        if len(cols) >= 1:
            rename_map[cols[0]] = f"line_{fast}_{slow}_{signal}"
        if len(cols) >= 2:
            rename_map[cols[1]] = f"signal_{fast}_{slow}_{signal}"
        if len(cols) >= 3:
            rename_map[cols[2]] = f"hist_{fast}_{slow}_{signal}"
        return out.rename(columns=rename_map)
    return out



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

    if isinstance(out, pd.DataFrame):
        cols = list(out.columns)
        rename_map = {}
        # pandas_ta returns columns typically like: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        # We normalize to line/signal/hist with explicit params
        if len(cols) >= 1:
            rename_map[cols[0]] = f"line_{fast}_{slow}_{lag}_{signal}"
        if len(cols) >= 2:
            rename_map[cols[1]] = f"signal_{fast}_{slow}_{lag}_{signal}"
        if len(cols) >= 3:
            rename_map[cols[2]] = f"hist_{fast}_{slow}_{lag}_{signal}"
        return out.rename(columns=rename_map)
    return out

# Percentage Price Oscillator (PPO)
def pta_ppo(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    PPO line, signal, and histogram using pandas_ta.

    Returns a DataFrame with columns renamed to:
    - line_<fast>_<slow>_<signal>
    - signal_<fast>_<slow>_<signal>
    - hist_<fast>_<slow>_<signal>
    When used with IndicatorPipeline(name="ppo") the final columns will be prefixed (e.g., ppo_line_12_26_9).
    """
    out = pta.ppo(close=df["Close"], fast=fast, slow=slow, signal=signal)
    if isinstance(out, pd.DataFrame):
        cols = list(out.columns)
        rename_map = {}
        for c in cols:
            lc = str(c).lower()
            if "ppo" in lc and ("ppos" not in lc and "ppoh" not in lc):
                rename_map[c] = f"line_{fast}_{slow}_{signal}"
            elif "ppos" in lc or "signal" in lc:
                rename_map[c] = f"signal_{fast}_{slow}_{signal}"
            elif "ppoh" in lc or "hist" in lc:
                rename_map[c] = f"hist_{fast}_{slow}_{signal}"
        # Fallback by position if detection missed
        if not rename_map and len(cols) >= 1:
            rename_map[cols[0]] = f"line_{fast}_{slow}_{signal}"
            if len(cols) >= 2:
                rename_map[cols[1]] = f"signal_{fast}_{slow}_{signal}"
            if len(cols) >= 3:
                rename_map[cols[2]] = f"hist_{fast}_{slow}_{signal}"
        return out.rename(columns=rename_map)
    return out

#RSI
def pta_rsi(
    df: pd.DataFrame,
    length: int = 14
) -> pd.Series:
    s = pta.rsi(close=df["Close"], length=length)
    return s.rename(f"{length}")




#Bollinger Bands
def pta_bbands(
    df: pd.DataFrame,
    length: int = 20,
    std: float = 2.0
) -> pd.DataFrame:
    out = pta.bbands(close=df["Close"], length=length, std=std)
    if isinstance(out, pd.DataFrame):
        cols = list(out.columns)
        rename_map = {}
        if len(cols) >= 1:
            rename_map[cols[0]] = f"lower_{length}_{std}"
        if len(cols) >= 2:
            rename_map[cols[1]] = f"mid_{length}_{std}"
        if len(cols) >= 3:
            rename_map[cols[2]] = f"upper_{length}_{std}"
        return out.rename(columns=rename_map)
    return out


#Actual True Range(ATR)
def pta_atr(
    df: pd.DataFrame,
    length: int = 14
) -> pd.Series:
    atr = pta.atr(high=df["High"], low=df["Low"], close=df["Close"], length=length).rename(f"{length}")
    return atr


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
    s = pta.mfi(high=high, low=low, close=close, volume=volume, length=length)
    return s.astype("float64").rename(f"{length}")


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
    return out.rename(f"{length}").astype("float64")


# Rate of Change (ROC)
def pta_roc(
    df: pd.DataFrame,
    length: int = 10,
) -> pd.Series:
    s = pta.roc(close=df["Close"], length=length)
    return s.rename(f"{length}")


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
    return width.rename(f"{length}_{std}")


# On-Balance Volume (OBV)
def pta_obv(
    df: pd.DataFrame,
) -> pd.Series:
    s = pta.obv(close=df["Close"], volume=df["Volume"])
    return s.rename("obv")


# Multi-horizon returns
def returns_multi(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Percentage returns over multiple horizons (bars).
    """
    horizons = horizons or [1, 5, 20]
    close = pd.to_numeric(df["Close"], errors="coerce")
    out = {}
    for h in horizons:
        out[f"{h}"] = close.pct_change(h)
    return pd.DataFrame(out, index=df.index)


# Rolling skew and kurtosis of 1-bar returns
def returns_rolling_moments(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    r = pd.to_numeric(df["Close"], errors="coerce").pct_change()
    skew = r.rolling(window).skew().rename(f"skew_{window}")
    kurt = r.rolling(window).kurt().rename(f"kurt_{window}")
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
    return corr.rename(f"autocorr_{lag}_{window}")


#Stochastic Oscillator (STOCH)
def pta_stoch(
    df: pd.DataFrame,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3
) -> pd.DataFrame:
    out = pta.stoch(high=df["High"], low=df["Low"], close=df["Close"], k=k, d=d, smooth_k=smooth_k)
    if isinstance(out, pd.DataFrame):
        cols = list(out.columns)
        rename_map = {}
        if len(cols) >= 1:
            rename_map[cols[0]] = f"k_{k}_{d}_{smooth_k}"
        if len(cols) >= 2:
            rename_map[cols[1]] = f"d_{k}_{d}_{smooth_k}"
        return out.rename(columns=rename_map)
    return out


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

    Returns a DataFrame with normalized column names:
    - srsi_k_<length>_<rsi_length>_<k>_<d>
    - srsi_d_<length>_<rsi_length>_<k>_<d>
    """
    out = pta.stochrsi(close=df["Close"], length=length, rsi_length=rsi_length, k=k, d=d)
    if isinstance(out, pd.DataFrame):
        cols = list(out.columns)
        rename_map = {}
        if len(cols) >= 1:
            rename_map[cols[0]] = f"k_{length}_{rsi_length}_{k}_{d}"
        if len(cols) >= 2:
            rename_map[cols[1]] = f"d_{length}_{rsi_length}_{k}_{d}"
        return out.rename(columns=rename_map)
    return out

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
    out = pta.adx(high=df["High"], low=df["Low"], close=df["Close"], length=length)
    if isinstance(out, pd.DataFrame):
        rename_map = {}
        for c in list(out.columns):
            lc = str(c).lower()
            if "adx" in lc:
                rename_map[c] = f"{length}"
            elif "dmp" in lc or "+di" in lc or "pdi" in lc or "plus" in lc:
                rename_map[c] = f"di_plus_{length}"
            elif "dmn" in lc or "-di" in lc or "mdi" in lc or "minus" in lc:
                rename_map[c] = f"di_minus_{length}"
        return out.rename(columns=rename_map)
    return out


