import pandas as pd
import numpy as np


def log_return_forward(df: pd.DataFrame, n: int = 1) -> pd.Series:
    """
    Forward log return over the next `n` bars.

    Formula: log(C_{t+n} / C_t)

    Parameters
    - df: DataFrame with a `Close` column
    - n: lookahead horizon in bars

    Returns
    - pd.Series named `logret_fwd_<n>` aligned to df.index
    """
    c = pd.to_numeric(df["Close"], errors="coerce").astype("float64")
    out = np.log(c.shift(-n) / c)
    return out.rename(f"{n}")


def profit_factor_forward(df: pd.DataFrame, n: int = 14, sigmoid: bool = False) -> pd.Series:
    """
    Forward Profit Factor (PF) for the next `n` bars.

    PF_t^{(n)} = sum_{i=t+1}^{t+n} (ΔC_i)^+ / sum_{i=t+1}^{t+n} (-ΔC_i)^+

    Where ΔC_i = C_i - C_{i-1}. Range is [0, ∞). If the denominator
    is 0 while the numerator is > 0, the result is +inf. If both are 0,
    result is NaN.

    Parameters
    - df: DataFrame with a `Close` column
    - n: lookahead horizon in bars

    Returns
    - pd.Series named `pf_fwd_<n>`; if `sigmoid=True`, returns a [0,1] mapping via PF/(1+PF)
    """
    c = pd.to_numeric(df["Close"], errors="coerce").astype("float64")
    d = c.diff()
    pos = d.clip(lower=0.0)
    neg = (-d).clip(lower=0.0)

    # Use forward-looking rolling sums by pre-shifting by -n
    pos_sum = pos.shift(-n).rolling(window=n, min_periods=n).sum()
    neg_sum = neg.shift(-n).rolling(window=n, min_periods=n).sum()

    with pd.option_context("mode.use_inf_as_na", False):
        pf = pos_sum / neg_sum.replace(0.0, 0.0)

    # If both pos_sum and neg_sum are 0 -> set 0 (no movement window)
    pf = pf.where(~((pos_sum == 0.0) & (neg_sum == 0.0)), other=0.0)

    if sigmoid:
        # Map [0, ∞) -> [0, 1) with neutral PF=1 -> 0.5
        pf = pf / (1.0 + pf)
    return pf.rename(f"{n}")


def return_over_variance_forward(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Forward "profit factor" as defined in the doc screenshot:

    PF_t^{(n)} = (Σ (ΔC_i)^+ − Σ (−ΔC_i)^+) / Σ |ΔC_i|,  i=t+1..t+n

    where ΔC_i = C_i − C_{i-1}. Note the numerator simplifies to
    Σ ΔC_i = C_{t+n} − C_t. Range is [-1, 1]. If the denominator is 0,
    returns NaN.

    Parameters
    - df: DataFrame with a `Close` column
    - n: lookahead horizon in bars

    Returns
    - pd.Series named `pf_fwd_<n>`
    """
    c = pd.to_numeric(df["Close"], errors="coerce").astype("float64")
    d = c.diff()

    numer = c.shift(-n) - c  # equals sum of ΔC_i over next n bars
    denom = d.abs().shift(-n).rolling(window=n, min_periods=n).sum()

    pf = numer / denom
    pf = pf.where(denom != 0.0, other=0.0)
    return pf.rename(f"{n}")




def forward_directional_index(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Forward Directional Index (FDI) for the next `n` bars, akin to a
    forward-looking ADX-style ratio capturing up/down momentum magnitude.

    Steps per bar i:
    - +DM_i = max(High_i - High_{i-1}, 0)
    - -DM_i = max(Low_{i-1} - Low_i, 0)
    - Keep only the larger of (+DM_i, -DM_i) (the other set to 0)

    Then over the next n bars:
    FDI_t^{(n)} = (Σ +DM_i − Σ −DM_i) / (Σ +DM_i + Σ −DM_i), i = t+1..t+n
    Range [-1, 1]. If denominator is 0, result is NaN.

    Parameters
    - df: DataFrame with `High` and `Low` columns
    - n: lookahead horizon in bars

    Returns
    - pd.Series named `fdi_fwd_<n>`
    """
    high = pd.to_numeric(df["High"], errors="coerce").astype("float64")
    low = pd.to_numeric(df["Low"], errors="coerce").astype("float64")

    plus_dm_raw = (high - high.shift(1)).clip(lower=0.0)
    minus_dm_raw = (low.shift(1) - low).clip(lower=0.0)

    # Only one of +DM/-DM can be positive per bar
    mask_plus = plus_dm_raw > minus_dm_raw
    plus_dm = plus_dm_raw.where(mask_plus, 0.0)
    minus_dm = minus_dm_raw.where(~mask_plus, 0.0)

    plus_sum = plus_dm.shift(-n).rolling(window=n, min_periods=n).sum()
    minus_sum = minus_dm.shift(-n).rolling(window=n, min_periods=n).sum()

    denom = plus_sum + minus_sum
    numer = plus_sum - minus_sum
    fdi = numer / denom
    fdi = fdi.where(denom != 0.0, other=0.0)
    return fdi.rename(f"{n}")
