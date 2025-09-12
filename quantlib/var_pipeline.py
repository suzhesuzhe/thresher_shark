

import pandas as pd
import numpy as np


from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union, List



@dataclass
class VariableSpec:
    """
    Defines one indicator to compute.

    - name: base name; final columns are always "{name}_{col}"
    - fn: callable accepting (df, **params) -> pd.Series | pd.DataFrame
    - params: kwargs forwarded to fn
    - shift: bars to shift output to prevent leakage (None => pipeline default)
    - dtype: dtype for output (None => pipeline default)
    """
    name: str
    fn: Callable[[pd.DataFrame], Union[pd.Series, pd.DataFrame]]
    params: Optional[Dict] = None
    shift: Optional[int] = None
    dtype: Optional[str] = None


class VariablePipeline:
    """
    Config-driven indicator computation pipeline.

    Features:
    - Accepts custom callables or library wrappers (pandas_ta / ta).
    - Works with single- or multi-symbol data (MultiIndex with a 'symbol' level recommended).
    - Shifts outputs by default to avoid target leakage.
    - Returns a separate, aligned features DataFrame (raw OHLCV remains unchanged).
    - Column naming: for both Series and DataFrame outputs, final columns are
      always named as f"{spec.name}_{original_column_name}". For Series without
      a name, the suffix defaults to "value".
    """

    def __init__(
        self,
        specs: List[VariableSpec],
        *,
        default_shift: int = 1,
        default_dtype: str = "float32",
    ) -> None:
        self.specs = specs
        self.default_shift = default_shift
        self.default_dtype = default_dtype
        # Column naming rule:
        # Always use: f"{spec.name}_{column_name}"

    def run(
        self,
        df: pd.DataFrame,
        *,
        per_symbol: bool = True,
        symbol_level: Union[int, str, None] = "symbol",
        drop_all_nan_rows: bool = False,
    ) -> pd.DataFrame:
        """
        Compute all indicators and return a features DataFrame aligned to df.index.

        - per_symbol: if df has a MultiIndex, compute per symbol to avoid cross-leakage.
        - symbol_level: name or index of the symbol level in a MultiIndex.
        - drop_all_nan_rows: drop rows where all features are NaN (e.g., warmup bars).
        """

        if isinstance(df.index, pd.MultiIndex) and per_symbol:
            level = self._resolve_symbol_level(df, symbol_level)
            feats = (
                df.groupby(level=level, group_keys=False, sort=False)
                .apply(self._compute_for_single)
            )
        else:
            feats = self._compute_for_single(df)

        if drop_all_nan_rows:
            feats = feats.dropna(how="all")
        return feats

    def _resolve_symbol_level(
        self, df: pd.DataFrame, symbol_level: Union[int, str, None]
    ) -> int:
        if isinstance(symbol_level, int):
            return symbol_level
        if symbol_level is None:
            return 0
        if symbol_level in (df.index.names or []):
            return (df.index.names or []).index(symbol_level)
        # Fallback to the first level
        return 0

    def _compute_for_single(self, df: pd.DataFrame) -> pd.DataFrame:
        outputs: List[pd.DataFrame] = []
        for spec in self.specs:
            fn = spec.fn
            params = spec.params or {}
            out = fn(df, **params)
            shift = self.default_shift if spec.shift is None else spec.shift
            dtype = self.default_dtype if spec.dtype is None else spec.dtype

            if isinstance(out, pd.Series):
                suffix = out.name if out.name is not None else "value"
                s = out.rename(f"{spec.name}_{suffix}")
                if shift:
                    s = s.shift(shift)
                s = s.astype(dtype)
                outputs.append(s.to_frame())
            elif isinstance(out, pd.DataFrame):
                df_out = out.copy()
                # Always prefix each column with the spec name
                df_out.columns = [f"{spec.name}_{c}" for c in df_out.columns]
                if shift:
                    df_out = df_out.shift(shift)
                df_out = df_out.astype(dtype)
                outputs.append(df_out)
            else:
                raise TypeError(
                    f"Indicator '{spec.name}' must return Series or DataFrame, got {type(out)}"
                )

        if not outputs:
            return pd.DataFrame(index=df.index)
        return pd.concat(outputs, axis=1)

