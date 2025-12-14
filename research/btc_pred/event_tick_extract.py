

# %%
import pandas as pd
import numpy as np

# %%
def lookback_features(g, close_bar_price):
        vwap = (g["price"] * g["size"]).sum() / g["size"].sum()
        qty = g["size"].sum()
        sell_vol_percent = (g["size"] * g["isBuyerMaker"].astype(int)).sum() / qty
        return pd.Series({
            "vwap": vwap,
            "sell_vol_percent": sell_vol_percent * 100,
            "qty": qty,
            "ret_in_bucket": 100 * (g["price"].iloc[-1] / g["price"].iloc[0] - 1),
            "ret_to_closebar": 100 * (vwap / close_bar_price - 1),
        })


def lookforward_target(
    values: list[float],
    base_value: float,
    up_percent: float,
    down_percent: float,
) -> int:
    """
    Returns:
        0  -> neither barrier touched
        1  -> upper barrier hits first (or lower never hits)
       -1  -> lower barrier hits first (or upper never hits)
    """
    upper = base_value * (1 + up_percent)
    lower = base_value * (1 - down_percent)

    upper_hit_idx = None
    lower_hit_idx = None

    for idx, price in enumerate(values):
        if upper_hit_idx is None and price >= upper:
            upper_hit_idx = idx
        if lower_hit_idx is None and price <= lower:
            lower_hit_idx = idx
        if upper_hit_idx is not None and lower_hit_idx is not None:
            break  # both found; order will be checked after loop

    if upper_hit_idx is None and lower_hit_idx is None:
        return 0
    if upper_hit_idx is None:
        return -1
    if lower_hit_idx is None:
        return 1
    return 1 if upper_hit_idx < lower_hit_idx else -1


def generate_inv_ts(ts_list: list[pd.Timestamp]) -> list[pd.Timestamp]:
    ts_offsets = pd.to_timedelta(np.arange(60, 130, 10), unit="s")
    return sorted({ts + offset for ts in ts_list for offset in ts_offsets})



def extract_pred(df_aggTrades: pd.DataFrame,
                         tl_inv: list[pd.Timestamp],
                         lookback_minutes: int = 2,
                         lookback_trades: int = 1000,
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:

    ts_arr = df_aggTrades["ts"].to_numpy(dtype="datetime64[ns]")
    feature_rows = []

    for i, cur_ts in enumerate(tl_inv):
        if i % 1000 == 0:
            print(f"Processing timestamp: {i} of {len(tl_inv)}")
        
        # consider the lookback LOOKBACK_MINUTES minutes,
        #  i.e  cur_ts - LOOKBACK_MINUTES minutes < t <= cur_ts
        i0 = ts_arr.searchsorted(np.datetime64(cur_ts - pd.Timedelta(minutes=lookback_minutes)), side="right")
        i1 = ts_arr.searchsorted(np.datetime64(cur_ts), side="right")
        i2 = i1 - lookback_trades


        # create the 10s group number for window 1, , used for groupby later
        grp_key_10s = df_aggTrades.iloc[i0:i1].groupby(pd.Grouper(key="ts", freq="10s")).ngroup()
        

        # create the 100 trades group number for window 2, used for groupby later
        grp_key_100trades = np.arange(LOOKBACK_TRADES) // 100
        
        # last seen price is the price of the last trade in the window
        # used as the baseline for lookback and lookforward price in each bucket
        last_seen_price = df_aggTrades.iloc[i1-1]["price"]
        
        # create the 10s group number for window 1
        df_10s = (
            df_aggTrades.iloc[i0:i1]
            .groupby(grp_key_10s, sort=False)
            .apply(lambda g: lookback_features(g, last_seen_price), include_groups=False)
        )

        df_100trades = (
            df_aggTrades.iloc[i2:i1]
            .groupby(grp_key_100trades, sort=False)
            .apply(lambda g: lookback_features(g, last_seen_price), include_groups=False)
        )
        
        #adding appropriate prefix for the column names and flattening the dataframe
        df_10s = df_10s.add_prefix("lookback10s_")
        df_10s_flat = df_10s.stack()
        df_10s_flat.index = [f"{m}_{b}" for (b, m) in df_10s_flat.index]
        df_10s_flat = df_10s_flat.to_dict()
        
        df_100trades = df_100trades.add_prefix("lookback100trades_")
        df_100trades_flat = df_100trades.stack()
        df_100trades_flat.index = [f"{m}_{b}" for (b, m) in df_100trades_flat.index]
        df_100trades_flat = df_100trades_flat.to_dict()
        
        
        # creating and appending the feature row
        feature_row = {"ts": cur_ts, **df_10s_flat, **df_100trades_flat}
        feature_rows.append(feature_row)


    tick_features = (
        pd.DataFrame(feature_rows)
        .set_index("ts")
        .sort_index()
    )

    return tick_features, tick_targets

def extract_pred(df_aggTrades: pd.DataFrame,
                tl_inv: list[pd.Timestamp],
                lookforward_minutes: int = 2,
                up_percent: float = 0.002,
                down_percent: float = 0.002,
                ) -> tuple[pd.DataFrame, pd.DataFrame]:

    ts_arr = df_aggTrades["ts"].to_numpy(dtype="datetime64[ns]")
    target_rows = []

    for i, cur_ts in enumerate(tl_inv):
        if i % 1000 == 0:
            print(f"Processing timestamp: {i} of {len(tl_inv)}")
        
        # consider the lookback lookforward_minutes miutes,
        #  i.e  cur_ts < t <= cur_ts + lookforward_minutes
        i1 = ts_arr.searchsorted(np.datetime64(cur_ts), side="right")
        i3 = ts_arr.searchsorted(np.datetime64(cur_ts + pd.Timedelta(minutes=lookforward_minutes)), side="right")

        # last seen price is the price of the last trade in the window
        # used as the baseline for lookback and lookforward price in each bucket
        last_seen_price = df_aggTrades.iloc[i1-1]["price"]
        
        
        # creating and appending the target row
        target_row = {
            "ts": cur_ts, 
            "target": lookforward_target(df_aggTrades.iloc[i1:i3]["price"], base_value=last_seen_price,
                        up_percent=up_percent, down_percent=down_percent)
        }
        target_rows.append(target_row)

    tick_targets = (
        pd.DataFrame(target_rows)
        .set_index("ts")
        .sort_index()
    )

    return tick_targets


# %%
if __name__ == "__main__":
    
    # loading klines data and set timestamp as index
    df = pd.read_csv("DATA/organized/BTCUSDT/klines_1m.csv")
    df["ts"] = pd.to_datetime(df["ts"])
    df.set_index("ts", inplace=True, drop=True)
    
    # loading aggTrades data and set timestamp as index
    df_aggTrades = pd.read_parquet("DATA/organized/BTCUSDT/aggTrades.parquet")
    df_aggTrades["ts"] = pd.to_datetime(df_aggTrades["ts"])
    df_aggTrades['sell_size'] = df_aggTrades['size'] * df_aggTrades['isBuyerMaker']
    df_aggTrades['buy_size'] = df_aggTrades['size'] * (1 - df_aggTrades['isBuyerMaker'])


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


    df['polynomial2_coeff'] = polynomial_coeff(df, window=4)

    print(f'Total minibars {df.shape[0]}')
    print(f'Total minibars after filtering {df.shape[0] - sum(df.polynomial2_coeff.isna())}')
    print(f'The left 10 percent quantile is {df.polynomial2_coeff.quantile(0.1)}')

    df['polynomial2_coeff_bool'] = df.polynomial2_coeff < df.polynomial2_coeff.quantile(0.1)
    print(f'Total minibars after filtering and left quantile  {sum(df['polynomial2_coeff_bool'])}')


    tl_inv = generate_inv_ts(df[df['polynomial2_coeff_bool']].index.tolist())

    LOOKBACK_MINUTES = 2
    LOOKBACK_TRADES = 1000
    LOOKFORWARD_MINUTES = 2
    UP_PERCENT = 0.002
    DOWN_PERCENT = 0.002

    tick_features = extract_pred(df_aggTrades,
                                tl_inv,
                                lookback_minutes=LOOKBACK_MINUTES,
                                lookback_trades=LOOKBACK_TRADES)

    tick_targets = extract_pred(df_aggTrades,
                                tl_inv,
                                lookforward_minutes=LOOKFORWARD_MINUTES,
                                up_percent=UP_PERCENT,
                                down_percent=DOWN_PERCENT)
    
    tick_features.to_parquet("DATA/organized/BTCUSDT/event_tick_features.parquet")
    tick_targets.to_parquet("DATA/organized/BTCUSDT/event_tick_targets.parquet")
