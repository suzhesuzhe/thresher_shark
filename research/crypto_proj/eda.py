

#%%
# Loading required packages 
import requests
import pandas as pd
import numpy as np
import pytz


# %% [markdown]
## Download Datat From Polygon and Preprocess


#%%
#Loading self-edited data_download functions
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from quantlib.data_download import *

#%%
#Configure Ticker, Date Range and Other global variable
TICKER = "X:DOGEUSD"
START_DATE = "2023-09-15"
END_DATE = "2025-09-11"
OUTPUT_FILE_INTRADAY = "./DATA/doge_OHCL_1m_20230915_20250911.csv"

#utc_tz = pytz.timezone('UTC')
#nyc_tz = pytz.timezone('America/New_York')


#%%
# Downloading or Loading into Memory
if os.path.exists(OUTPUT_FILE_INTRADAY):
    print(f"Loading existing file: {OUTPUT_FILE_INTRADAY}")
    data_intraday = pd.read_csv(OUTPUT_FILE_INTRADAY)
else:
    print(f"File not found. Downloading data...")
    data_intraday = download_polygon_data(ticker=TICKER, from_date=START_DATE, to_date=END_DATE, 
                                 output_file=OUTPUT_FILE_INTRADAY,
                                 multiplier=1, timespan='minute', adjusted=False)


#%%
# check how many data points in a day, check potential time discontinuity
doge_intraday_processed_nofill = process_date(data_intraday, fill_missing_minutes=False)
days_count = plot_daily_row_counts(doge_intraday_processed_nofill, TICKER)



#%%
#Found the data has time discontinuity, missing minute rows, fill
doge_intraday_processed_fill = process_date(data_intraday, fill_missing_minutes=True)
doge_intraday_processed_fill.to_csv('./DATA/doge_OHCL_1m_20230915_20250911_processed_fill.csv')



# %% [markdown]
## Generate lookback and lookforward features

# %%
# If needed, aggregate 1min bar data to kmin bar data
from quantlib.utils import agg_to_kmin
doge_1min = pd.read_csv('./DATA/doge_intraday_processed_fill.csv', index_col=0, parse_dates=True)
#doge_5min = agg_to_kmin(doge_1min, k=5)




# %% 
# Loading required functions for building features
from quantlib.var_pipeline import *
from quantlib.look_forward_vars import *
from quantlib.look_back_vars import *


# %%
# Build a small indicator set; add/remove specs as needed.

lookback_specs = [
    # Trend
    VariableSpec(name="sma", fn=pta_sma, params={"length": 14}),
    VariableSpec(name="ema", fn=pta_ema, params={"length": 20}),
    VariableSpec(name="ema", fn=pta_ema, params={"length": 50}),
    VariableSpec(name="ema", fn=pta_ema, params={"length": 200}),
    VariableSpec(name="macd", fn=pta_macd, params={"fast": 12, "slow": 26, "signal": 9}),
    VariableSpec(name="macd_scaled", fn=macd_atr_scaled, params={"fast": 12, "slow": 26, "signal": 9, "lag": 12}),
    VariableSpec(name="ppo", fn=pta_ppo, params={"fast": 12, "slow": 26, "signal": 9}),
    # Past returns
    VariableSpec(name="roc", fn=pta_roc, params={"length": 1}),
    VariableSpec(name="roc", fn=pta_roc, params={"length": 5}),
    VariableSpec(name="roc", fn=pta_roc, params={"length": 10}),
    VariableSpec(name="roc", fn=pta_roc, params={"length": 20}),
    VariableSpec(name="ret_moments", fn=returns_rolling_moments, params={"window": 20}),
    VariableSpec(name="ret_autocorr", fn=returns_autocorr, params={"lag": 1, "window": 100}),
    # Momentum
    VariableSpec(name="rsi", fn=pta_rsi, params={"length": 14}),
    VariableSpec(name="stoch", fn=pta_stoch, params={"k": 14, "d": 3, "smooth_k": 3}),
    VariableSpec(name="stochrsi", fn=pta_stochrsi, params={"length": 14, "rsi_length": 14, "k": 3, "d": 3}),
    VariableSpec(name="adx", fn=pta_adx, params={"length": 14}),
    # Volatility
    VariableSpec(name="atr", fn=pta_atr, params={"length": 14}),
    VariableSpec(name="bbw", fn=pta_bbw, params={"length": 20, "std": 2.0}),
    # Volume
    VariableSpec(name="obv", fn=pta_obv, params={} ),
    VariableSpec(name="mfi", fn=mfi_custom, params={"length": 14}),
    #Customized
    VariableSpec(name="diff_close_legendre_fit", fn=diff_close_legendre_fit, params={"n": 20}),
    VariableSpec(name="cmma", fn=cmma, params={"n": 20, "c": 0.5, "atr_length": 14}), 
    VariableSpec(name="legendre_trend_r2_atr_scaled", fn=legendre_trend_r2_atr_scaled, params={"lookback": 20, "atr_long": 100, "price_col": "Close"}),
]

lookback_pipeline = VariablePipeline(lookback_specs, default_shift=1, default_dtype="float32")


lookforward_specs = [
    VariableSpec(name="logret_fwd", fn=log_return_forward, params={"n": 5}),
    VariableSpec(name="pf_fwd", fn=profit_factor_forward, params={"n": 14}),
    VariableSpec(name="rov_fwd", fn=return_over_variance_forward, params={"n": 14}),
    VariableSpec(name="fdi_fwd", fn=forward_directional_index, params={"n": 14}),
]
lookforward_pipeline = VariablePipeline(lookforward_specs, default_shift=1, default_dtype="float32")


# Compute features for a sample slice to illustrate usage
lookback_doge_1min = lookback_pipeline.run(doge_1min)
lookforward_doge_1min = lookforward_pipeline.run(doge_1min)
#lookback_doge_5min = lookback_pipeline.run(doge_5min)
#lookforward_doge_5min = lookforward_pipeline.run(doge_5min)


#lookback_doge_1min.to_csv('./DATA/doge_1min_lookback.csv')
#lookforward_doge_1min.to_csv('./DATA/doge_1min_lookforward.csv')
#lookback_doge_5min.to_csv('./DATA/doge_5min_lookback.csv')
#lookforward_doge_5min.to_csv('./DATA/doge_5min_lookforward.csv')





# %%
from quantlib.plot_features_stat import visualize_features_report
from quantlib.plot_ochl_features import *


#%%
visualize_features_report(lookback_doge_1min, 
                          out_path="doge_lookback_raw.pdf",
                          window=1440*14,
                          step=1440*7, bins=60,
                          title="DOGE 1m Features")

#%%
visualize_features_report(lookforward_doge_1min,
                          out_path= "doge_lookforward_raw.pdf",
                          window=1440*14,
                          step=1440*7, 
                          bins=60,
                          title="DOGE 1m Features")



#%%
plot_df = pd.concat([doge_1min, lookback_doge_1min, lookforward_doge_1min], axis=1)

plot_ohlc_basic(plot_df, 
                window_bars=300,
                step_bars=120,
                indicators={'pf_fwd_14':'separate',
                            'ema_20':'overlay'},
                rules_dict=None)
