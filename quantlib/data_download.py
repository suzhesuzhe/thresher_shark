
# ================================
# Global Parameters & Imports
# ================================


import os
from pathlib import Path
import requests
import pandas as pd
import numpy as np
import pytz
import math
import time
from datetime import datetime, timedelta
from IPython.display import display, Markdown

# ===== Configuration =====
def _load_api_key():
    """Load Polygon API key from env or common file locations.

    Priority:
    - Environment variable `POLYGON_API_KEY`
    - Files relative to project root (parent of this file's directory):
      - `api_key.txt`
      - `secrets/api_key.txt`
      - `research/secrets/api_key.txt`
    - User config locations:
      - `~/.polygon/api_key`
      - `~/.config/polygon/api_key`
    """
    # 1) Environment variable
    env_key = os.getenv("POLYGON_API_KEY")
    if env_key:
        return env_key.strip()

    # 2) Project-root based files
    try:
        project_root = Path(__file__).resolve().parent.parent  # repo root (parent of quantlib)
    except Exception:
        project_root = Path.cwd()

    candidate_paths = [
        project_root / "api_key.txt",
        project_root / "secrets" / "api_key.txt",
        project_root / "research" / "secrets" / "api_key.txt",
        Path.home() / ".polygon" / "api_key",
        Path.home() / ".config" / "polygon" / "api_key",
    ]

    for p in candidate_paths:
        try:
            if p.exists():
                return p.read_text().strip()
        except Exception:
            continue

    raise RuntimeError(
        "Polygon API key not found. Set POLYGON_API_KEY env var or place api_key.txt at project root (or ~/.polygon/api_key)."
    )

API_KEY = _load_api_key()

# Set to True if you have a paid Polygon subscription; otherwise, set to False.
PAID_POLYGON_SUBSCRIPTION = False




# ================================
# Data Download Functions
# ================================

def get_polygon_data(ticker, from_date, to_date, timespan,  
                    multiplier=1, adjusted=False, url=None):
    """Retrieve intraday aggregate data from Polygon.io."""
    if url is None:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": "true" if adjusted else "false",
            "sort": "asc",
            "limit": 50000,
            "apiKey": API_KEY
        }
        response = requests.get(url, params=params)
    else:
        if "apiKey" not in url:
            url = f"{url}&apiKey={API_KEY}" if "?" in url else f"{url}?apiKey={API_KEY}"
        response = requests.get(url)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None, None

    data = response.json()
    next_url = data.get("next_url")
    if next_url and "apiKey" not in next_url:
        next_url = f"{next_url}&apiKey={API_KEY}" if "?" in next_url else f"{next_url}?apiKey={API_KEY}"

    return data.get("results", []), next_url




def download_polygon_data(ticker, from_date, to_date, output_file,
                          timespan, multiplier=1, adjusted=False):
    """
    Download intraday and daily adjusted data, merge them, and save to CSV.

    If PAID_POLYGON_SUBSCRIPTION is False, simply warn or stop if START_DATE is older than 2 years.
    """
    if not PAID_POLYGON_SUBSCRIPTION:
        two_years_ago = datetime.now() - timedelta(days=730)
        start_dt = datetime.strptime(from_date, '%Y-%m-%d')
        if start_dt < two_years_ago:
            print("ERROR: For free Polygon subscriptions, START_DATE must be within the past 2 years.")
            return None

    # Download all data first, then process it all at once
    all_raw_data = []
    next_url = None

    batch_count = 0
    print(f"Downloading intraday data for {ticker} from {from_date} with timespan {timespan}...")

    while True:
        batch_count += 1
        print(f"Batch {batch_count}...")
        results, next_url = get_polygon_data(ticker, from_date, to_date,
                           timespan, multiplier=1, adjusted=adjusted, url=next_url)
        if not results:
            print("No more data.")
            break

        # Just add raw results to our collection, don't process yet
        all_raw_data.extend(results)
        print(f"Batch {batch_count}: Retrieved {len(results)} records")

        if not next_url:
            print("Download complete.")
            break

        if not PAID_POLYGON_SUBSCRIPTION:
            # Enforce a rate limit: 5 requests per minute (sleep for 12 seconds)
            time.sleep(12)

    if all_raw_data:
        # Now process all data at once
        print(f"Processing {len(all_raw_data)} total records...")
        final_df = pd.DataFrame(all_raw_data)
        final_df = final_df.rename(columns={
        'v': 'Volume',
        'vw': 'Vwap',
        'o': 'Open',
        'c': 'Close',
        'h': 'High',
        'l': 'Low',
        't': 'timestamp_ms',
        'n': 'transactions'
        })
        final_df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        return final_df
    else:
        print("No data collected.")
        return None

# check how many data points in a day, get a sense of time discontinuity
def plot_daily_row_counts(df, title_suffix=None):
    """
    Plot the number of rows (data points) for each day in the dataset.
    
    Parameters:
    df: DataFrame with 'date' column containing date information
    title_suffix: String to add to the plot title
    
    Returns:
    matplotlib figure showing daily row counts
    """
    import matplotlib.pyplot as plt
    
    # Group by date and count rows
    daily_counts = df.groupby('date').size().reset_index(name='row_count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    daily_counts = daily_counts.sort_values('date')
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    plt.plot(daily_counts['date'], daily_counts['row_count'], linewidth=0.8, alpha=0.7)
    plt.title(f'Number of Data Points (Rows) per Trading Day - {title_suffix}', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Rows', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\nSummary Statistics for {title_suffix}:")
    print(f"Mean rows per day: {daily_counts['row_count'].mean():.1f}")
    print(f"Median rows per day: {daily_counts['row_count'].median():.1f}")
    print(f"Min rows per day: {daily_counts['row_count'].min()}")
    print(f"Max rows per day: {daily_counts['row_count'].max()}")
    print(f"Total days: {len(daily_counts)}")
    print(f"Date range: {daily_counts['date'].min().date()} to {daily_counts['date'].max().date()}")
    
    plt.show()
    
    return daily_counts

def process_date(df, fill_missing_minutes=False):
    """
    Process intraday data with datetime conversion and market hours filtering.
    Takes a DataFrame with 't' column (timestamp in ms) and processes it.
    Fills missing minutes with adjacent close prices.
    """
    # Convert timestamp to datetime columns
    df['datetime_utc'] = pd.to_datetime(df['timestamp_ms'], unit='ms')

    # Extract date information
    df['date'] = df['datetime_utc'].dt.date

    # Set timestamp as index
    df.set_index('datetime_utc', inplace=True)
    
    if fill_missing_minutes:
        # Create complete minute range from start to end
        start_time = df.index.min()
        end_time = df.index.max()
        complete_range = pd.date_range(start=start_time, end=end_time, freq='1min')
        
        # Reindex to fill missing minutes
        df_complete = df.reindex(complete_range)
        
        # Forward fill Close prices for missing minutes
        df_complete['Close'] = df_complete['Close'].ffill()
        
        # For missing minutes, set OHLV using the forward-filled close price
        missing_mask = df_complete['Open'].isna()
        df_complete.loc[missing_mask, 'Open'] = df_complete.loc[missing_mask, 'Close']
        df_complete.loc[missing_mask, 'High'] = df_complete.loc[missing_mask, 'Close']
        df_complete.loc[missing_mask, 'Low'] = df_complete.loc[missing_mask, 'Close']
        df_complete.loc[missing_mask, 'Vwap'] = df_complete.loc[missing_mask, 'Close']
        
        # Set Volume and transactions to 0 for missing minutes
        df_complete.loc[missing_mask, 'Volume'] = 0
        df_complete.loc[missing_mask, 'transactions'] = 0
        
        # Update timestamp_ms for missing minutes
        df_complete.loc[missing_mask, 'timestamp_ms'] = (df_complete.index[missing_mask].astype('int64') // 10**6).astype('int64')
        
        # Update date for missing minutes
        df_complete.loc[missing_mask, 'date'] = df_complete.index[missing_mask].date

    else:
        df_complete = df
    
    # Ensure consistent column ordering
    standard_columns = ['timestamp_ms', 'Open', 'High', 'Low', 'Close', 'Volume', 'Vwap', 'transactions', 'date']
    available_columns = [col for col in standard_columns if col in df_complete.columns]
    remaining_columns = [col for col in df_complete.columns if col not in standard_columns]
    df_complete = df_complete[available_columns + remaining_columns]

    print(f"Original data: {len(df)} rows")
    print(f"Complete data: {len(df_complete)} rows")
    print(f"Missing minutes filled: {len(df_complete) - len(df)}")

    return df_complete
