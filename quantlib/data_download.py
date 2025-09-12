
# ================================
# Global Parameters & Imports
# ================================


import os
import requests
import pandas as pd
import numpy as np
import pytz
import math
import time
from datetime import datetime, timedelta
from IPython.display import display, Markdown

# ===== Configuration =====
# Please set your Polygon.io API key below:
with open('/Users/robertsu/Documents/trading/api_key.txt', 'r') as f:
    API_KEY = f.read().strip()

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




