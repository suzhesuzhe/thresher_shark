


def agg_to_kmin(df, k=5):
    """
    Aggregate 1-minute OHLCV data to k-minute bars.
    
    Parameters:
    df: DataFrame with DatetimeIndex and OHLCV columns
    k: Integer, number of minutes to aggregate (default 5)
    
    Returns:
    DataFrame with k-minute aggregated OHLCV data
    """
    
    # Create the resampling rule (e.g., '5T' for 5 minutes)
    rule = f'{k}min'
    
    # Aggregate the data
    agg_data = df.resample(rule).agg({
        'Open': 'first',        # First open price in the period
        'High': 'max',          # Highest price in the period  
        'Low': 'min',           # Lowest price in the period
        'Close': 'last',        # Last close price in the period
        'Volume': 'sum',        # Sum of volume
        'transactions': 'sum'   # Sum of transactions
    })
    
    # Calculate volume-weighted VWAP for each k-minute period
    def calculate_vwap(group):
        if group['Volume'].sum() == 0:
            return group['Vwap'].mean()  # Fallback to simple average if no volume
        return (group['Vwap'] * group['Volume']).sum() / group['Volume'].sum()
    
    # Apply VWAP calculation to each resampled group
    agg_data['Vwap'] = df.resample(rule).apply(calculate_vwap)
    
    # Remove any rows where all OHLC values are NaN (no data in that period)
    agg_data = agg_data.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    # Ensure High >= Low and handle any edge cases
    agg_data['High'] = agg_data[['Open', 'High', 'Low', 'Close']].max(axis=1)
    agg_data['Low'] = agg_data[['Open', 'High', 'Low', 'Close']].min(axis=1)
    
    print(f"Original 1-minute bars: {len(df)}")
    print(f"Aggregated {k}-minute bars: {len(agg_data)}")
    print(f"Compression ratio: {len(df) / len(agg_data):.1f}:1")
    
    return agg_data