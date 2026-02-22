"""Polygon.io data fetcher for ETF prices and technical indicators.

Fetches OHLCV data plus RSI, MACD, EMA, and Bollinger Bands for each
ticker in the ETF universe, then saves a merged CSV.

Requires POLYGONIO_API_KEY in a .env file or environment variable.
"""
import os
import pandas as pd
from datetime import datetime, timedelta
import requests
from time import sleep
import numpy as np
from polygon.rest import RESTClient

from dotenv import load_dotenv
load_dotenv()
    
api_key = os.getenv("POLYGONIO_API_KEY")
assert api_key, "Please set POLYGONIO_API_KEY in your environment"

# Helper functions for fetching technical indicators from Polygon.io
# I use polygon, use whatever you like

def fetch_rsi_data(client, symbol, start_date=None, window=14):
    try:
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
            days_needed = (datetime.now() - start_dt).days + 100
            limit = min(max(days_needed, 3000), 50000)
        else:
            limit = 5000
        rsi_data = client.get_rsi(
            ticker=symbol,
            timespan="day",
            adjusted=True,
            window=window,
            series_type="close",
            order="desc",
            limit=limit
        )
        if hasattr(rsi_data, 'values') and rsi_data.values:
            df = pd.DataFrame([{
                'date': pd.to_datetime(val.timestamp, unit='ms'),
                'rsi_14': val.value
            } for val in rsi_data.values])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def fetch_macd_data(client, symbol, start_date=None, short_window=12, long_window=26, signal_window=9):
    try:
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
            days_needed = (datetime.now() - start_dt).days + 100
            limit = min(max(days_needed, 3000), 50000)
        else:
            limit = 5000
        macd_data = client.get_macd(
            ticker=symbol,
            timespan="day",
            adjusted=True,
            short_window=short_window,
            long_window=long_window,
            signal_window=signal_window,
            series_type="close",
            order="desc",
            limit=limit
        )
        if hasattr(macd_data, 'values') and macd_data.values:
            df = pd.DataFrame([{
                'date': pd.to_datetime(val.timestamp, unit='ms'),
                'macd': val.value,
                'signal': val.signal,
                'macd_diff': val.histogram
            } for val in macd_data.values])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def fetch_ema_data(client, symbol, start_date=None, window=20):
    try:
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
            days_needed = (datetime.now() - start_dt).days + 100
            limit = min(max(days_needed, 3000), 50000)
        else:
            limit = 5000
        ema_data = client.get_ema(
            ticker=symbol,
            timespan="day",
            adjusted=True,
            window=window,
            series_type="close",
            order="desc",
            limit=limit
        )
        if hasattr(ema_data, 'values') and ema_data.values:
            df = pd.DataFrame([{
                'date': pd.to_datetime(val.timestamp, unit='ms'),
                'ema_20': val.value
            } for val in ema_data.values])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def calculate_data_start_date(target_start_date: str, buffer_days: int = 60) -> str:
    target_date = datetime.strptime(target_start_date, '%Y-%m-%d')
    actual_start = target_date - timedelta(days=buffer_days)
    return actual_start.strftime('%Y-%m-%d')

def fetch_and_prepare_data(
    api_key: str,
    symbols: list,
    start_date: str = '2013-01-01',
    end_date: str = None,
    out_path: str = None,
    rate_limit: float = 1.0
):

    # Download daily price data (OHLCV) and technical indicators from Polygon.io API using the official SDK, and save to a single CSV file. The output CSV will have a multi-level index for columns: (symbol, feature).
    
    # For technical indicators, we fetch extra data from before start_date to ensure indicators are properly calculated from the target start date onward.

    if out_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(script_dir, 'etf_data_with_indicators.csv')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # Calculate buffer start date for technical indicators
    # MACD needs ~26 days for long EMA + signal calculation, so we use 60 days buffer to be safe
    target_start = datetime.strptime(start_date, '%Y-%m-%d')
    buffer_start = target_start - timedelta(days=60)  # Start from ~November 2012 for 2013-01-01 target
    buffer_start_str = buffer_start.strftime('%Y-%m-%d')
    
    print(f"Target date range: {start_date} to {end_date}")
    print(f"Fetching with buffer from: {buffer_start_str}")
    print(f"Filtering to: {start_date} onward")

    all_assets_data = {}
    client = RESTClient(api_key)

    print(f"Symbols: {symbols}")

    # Fetch data for each symbol
    for symbol in symbols:
        print(f"\nProcessing {symbol}")
        # 1. Fetch OHLCV data (use buffer start date)
        print(f"Fetching OHLCV for {symbol}")
        try:
            start_dt = datetime.strptime(buffer_start_str, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
            days_needed = (end_dt - start_dt).days + 100
            ohlcv_limit = min(max(days_needed, 3000), 50000)
            aggs = list(client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=buffer_start_str,
                to=end_date,
                adjusted=True,
                sort="asc",
                limit=ohlcv_limit
            ))
            if not aggs:
                print(f"No OHLCV results for {symbol}")
                continue
            df = pd.DataFrame([{
                'date': pd.to_datetime(a.timestamp, unit='ms'),
                'open': a.open,
                'high': a.high,
                'low': a.low,
                'close': a.close,
                'volume': a.volume,
            } for a in aggs])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            print(f"OHLCV shape: {df.shape}")
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            continue
        sleep(rate_limit)
        # 2. Fetch RSI
        print(f"Fetching RSI for {symbol}")
        rsi_df = fetch_rsi_data(client, symbol, buffer_start_str)
        if not rsi_df.empty:
            df = df.join(rsi_df, how='left')
            print(f"RSI shape: {rsi_df.shape}")
        else:
            df['rsi_14'] = np.nan
            print("RSI failed, filling with NaN")
        sleep(rate_limit)
        # 3. Fetch MACD
        print(f"Fetching MACD for {symbol}")
        macd_df = fetch_macd_data(client, symbol, buffer_start_str)
        if not macd_df.empty:
            df = df.join(macd_df, how='left')
            print(f"MACD shape: {macd_df.shape}")
        else:
            df['macd'] = np.nan
            df['signal'] = np.nan
            df['macd_diff'] = np.nan
            print("MACD failed, filling with NaN")
        sleep(rate_limit)
        # 4. Fetch EMA
        print(f"Fetching EMA for {symbol}")
        ema_df = fetch_ema_data(client, symbol, buffer_start_str)
        if not ema_df.empty:
            df = df.join(ema_df, how='left')
            print(f"EMA shape: {ema_df.shape}")
        else:
            df['ema_20'] = np.nan
            print("EMA failed, filling with NaN")
        df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
        all_assets_data[symbol] = df
        print(f"Completed {symbol}. Shape: {df.shape}")
        sleep(rate_limit)

    if not all_assets_data:
        print("No data was fetched for any symbols.")
        return None

    # Combine all assets
    print(f"\nCombining Data")
    all_indicators_df = pd.DataFrame()
    for symbol, df in all_assets_data.items():
        if all_indicators_df.empty:
            all_indicators_df = df
        else:
            all_indicators_df = all_indicators_df.join(df, how='outer')

    # Sort by date
    all_indicators_df.sort_index(inplace=True)
    
    # NOW FILTER TO TARGET DATE RANGE (remove buffer period)
    print(f"\nFiltering to Target Date Range")
    print(f"Pre-filter shape: {all_indicators_df.shape}")
    print(f"Date range: {all_indicators_df.index.min()} to {all_indicators_df.index.max()}")
    
    # Filter to only include data from the target start date onward
    target_start_dt = pd.to_datetime(start_date)
    all_indicators_df = all_indicators_df[all_indicators_df.index >= target_start_dt]
    
    print(f"Post-filter shape: {all_indicators_df.shape}")
    print(f"Final date range: {all_indicators_df.index.min()} to {all_indicators_df.index.max()}")
    
    # Save to CSV
    all_indicators_df.to_csv(out_path)
    print(f"\nSaved data to: {out_path}")
    print(f"Final dataset shape: {all_indicators_df.shape}")
    print(f"Columns: {list(all_indicators_df.columns.get_level_values(1).unique())}")
    
    return all_indicators_df

def load_and_preprocess_data(data_path, etfs=None, features_to_keep=None):
    
    """
    Args:
        data_path (str): The path to the CSV file.
        etfs (list, optional): A list of ETF tickers to load. 
                               If None, it will infer from the CSV.
        features_to_keep (list, optional): List of features to keep per asset.
            Default: ["open", "high", "low", "close", "volume", "rsi_14", "macd", "signal", "macd_diff"]

    Returns:
        np.ndarray: The preprocessed data as a numpy array of shape 
                    (timesteps, n_assets, n_features).
    """

    if features_to_keep is None:
        features_to_keep = ["open", "high", "low", "close", "volume", "rsi_14", "macd", "signal", "macd_diff"]

    # Load the data with a multi-level header
    df = pd.read_csv(data_path, header=[0, 1], index_col=0, parse_dates=True)
    
    # If ETFs are not specified, get them from the columns
    if etfs is None:
        etfs = df.columns.get_level_values(0).unique().tolist()

    # Only keep selected features
    valid_cols = [(etf, feat) for etf in etfs for feat in features_to_keep if (etf, feat) in df.columns]
    df = df[valid_cols]

    # Fill missing values (forward fill, then backward fill)
    df = df.ffill().bfill()

    # Drop columns that are still all NaN
    df = df.dropna(axis=1, how='all')

    # Normalize each feature per asset (z-score normalization)
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - mean) / std
        else:
            df[col] = 0.0

    # Replace any remaining NaN/Inf with zero
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    # Build features array
    features_list = []
    for etf in etfs:
        etf_cols = [col for col in df.columns if col[0] == etf]
        if etf_cols:
            features_list.append(df[etf_cols].values)
    
    if features_list:
        features = np.stack(features_list, axis=1)
        return features
    print("No valid features found for any ETF")
    return np.array([])

def validate_data_quality(df, symbols):

    """    
    Args:
        df (pd.DataFrame): The combined dataframe with multi-level columns
        symbols (list): List of symbols that were requested
    
    Returns:
        dict: Data quality summary
    """
    summary = {
        'total_symbols_requested': len(symbols),
        'symbols_with_data': 0,
        'date_range': None,
        'missing_data_summary': {},
        'data_completeness': {}
    }
    
    if df.empty:
        summary['symbols_with_data'] = 0
        return summary
    
    # Get date range
    summary['date_range'] = {
        'start': df.index.min(),
        'end': df.index.max(),
        'total_days': len(df)
    }
    
    # Check data completeness per symbol
    for symbol in symbols:
        symbol_cols = [col for col in df.columns if col[0] == symbol]
        if symbol_cols:
            summary['symbols_with_data'] += 1
            symbol_df = df[symbol_cols]
            
            # Calculate missing data percentage for each column
            missing_pct = symbol_df.isnull().sum() / len(symbol_df) * 100
            summary['missing_data_summary'][symbol] = missing_pct.to_dict()
            
            # Overall completeness for this symbol (only consider core OHLCV data complete)
            core_cols = [col for col in symbol_cols if col[1] in ['open', 'high', 'low', 'close', 'volume']]
            if core_cols:
                core_df = symbol_df[core_cols]
                overall_completeness = (1 - core_df.isnull().any(axis=1).sum() / len(core_df)) * 100
            else:
                overall_completeness = 0
            summary['data_completeness'][symbol] = overall_completeness
    
    return summary

if __name__ == '__main__':
    
    # Full ETF universe for portfolio optimization
    etfs = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ', 'TLT', 'IEF', 'GLD', 'USO']
    
    print("="*60)
    print("FETCHING ETF DATA WITH TECHNICAL INDICATORS")
    print("="*60)
    
    # Fetch full historical data from 2013-01-01 to present
    result_df = fetch_and_prepare_data(
        api_key, 
        etfs, 
        start_date='2013-01-01'
        # end_date defaults to today
    )
    
    if result_df is not None and not result_df.empty:
        # Validate data quality
        quality_summary = validate_data_quality(result_df, etfs)
        
        print("\n" + "="*60)
        print("DATA QUALITY SUMMARY")
        print("="*60)
        print(f"Symbols requested: {quality_summary['total_symbols_requested']}")
        print(f"Symbols with data: {quality_summary['symbols_with_data']}")
        
        if quality_summary['date_range']:
            print(f"Date range: {quality_summary['date_range']['start']} to {quality_summary['date_range']['end']}")
            print(f"Total trading days: {quality_summary['date_range']['total_days']}")
        
        print(f"\nData completeness by symbol:")
        for symbol, completeness in quality_summary['data_completeness'].items():
            print(f"  {symbol}: {completeness:.1f}%")
        
        print(f"\nDataset shape: {result_df.shape}")
        print(f"Columns: {result_df.columns.get_level_values(1).unique().tolist()}")
        
        # Test data loading
        print(f"\nTesting data preprocessing...")
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(script_dir, 'etf_data_with_indicators.csv')
            features = load_and_preprocess_data(data_path, etfs)
            print(f"Preprocessed features shape: {features.shape}")
            print("Data preprocessing successful!")
        except Exception as e:
            print(f"Data preprocessing failed: {e}")
    else:
        print("No data was fetched.")