"""Lightweight ETF price loader.

Extracts close prices from the full indicator CSV (or loads a
pre-saved etf_prices.csv) and provides simple return helpers.
"""
import os
import pandas as pd

def create_etf_prices():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    etf_prices_path = os.path.join(script_dir, 'etf_prices.csv')
    if os.path.exists(etf_prices_path):
        prices_df = pd.read_csv(etf_prices_path, index_col=0, parse_dates=True)
        print(f"Loaded ETF prices: {prices_df.shape}")
        print(f"ETFs: {list(prices_df.columns)}")
        print(f"Date range: {prices_df.index.min()} to {prices_df.index.max()}")
        return prices_df
    full_data_path = os.path.join(script_dir, 'etf_data_with_indicators.csv')
    if os.path.exists(full_data_path):
        print(f"Extracting prices from full data file...")
        full_df = pd.read_csv(full_data_path, header=[0, 1], index_col=0, parse_dates=True)
        etfs = full_df.columns.get_level_values(0).unique().tolist()
        prices_data = {etf: full_df[(etf, 'close')] for etf in etfs if (etf, 'close') in full_df.columns}
        prices_df = pd.DataFrame(prices_data)
        prices_df = prices_df.ffill().bfill().dropna(how='all')
        print(f"Extracted ETF prices: {prices_df.shape}")
        print(f"ETFs: {list(prices_df.columns)}")
        print(f"Date range: {prices_df.index.min()} to {prices_df.index.max()}")
        prices_df.to_csv(etf_prices_path)
        print(f"Saved to: {etf_prices_path}")
        return prices_df
    print("No data found. Run the full data loader first.")
    return None


def get_etf_returns(prices_df=None):
    if prices_df is None:
        prices_df = create_etf_prices()
    if prices_df is None:
        return None
    returns_df = prices_df.pct_change().dropna()
    print(f"Returns calculated: {returns_df.shape}")
    print(f"Date range: {returns_df.index.min()} to {returns_df.index.max()}")
    return returns_df

def get_price_data_summary(prices_df=None):
    if prices_df is None:
        prices_df = create_etf_prices()
    if prices_df is None:
        return None
    summary = {
        'n_etfs': len(prices_df.columns),
        'n_days': len(prices_df),
        'etf_symbols': list(prices_df.columns),
        'date_range': {
            'start': prices_df.index.min(),
            'end': prices_df.index.max()
        },
        'price_stats': prices_df.describe(),
        'missing_data': prices_df.isnull().sum()
    }
    return summary

def split_price_data(prices_df=None, train_end='2022-12-31', val_end='2023-12-31'):
    if prices_df is None:
        prices_df = create_etf_prices()
    if prices_df is None:
        return None
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    train_mask = prices_df.index <= train_end_dt
    val_mask = (prices_df.index > train_end_dt) & (prices_df.index <= val_end_dt)
    test_mask = prices_df.index > val_end_dt
    split_data = {
        'train_prices': prices_df[train_mask],
        'val_prices': prices_df[val_mask],
        'test_prices': prices_df[test_mask],
        'full_prices': prices_df
    }
    print(f"Price data splits:")
    print(f"   Train: {len(split_data['train_prices'])} days ({split_data['train_prices'].index.min()} to {split_data['train_prices'].index.max()})")
    print(f"   Val:   {len(split_data['val_prices'])} days ({split_data['val_prices'].index.min()} to {split_data['val_prices'].index.max()})")
    print(f"   Test:  {len(split_data['test_prices'])} days ({split_data['test_prices'].index.min()} to {split_data['test_prices'].index.max()})")
    return split_data

if __name__ == "__main__":
    prices = create_etf_prices()
    if prices is not None:
        summary = get_price_data_summary(prices)
        print(f"Number of ETFs: {summary['n_etfs']}")
        print(f"Number of days: {summary['n_days']}")
        print(f"ETF symbols: {summary['etf_symbols']}")
        print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        returns = get_etf_returns(prices)
        if returns is not None:
            print(f"Returns shape: {returns.shape}")
            print(f"Sample returns (first 5 days):")
            print(returns.head())
        split_data = split_price_data(prices)
        print(f"Mini data loader test completed successfully.")
    else:
        print("Failed to create ETF prices data")
