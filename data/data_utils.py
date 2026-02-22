"""Feature engineering and data splitting utilities.

Builds the 274-dimensional state vector from raw ETF prices:
stacked normalised log returns, multi-horizon returns, RSI-14,
realised volatility, downside semi-vol, cross-sectional ranks,
mean correlation, absolute returns, and cyclical time encodings.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union
from rl_ppo.config import Config


def load_and_process_data() -> pd.DataFrame:
    """Load ETF price panel using the OPEN field (intentionally, for reproducibility).

    Rationale: Historical runs that produced the published Sharpe used the first
    column selected via a fallback path, which corresponded to the 'open' price.
    To lock in those results we now explicitly and transparently select the
    'open' field case‑insensitively for each ticker in the multi‑index CSV.

    If a ticker lacks an 'open' field we raise a clear error instead of silently
    substituting something else. If the file is ever flattened to single level
    we accept any direct column matching the ticker (assumed already open).
    """
    df = pd.read_csv(Config.ETF_DATA_PATH, header=[0, 1], index_col=0, parse_dates=True)
    df = df.sort_index()
    if not isinstance(df.columns, pd.MultiIndex):  # unexpected, but handle
        subset = {sym: df[sym] for sym in Config.ETF_TICKERS if sym in df.columns}
        if len(subset) != len(Config.ETF_TICKERS):
            missing = [s for s in Config.ETF_TICKERS if s not in subset]
            raise KeyError(f"Missing expected tickers in flat file: {missing}")
        prices = pd.DataFrame(subset).sort_index().ffill().dropna(how='all')
        return prices
    lvl0 = df.columns.get_level_values(0)
    lvl1 = df.columns.get_level_values(1)
    open_prices: dict[str, pd.Series] = {}
    for sym in Config.ETF_TICKERS:
        # Collect fields for this ticker
        fields = [f for t, f in zip(lvl0, lvl1) if t == sym]
        if not fields:
            raise KeyError(f"Ticker {sym} not found in data columns")
        # Case-insensitive match to 'open'
        match = None
        for f in fields:
            if f.lower() == 'open':
                match = f
                break
        if match is None:
            raise KeyError(f"Ticker {sym} has no 'open' field (available: {fields})")
        open_prices[sym] = df[(sym, match)]
    prices = pd.DataFrame(open_prices).sort_index().ffill().dropna(how='all')
    return prices


def create_features(
    prices: pd.DataFrame,
    stack_len: Optional[int] = None,
    norm_window: Optional[int] = None,
    fit_end_date: Optional[Union[str, pd.Timestamp]] = None,
    raw_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Components: stacked normalized log returns, multi-horizon returns, RSI, volume z-scores,
    realized vol, downside semivol, cross-sectional ranks, mean correlation, abs returns,
    cyclical time encodings.
    """
    
    stack_len = stack_len if stack_len is not None else getattr(Config, 'STACK_LEN', 10)
    norm_window = norm_window if norm_window is not None else getattr(Config, 'NORM_WINDOW', 63)

    log_ret = np.log(prices / prices.shift(1))
    mu_full = log_ret.rolling(norm_window, min_periods=norm_window).mean()
    sigma_full = log_ret.rolling(norm_window, min_periods=norm_window).std(ddof=0)
    if fit_end_date is not None:
        cutoff = pd.to_datetime(fit_end_date)
        mu = mu_full.loc[:cutoff].reindex(mu_full.index).ffill()
        sigma = sigma_full.loc[:cutoff].reindex(sigma_full.index).ffill()
    else:
        mu, sigma = mu_full, sigma_full
    r_norm = (log_ret - mu) / (sigma + 1e-8)

    frames: List[pd.DataFrame] = []
    col_names: List[str] = []
    for lag in range(stack_len):
        shifted = r_norm.shift(lag)
        frames.append(shifted)
        for c in prices.columns:
            col_names.append(f"ret_z_lag{lag}_{c}")

    if getattr(Config, 'FEAT_ADD_RET_HORIZONS', True):
        horizons = getattr(Config, 'FEAT_RET_HORIZONS', (1, 5, 21, 63))
        for h in horizons:
            rh = prices / prices.shift(h) - 1.0
            frames.append(rh)
            for c in prices.columns:
                col_names.append(f"ret_{h}d_{c}")
        if 20 not in horizons:
            mom20 = prices / prices.shift(20) - 1.0
            frames.append(mom20)
            for c in prices.columns:
                col_names.append(f"mom20_{c}")
        if 60 not in horizons:
            mom60 = prices / prices.shift(60) - 1.0
            frames.append(mom60)
            for c in prices.columns:
                col_names.append(f"mom60_{c}")
    else:
        mom20 = prices / prices.shift(20) - 1.0
        mom60 = prices / prices.shift(60) - 1.0
        frames += [mom20, mom60]
        for c in prices.columns:
            col_names.append(f"mom20_{c}")
        for c in prices.columns:
            col_names.append(f"mom60_{c}")

    diff = prices.diff(); gain = diff.clip(lower=0.0); loss = (-diff).clip(lower=0.0)
    span = 14; alpha = 1.0 / span
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=span).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=span).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi14 = 100.0 - (100.0 / (1.0 + rs))
    frames.append(rsi14)
    for c in prices.columns:
        col_names.append(f"rsi14_{c}")

    if raw_df is not None:
        try:
            vol_df = raw_df.loc[:, pd.IndexSlice[:, 'volume']]
            vol_df.columns = [c[0] for c in vol_df.columns]
        except Exception:
            vol_df = None
    else:
        vol_df = None
    if vol_df is not None and not vol_df.empty:
        vol_log = np.log1p(vol_df)
        vol_mu = vol_log.rolling(norm_window, min_periods=norm_window).mean()
        vol_sigma = vol_log.rolling(norm_window, min_periods=norm_window).std(ddof=0)
        vol_z = (vol_log - vol_mu) / (vol_sigma + 1e-8)
        frames.append(vol_z)
        for c in prices.columns:
            col_names.append(f"vol_z_{c}")

    if getattr(Config, 'FEAT_ADD_REALIZED_VOL', True):
        vol_windows = getattr(Config, 'FEAT_VOL_WINDOWS', (5, 21, 63))
        pct = prices.pct_change()
        for w in vol_windows:
            rv = pct.rolling(w).std(ddof=0) * np.sqrt(252)
            frames.append(rv)
            for c in prices.columns:
                col_names.append(f"rv_{w}_{c}")

    if getattr(Config, 'FEAT_ADD_DOWNSIDE_SEMIVOL', True):
        pct = prices.pct_change(); neg = pct.clip(upper=0.0)
        for w in getattr(Config, 'FEAT_SEMIVOL_WINDOWS', (21, 63)):
            semi = neg.rolling(w).std(ddof=0) * np.sqrt(252)
            frames.append(semi)
            for c in prices.columns:
                col_names.append(f"semivol_{w}_{c}")

    if getattr(Config, 'FEAT_ADD_XS_RANKS', True):
        ret21 = prices / prices.shift(21) - 1.0
        vol21 = prices.pct_change().rolling(21).std(ddof=0)
        def _rank_df(df: pd.DataFrame, prefix: str):
            ranked = df.rank(axis=1, pct=True)
            frames.append(ranked)
            for c in prices.columns:
                col_names.append(f"{prefix}_{c}")
        _rank_df(ret21, 'rank_ret21')
        _rank_df(vol21, 'rank_vol21')

    if getattr(Config, 'FEAT_ADD_MEAN_CORR', True):
        cw = int(getattr(Config, 'FEAT_CORR_WINDOW', 21))
        pct = prices.pct_change(); mean_corr_list = []
        idx = prices.index; cols = prices.columns
        for i in range(len(idx)):
            if i < cw:
                mean_corr_list.append([np.nan]*len(cols)); continue
            window = pct.iloc[i-cw+1:i+1]
            corr = window.corr().values; mc = []
            for j in range(len(cols)):
                row = np.delete(corr[j], j); mc.append(float(np.nanmean(row)))
            mean_corr_list.append(mc)
        mean_corr_df = pd.DataFrame(mean_corr_list, index=idx, columns=[f"meancorr_{c}" for c in cols])
        frames.append(mean_corr_df)
        for c in cols: col_names.append(f"meancorr_{c}")

    if getattr(Config, 'FEAT_ADD_ABS_RET', True):
        abs_r = prices.pct_change().abs(); frames.append(abs_r)
        for c in prices.columns: col_names.append(f"absret_{c}")

    if getattr(Config, 'FEAT_ADD_TIME_CYCLICAL', True):
        idx = prices.index
        day = idx.day.values; month = idx.month.values
        cyc = pd.DataFrame({
            'day_sin': np.sin(2*np.pi*day/31.0),
            'day_cos': np.cos(2*np.pi*day/31.0),
            'month_sin': np.sin(2*np.pi*month/12.0),
            'month_cos': np.cos(2*np.pi*month/12.0)
        }, index=idx)
        frames.append(cyc); col_names += list(cyc.columns)

    feat = pd.concat(frames, axis=1); feat.columns = col_names
    feat = feat.dropna().copy()
    assert feat.shape[1] == len(col_names), f"Feature column mismatch: names={len(col_names)} vs data={feat.shape[1]}"
    prices_aligned = prices.loc[feat.index].copy()
    return feat, prices_aligned


def train_val_test_split(prices: pd.DataFrame):
    idx = prices.index
    ts = pd.to_datetime(Config.TRAIN_START)
    te = pd.to_datetime(getattr(Config, 'TRAIN_END', Config.VAL_START))
    vs = pd.to_datetime(Config.VAL_START)
    ve = pd.to_datetime(getattr(Config, 'VAL_END', Config.TEST_START))
    tes = pd.to_datetime(Config.TEST_START)
    tee = pd.to_datetime(Config.TEST_END)
    train_mask = (idx >= ts) & (idx <= te)
    val_mask = (idx >= vs) & (idx <= ve)
    test_mask = (idx >= tes) & (idx <= tee)
    return prices.loc[train_mask], prices.loc[val_mask], prices.loc[test_mask]


__all__ = ["load_and_process_data", "create_features", "train_val_test_split"]