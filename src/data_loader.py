import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from bisect import bisect_right
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class WalkForwardSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

def expanding_window_walk_forward(index, initial_train_size, val_size, test_size, step_size):
    splits, train_end = [], initial_train_size
    while True:
        if train_end + val_size + test_size > len(index): break
        splits.append(WalkForwardSplit(
            np.arange(0, train_end), 
            np.arange(train_end, train_end + val_size), 
            np.arange(train_end + val_size, train_end + val_size + test_size)
        ))
        train_end += step_size
    return splits

def prepare_text_dataframe(text_path):
    df = pd.read_csv(text_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, format="mixed")
    for c in ['kw_china', 'kw_tariffs', 'kw_chips', 'kw_ai', 'vader_compound', 'weighted_vader', 'emotion_score']:
        if c not in df.columns: df[c] = 0.0
    if 'emotion_label' not in df.columns: df['emotion_label'] = 'neutral'
    df['high_risk_post'] = (((df['kw_china'] == 1) | (df['kw_tariffs'] == 1)) & (df['vader_compound'] < -0.3)).astype(float)
    df['tech_positive_post'] = (((df['kw_chips'] == 1) | (df['kw_ai'] == 1)) & (df['vader_compound'] > 0.3)).astype(float)
    df['impact_score'] = df['weighted_vader'] * df['emotion_score']
    df['anger_impact'] = np.where(df['emotion_label'] == 'anger', df['weighted_vader'] * df['emotion_score'], 0.0)
    return df

def load_market_features(prices_path, volumes_path, inst_path, margin_path, tx_path, target_ticker):
    def read_market_csv(path):
        df = pd.read_csv(path)
        df = df.set_index(df.columns[0] if "Date" not in df.columns and "Unnamed: 0" not in df.columns else "Date" if "Date" in df.columns else "Unnamed: 0")
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        return df

    prices, volumes = read_market_csv(prices_path), read_market_csv(volumes_path)
    if target_ticker not in prices.columns: raise ValueError(f"{target_ticker} not found in prices.")

    close_returns = prices[[target_ticker]].pct_change().replace([np.inf, -np.inf], 0.0)
    volume_returns = volumes[[target_ticker]].pct_change().replace([np.inf, -np.inf], 0.0)

    us_features = [c for c in ["TSM", "^SOX", "^NDX", "^GSPC", "^VIX", "TWD=X", "^TNX"] if c in prices.columns]
    us_returns = prices[[c for c in us_features if c not in ["^VIX", "^TNX"]]].pct_change().replace([np.inf, -np.inf], 0.0)
    
    us_diff = pd.DataFrame(index=prices.index)
    if "^VIX" in prices.columns:
        us_diff["^VIX"] = prices["^VIX"].diff()
        us_diff['vix_z20'] = (prices["^VIX"] - prices["^VIX"].rolling(20).mean()) / prices["^VIX"].rolling(20).std().replace(0, 1)
        us_diff['vix_spike'] = (us_diff['vix_z20'] > 2.0).astype(float)
    if "^TNX" in prices.columns: us_diff["^TNX"] = prices["^TNX"].diff()
        
    if 'TSM' in prices.columns and '^SOX' in prices.columns:
        us_diff['tsm_vs_sox_5d'] = (prices['TSM'].pct_change() - prices['^SOX'].pct_change()).rolling(5).mean()

    us_shifted = pd.concat([us_returns, us_diff], axis=1).shift(1)

    inst_pivot = pd.DataFrame(index=prices.index)
    if target_ticker.endswith(".TW"):
        inst_df = pd.read_csv(inst_path, parse_dates=["date"])
        inst_df = inst_df[inst_df["stock_id"].astype(str) == target_ticker.replace(".TW", "")]
        if not inst_df.empty:
            inst_df["net_buy"] = inst_df["buy"] - inst_df["sell"]
            inst_pivot = inst_df.pivot_table(index="date", columns="name", values="net_buy", aggfunc="sum").sort_index()
            inst_pivot.columns = [f"inst_{c}_net_buy" for c in inst_pivot.columns]
            if 'inst_Foreign_Investor_net_buy' in inst_pivot.columns:
                direction = np.sign(inst_pivot['inst_Foreign_Investor_net_buy'])
                inst_pivot['foreign_streak'] = (direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1) * direction

    margin_df = pd.DataFrame(index=prices.index)
    if target_ticker.endswith(".TW"):
        margin_raw = pd.read_csv(margin_path, parse_dates=["date"])
        margin_raw = margin_raw[margin_raw["stock_id"].astype(str) == target_ticker.replace(".TW", "")].set_index("date").sort_index()
        if not margin_raw.empty and "MarginPurchaseTodayBalance" in margin_raw.columns:
            margin_df = margin_raw[["MarginPurchaseTodayBalance", "ShortSaleTodayBalance"]].pct_change().replace([np.inf, -np.inf], 0.0)
            margin_df = margin_df.rename(columns={"MarginPurchaseTodayBalance": "margin_purchase_change", "ShortSaleTodayBalance": "short_sale_change"})

    tx_df = pd.DataFrame(index=prices.index)
    if target_ticker.endswith(".TW"):
        tx_df_raw = pd.read_csv(tx_path, parse_dates=["date"])
        if "trading_session" in tx_df_raw.columns: tx_df_raw = tx_df_raw[tx_df_raw["trading_session"] == "after_market"]
        tx_df_raw = tx_df_raw.set_index("date").sort_index()
        if "spread_per" in tx_df_raw.columns:
            tx_df["tx_spread_per"] = tx_df_raw["spread_per"]
            tx_df['tx_spread_ema3'] = tx_df['tx_spread_per'].ewm(span=3).mean()
            tx_df['tx_spread_z20'] = (tx_df['tx_spread_per'] - tx_df['tx_spread_per'].rolling(20).mean()) / tx_df['tx_spread_per'].rolling(20).std().replace(0, 1)
        if "volume" in tx_df_raw.columns:
            tx_df["tx_volume_change"] = tx_df_raw["volume"].pct_change().replace([np.inf, -np.inf], 0.0)

    features = [
        prices.rename(columns={target_ticker: f"close_{target_ticker}"}),
        close_returns.rename(columns={target_ticker: f"close_ret_{target_ticker}"}),
        volume_returns.rename(columns={target_ticker: f"volume_{target_ticker}"}),
        us_shifted, inst_pivot, margin_df, tx_df,
    ]

    market_df = pd.concat(features, axis=1, sort=False).sort_index()
    market_df = market_df.ffill()
    market_df = market_df.dropna(subset=[f"close_{target_ticker}"])
    market_df = market_df.fillna(0.0)
    
    return market_df

class CustomDataset(Dataset):
    def __init__(
        self, market_df: pd.DataFrame, text_df: pd.DataFrame, window_size: int,
        market_feature_cols: Optional[List[str]] = None, text_feature_cols: Optional[List[str]] = None,
        text_timestamp_col: str = "Timestamp", text_timezone: str = "UTC",
        market_timezone: str = "Asia/Taipei", market_open_time: str = "09:00",
        market_close_time: str = "13:30", aggregation: str = "weighted_mean",
        weight_col: Optional[str] = "Likes", label_series: Optional[pd.Series] = None,
        close_price_col: Optional[str] = None, open_price_col: Optional[str] = None,
        volatility_window: int = 20, z_score: float = 1.0,
    ) -> None:
        if window_size < 2: raise ValueError("window_size must be >= 2")
        self.window_size = window_size

        market_df = market_df.copy()
        if not isinstance(market_df.index, pd.DatetimeIndex): market_df.index = pd.to_datetime(market_df.index)
        market_df = market_df.sort_index()

        self.trading_index = market_df.index.normalize()
        self.trading_dates = [d.date() for d in self.trading_index]
        self.trading_date_set = set(self.trading_dates)

        if market_feature_cols is None:
            market_feature_cols = [c for c in market_df.columns if pd.api.types.is_numeric_dtype(market_df[c])]
        self.market_feature_cols = market_feature_cols
        self.market_features = market_df[self.market_feature_cols].astype(np.float32).to_numpy()

        if label_series is None:
            if close_price_col is None: raise ValueError("close_price_col must be provided.")
            open_series = market_df[open_price_col] if open_price_col in market_df.columns else market_df[close_price_col]
            label_series = self._build_dynamic_gap_labels(market_df[close_price_col], open_series, volatility_window, z_score)
        label_series = label_series.reindex(self.trading_index)
        self.labels = label_series.to_numpy()

        text_daily = self._aggregate_text_features(
            text_df, text_feature_cols, text_timestamp_col, text_timezone, 
            market_timezone, market_open_time, market_close_time, aggregation, weight_col
        )
        self.text_features = text_daily.to_numpy(dtype=np.float32)

        valid_mask = ~pd.isna(self.labels)
        valid_indices = np.where(valid_mask)[0]
        self.valid_indices = valid_indices[valid_indices >= (window_size - 1)]
        self.sample_index = self.trading_index[self.valid_indices]

    @staticmethod
    def _build_dynamic_gap_labels(close_series: pd.Series, open_series: pd.Series, volatility_window: int, z_score: float) -> pd.Series:
        gap_returns = (open_series.shift(-1) - close_series) / close_series
        rolling_std = gap_returns.rolling(window=volatility_window).std().bfill().replace(0.0, np.nan).bfill()
        threshold = z_score * rolling_std
        labels = pd.Series(index=close_series.index, dtype="float32")
        labels[gap_returns < -threshold] = 0
        labels[(gap_returns >= -threshold) & (gap_returns <= threshold)] = 1
        labels[gap_returns > threshold] = 2
        return labels

    def _aggregate_text_features(self, text_df, text_feature_cols, text_timestamp_col, text_timezone, market_timezone, market_open_time, market_close_time, aggregation, weight_col) -> pd.DataFrame:
        if text_df is None or text_df.empty: return pd.DataFrame(np.zeros((len(self.trading_index), 1), dtype=np.float32), index=self.trading_index, columns=["text_placeholder"])

        df = text_df.copy()
        if "emotion_label" in df.columns and "emotion_label" not in (text_feature_cols or []):
            emotion_ohe = pd.get_dummies(df["emotion_label"], prefix="emo")
            df = pd.concat([df.drop(columns=["emotion_label"]), emotion_ohe], axis=1)

        if text_feature_cols is None:
            text_feature_cols = [c for c in df.columns if c != text_timestamp_col and pd.api.types.is_numeric_dtype(df[c])]
        
        df[text_timestamp_col] = pd.to_datetime(df[text_timestamp_col], errors="coerce")
        df = df.dropna(subset=[text_timestamp_col])
        ts = df[text_timestamp_col]
        if ts.dt.tz is None: ts = ts.dt.tz_localize(text_timezone)
        ts = ts.dt.tz_convert(market_timezone)

        market_close = pd.to_datetime(market_close_time).time()

        def next_trading_date(d):
            pos = bisect_right(self.trading_dates, d)
            return self.trading_dates[pos] if pos < len(self.trading_dates) else None

        target_dates = []
        for t in ts:
            d, tm = t.date(), t.time()
            if d in self.trading_date_set:
                target_dates.append(next_trading_date(d) if tm >= market_close else d)
            else:
                target_dates.append(next_trading_date(d))

        df["trade_date"] = target_dates
        df = df.dropna(subset=["trade_date"])

        def weighted_mean(group):
            feats = group[text_feature_cols].to_numpy(dtype=np.float32)
            if weight_col is None or weight_col not in group.columns: return pd.Series(feats.mean(axis=0), index=text_feature_cols)
            weights = np.maximum(group[weight_col].to_numpy(dtype=np.float32), 0.0)
            if weights.sum() == 0: return pd.Series(feats.mean(axis=0), index=text_feature_cols)
            return pd.Series((feats.T * weights).T.sum(axis=0) / weights.sum(), index=text_feature_cols)

        if aggregation == "max": agg_df = df.groupby("trade_date")[text_feature_cols].max()
        elif aggregation == "mean": agg_df = df.groupby("trade_date")[text_feature_cols].mean()
        else: agg_df = df.groupby("trade_date").apply(weighted_mean)

        agg_df.index = pd.to_datetime(agg_df.index)
        agg_df = agg_df.sort_index().reindex(self.trading_index, fill_value=0.0)

        sentiment_cols = [c for c in agg_df.columns if c in ['vader_compound', 'weighted_vader', 'emotion_score', 'impact_score', 'anger_impact'] or c.startswith('emo_')]
        kw_cols = [c for c in agg_df.columns if c.startswith('kw_') or c in ['high_risk_post', 'tech_positive_post']]

        if sentiment_cols: agg_df[sentiment_cols] = agg_df[sentiment_cols].replace(0, np.nan).ewm(halflife=3, min_periods=1).mean().fillna(0)
        if kw_cols: agg_df[kw_cols] = agg_df[kw_cols].rolling(5, min_periods=1).sum().fillna(0)
            
        return agg_df

    def __len__(self) -> int: return len(self.valid_indices)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.valid_indices[idx]
        return (torch.from_numpy(self.market_features[(t - self.window_size + 1) : (t + 1)]), torch.from_numpy(self.text_features[t]), torch.tensor(int(self.labels[t]), dtype=torch.long))