from __future__ import annotations

import bisect
from datetime import date

import pandas as pd


def next_trading_date(day: date, trading_dates: list[pd.Timestamp]) -> pd.Timestamp | pd.NaT:
    key = pd.Timestamp(day)
    idx = bisect.bisect_left(trading_dates, key)
    if idx >= len(trading_dates):
        return pd.NaT
    return trading_dates[idx]


def assign_effective_trade_date(
    timestamps: pd.Series,
    trading_dates: list[pd.Timestamp],
    cutoff_hour: int = 13,
    cutoff_minute: int = 30,
) -> pd.Series:
    ts_tw = pd.to_datetime(timestamps, utc=True, format="mixed").dt.tz_convert("Asia/Taipei")
    cutoff_minutes = cutoff_hour * 60 + cutoff_minute
    minutes = ts_tw.dt.hour * 60 + ts_tw.dt.minute
    base_dates = ts_tw.dt.date
    late_mask = minutes > cutoff_minutes
    shifted = pd.Series(base_dates, index=timestamps.index, dtype="object")
    shifted.loc[late_mask] = (ts_tw.loc[late_mask] + pd.Timedelta(days=1)).dt.date
    return shifted.map(lambda d: next_trading_date(d, trading_dates))

