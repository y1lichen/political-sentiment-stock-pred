import argparse
import json
import random
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


PRICE_PATH = "data/taiwan_market_data/global_prices.csv"
RAW_EVENT_INPUT_PATH = "data/text/trump_posts_features_2017_2026.csv"
# 事件特徵改成 per-market;由 event_path_for_target() 依 --target 挑 _us / _tw 那份。
INSTITUTION_PATH = "data/taiwan_market_data/institutional_investors.csv"
MARGIN_PATH = "data/taiwan_market_data/margin_trading.csv"
FUTURES_NIGHT_PATH = "data/taiwan_market_data/tx_futures_night.csv"
US_INSTITUTION_PATH = "data/taiwan_market_data/us_institutional_investors.csv"
US_MARGIN_PATH = "data/taiwan_market_data/us_margin_trading.csv"
US_FUTURES_NIGHT_PATH = "data/taiwan_market_data/us_futures_night.csv"

COUNT_EVENT_COLS = {
    "post_count",
    "tariff_count",
    "deal_count",
    "relief_count",
    "china_count",
    "taiwan_count",
    "chips_count",
    "ai_count",
    "night_post_count",
    "pre_post_count",
    "open_post_count",
    "total_excl",
    "avg_post_len",
    "all_caps_post_count",
    "exclaim_post_count",
    "uppercase_phrase_count",
}

REGIME_LABELS = {
    0: "calm_bull",
    1: "risk_off",
    2: "oversold",
    3: "neutral",
}

DIRECTION_LABELS = {
    0: "down",
    1: "up",
}

PRESIDENTIAL_TERMS = [
    ("2017-01-20", "2021-01-19"),
    ("2025-01-20", "2100-12-31"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trump event brute-force + deep learning market-regime fusion model "
            "for Taiwan stock next-day return classification."
        )
    )
    parser.add_argument("--target", default="2330.TW", help="Target ticker in global_prices.csv.")
    parser.add_argument("--hold", type=int, default=1, help="Prediction horizon in trading days.")
    parser.add_argument(
        "--presidential-terms-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only Trump presidential-term samples and remove out-of-office dates.",
    )
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=0.0,
        help="Future return threshold for binary label. > threshold is up, otherwise down.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Lookback window for --model-type lstm. Ignored by gated_mlp except for split overlap.",
    )
    parser.add_argument("--min-n", type=int, default=20, help="Minimum samples for a brute-force event combo.")
    parser.add_argument("--min-abs-mean-ret", type=float, default=0.001, help="Minimum absolute mean return.")
    parser.add_argument("--min-hit-rate", type=float, default=0.55, help="Minimum directional hit rate.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.03,
        help=(
            "Minimum brute-force event score before top-k truncation. "
            "Score = abs(mean_ret) * sqrt(n) * directional_hit."
        ),
    )
    parser.add_argument("--top-k-events", type=int, default=80, help="Max brute-force event combos used by DL.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--model-type",
        choices=["gated_mlp", "lstm"],
        default="gated_mlp",
        help="Deep learning architecture. gated_mlp is recommended for small tabular datasets.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["full", "market_only"],
        default="full",
        help="Use full event+market features or a market-only baseline with Trump event features removed.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-class-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inverse-frequency class weights for the binary direction loss.",
    )
    parser.add_argument(
        "--trade-edge-threshold",
        type=float,
        default=0.10,
        help=(
            "Neutral aggregation threshold. Trade LONG if prob_up - prob_down is above this, "
            "SHORT if below its negative value, otherwise stay NEUTRAL."
        ),
    )
    parser.add_argument(
        "--auto-trade-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tune trade-edge threshold on the validation set, then apply the chosen threshold to test.",
    )
    parser.add_argument(
        "--min-val-trade-coverage",
        type=float,
        default=0.05,
        help="Minimum validation trade coverage when auto-tuning the neutral threshold.",
    )
    parser.add_argument(
        "--trade-mode",
        choices=["long_short", "long_cash", "short_cash"],
        default="long_short",
        help="Trading interpretation of binary predictions after confidence filtering.",
    )
    parser.add_argument("--output-dir", default="data/output/deep_regime_fusion")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_price_data(path, target):
    price_df = pd.read_csv(path)
    if "Date" in price_df.columns:
        date_col = "Date"
    elif "date" in price_df.columns:
        date_col = "date"
    elif "Unnamed: 0" in price_df.columns:
        date_col = "Unnamed: 0"
    else:
        date_col = price_df.columns[0]
    price_df = price_df.rename(columns={date_col: "Date"})
    price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce")
    price_df = price_df.dropna(subset=["Date"])
    price_df = price_df.sort_values("Date").set_index("Date")
    if target not in price_df.columns:
        raise ValueError(f"Target {target!r} not found in {path}.")
    return price_df


def ensure_event_feature_files():
    """Build relabeled per-market Trump event files from the raw post CSV when missing."""
    expected = [
        Path("data/output/trump_posts_with_event_features_us.csv"),
        Path("data/output/trump_posts_with_event_features_tw.csv"),
    ]
    if all(path.exists() for path in expected):
        return

    raw_path = Path(RAW_EVENT_INPUT_PATH)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw Trump post file not found: {RAW_EVENT_INPUT_PATH}. "
            "Cannot rebuild relabeled event features."
        )

    from data.data_preprocess import main as preprocess_main

    print("Relabeled Trump event feature files are missing; rebuilding from raw CSV.")
    preprocess_main()


def event_path_for_target(target):
    """依標的所在市場挑 post-level 事件特徵 CSV, 若分市場檔不存在則回退到共用事件檔。"""
    suffix = "tw" if target.endswith(".TW") else "us"
    market_path = Path(f"data/output/trump_posts_with_event_features_{suffix}.csv")
    if market_path.exists():
        return str(market_path)

    ensure_event_feature_files()
    if market_path.exists():
        return str(market_path)

    fallback_path = Path("data/output/trump_posts_with_event_features.csv")
    if fallback_path.exists():
        print(
            f"Warning: {market_path} not found; using fallback event features: {fallback_path}"
        )
        return str(fallback_path)

    raise FileNotFoundError(
        "No Trump post-level event feature file found. Expected either "
        f"{market_path} or {fallback_path}. Run `python data/data_preprocess.py` first."
    )


def read_event_data(path):
    post_df = pd.read_csv(path)
    post_df["Timestamp"] = pd.to_datetime(post_df["Timestamp"], utc=True, errors="coerce")
    post_df = post_df.dropna(subset=["Timestamp"])
    return post_df.sort_values("Timestamp").reset_index(drop=True)


def real_price_series(price_df, target):
    """去掉 forward-fill 的假日列 (與前一交易日收盤完全相同, diff == 0)。

    統一日曆對非交易日做了 ffill,會造成假的 0 報酬,且其前一日的 future_ret 會指向
    stale 價而把 direction_label 標錯。首列 diff 為 NaN 視為真實交易日保留。
    """
    price = price_df[target].dropna()
    return price[price.diff().fillna(1) != 0]


def market_timezone(ticker):
    return "Asia/Taipei" if ticker.endswith(".TW") else "America/New_York"


def market_close_available_at(dates, ticker):
    dates = pd.to_datetime(dates).normalize()
    tz = market_timezone(ticker)
    if ticker.endswith(".TW"):
        offset = pd.Timedelta(hours=14)
    else:
        offset = pd.Timedelta(hours=16, minutes=30)
    local_time = pd.DatetimeIndex(dates).tz_localize(tz) + offset
    return pd.Series(local_time.tz_convert("UTC"), index=dates)


def next_morning_available_at(dates, ticker):
    dates = pd.to_datetime(dates).normalize()
    tz = market_timezone(ticker)
    local_time = pd.DatetimeIndex(dates + pd.Timedelta(days=1)).tz_localize(tz) + pd.Timedelta(hours=9)
    return pd.Series(local_time.tz_convert("UTC"), index=dates)


def same_morning_available_at(dates, ticker):
    dates = pd.to_datetime(dates).normalize()
    tz = market_timezone(ticker)
    local_time = pd.DatetimeIndex(dates).tz_localize(tz) + pd.Timedelta(hours=9)
    return pd.Series(local_time.tz_convert("UTC"), index=dates)


def decision_times_for_samples(sample_dates, target):
    # close-to-next-close: decide after the target market close on sample date D.
    return market_close_available_at(sample_dates, target)


def asof_join_features(sample_dates, decision_times, source_df, feature_cols):
    if source_df is None or source_df.empty or not feature_cols:
        return pd.DataFrame(index=sample_dates)

    right = source_df[["available_at"] + feature_cols].copy()
    right = right.dropna(subset=["available_at"]).sort_values("available_at")
    if right.empty:
        return pd.DataFrame(index=sample_dates)

    left = pd.DataFrame(
        {
            "sample_date": pd.to_datetime(sample_dates),
            "decision_time": pd.Series(decision_times, index=sample_dates).to_numpy(),
        }
    ).sort_values("decision_time")

    merged = pd.merge_asof(
        left,
        right,
        left_on="decision_time",
        right_on="available_at",
        direction="backward",
    )
    merged = merged.set_index("sample_date")
    return merged[feature_cols].reindex(pd.to_datetime(sample_dates))


def price_feature_source(price_df, ticker, prefix, feature_kind):
    price = real_price_series(price_df, ticker)
    out = pd.DataFrame(index=price.index)

    if feature_kind == "target":
        out[f"{prefix}_ret_1d"] = price.pct_change()
        out[f"{prefix}_ret_3d"] = price.pct_change(3)
        out[f"{prefix}_ret_5d"] = price.pct_change(5)
        out[f"{prefix}_ret_10d"] = price.pct_change(10)
        out[f"{prefix}_vol_10d"] = out[f"{prefix}_ret_1d"].rolling(10).std()
        out[f"{prefix}_vol_20d"] = out[f"{prefix}_ret_1d"].rolling(20).std()
        out[f"{prefix}_ma_gap_5"] = price / price.rolling(5).mean() - 1
        out[f"{prefix}_ma_gap_20"] = price / price.rolling(20).mean() - 1
        out[f"{prefix}_ma_gap_60"] = price / price.rolling(60).mean() - 1
        rolling_high = price.rolling(60, min_periods=20).max()
        out[f"{prefix}_drawdown_60"] = price / rolling_high - 1
    elif feature_kind == "lead":
        out[f"{ticker}_ret_1d"] = price.pct_change()
        out[f"{ticker}_ret_5d"] = price.pct_change(5)
    elif feature_kind == "vix":
        out["vix_level"] = price
        out["vix_diff_1d"] = price.diff()
        out["vix_pct_chg"] = price.pct_change()
        out["vix_z20"] = (price - price.rolling(20).mean()) / price.rolling(20).std()
        out["vix_low"] = (price < price.rolling(252, min_periods=60).quantile(0.35)).astype(float)
        out["vix_high"] = (price > price.rolling(252, min_periods=60).quantile(0.75)).astype(float)
    elif feature_kind == "tnx":
        out["tnx_diff_1d"] = price.diff()

    out["available_at"] = market_close_available_at(out.index, ticker).to_numpy()
    return out


def institution_feature_source(path, target):
    stock_id = target.split(".")[0]
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    inst = pd.read_csv(p)
    inst = inst[inst["stock_id"].astype(str) == stock_id].copy()
    if inst.empty:
        return pd.DataFrame()

    inst["date"] = pd.to_datetime(inst["date"])
    inst["net_buy"] = inst["buy"] - inst["sell"]
    wide = inst.pivot_table(index="date", columns="name", values="net_buy", aggfunc="sum")
    wide = wide.add_prefix("inst_net_")
    wide["inst_net_total"] = wide.sum(axis=1)
    wide["available_at"] = next_morning_available_at(wide.index, target).to_numpy()
    return wide


def margin_feature_source(path, target):
    stock_id = target.split(".")[0]
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    margin = pd.read_csv(p)
    margin = margin[margin["stock_id"].astype(str) == stock_id].copy()
    if margin.empty:
        return pd.DataFrame()

    margin["date"] = pd.to_datetime(margin["date"])
    margin = margin.set_index("date").sort_index()
    cols = [
        "MarginPurchaseTodayBalance",
        "ShortSaleTodayBalance",
        "MarginPurchaseBuy",
        "MarginPurchaseSell",
        "ShortSaleBuy",
        "ShortSaleSell",
    ]
    cols = [c for c in cols if c in margin.columns]
    if not cols:
        return pd.DataFrame()
    out = margin[cols].add_prefix("margin_")
    if "margin_MarginPurchaseTodayBalance" in out.columns:
        out["margin_balance_chg"] = out["margin_MarginPurchaseTodayBalance"].pct_change()
    if "margin_ShortSaleTodayBalance" in out.columns:
        out["short_balance_chg"] = out["margin_ShortSaleTodayBalance"].pct_change()
    out["available_at"] = next_morning_available_at(out.index, target).to_numpy()
    return out


def futures_night_feature_source(path, target=None):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    futures = pd.read_csv(p)
    if target is not None and "stock_id" in futures.columns:
        futures = futures[futures["stock_id"].astype(str) == target].copy()
    if futures.empty:
        return pd.DataFrame()

    futures["date"] = pd.to_datetime(futures["date"])
    futures = futures.sort_values("date").set_index("date")
    cols = [c for c in ["spread", "spread_per", "volume"] if c in futures.columns]
    if not cols:
        return pd.DataFrame()

    out = futures[cols].add_prefix("tx_night_")
    if "tx_night_volume" in out.columns:
        out["tx_night_volume_z20"] = (
            out["tx_night_volume"] - out["tx_night_volume"].rolling(20).mean()
        ) / out["tx_night_volume"].rolling(20).std()
    out["available_at"] = same_morning_available_at(out.index, target or "2330.TW").to_numpy()
    return out


def presidential_term_segment(index):
    dates = pd.to_datetime(index)
    segment = pd.Series(0, index=index, dtype="int64")
    for i, (start, end) in enumerate(PRESIDENTIAL_TERMS, start=1):
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        segment.loc[mask] = i
    return segment


def filter_presidential_terms(df):
    segment = presidential_term_segment(df.index)
    out = df.loc[segment > 0].copy()
    out["term_segment"] = segment.loc[out.index].astype(int)
    return out


def binary_event_columns(event_df):
    cols = []
    for col in event_df.columns:
        if col in COUNT_EVENT_COLS:
            continue
        values = event_df[col].dropna().unique()
        if len(values) > 0 and set(values).issubset({0, 1, 0.0, 1.0}):
            cols.append(col)
    return cols


def aggregate_posts_for_decisions(post_df, sample_dates, decision_times):
    """Aggregate posts into the next holding-period window for each sample.

    For close-to-next-close experiments, sample D predicts D close -> next close.
    Trump posts after D close and before the next decision time are assigned to
    sample D, matching the project assumption that overnight Trump posts can be
    used to explain/predict the next trading day.
    """
    sample_dates = pd.to_datetime(sample_dates)
    decision_times = pd.Series(decision_times, index=sample_dates).sort_index()
    out_index = decision_times.index

    daily = pd.DataFrame(index=out_index)
    base_count_cols = [
        "post_count",
        "tariff_count",
        "deal_count",
        "relief_count",
        "action_count",
        "attack_count",
        "positive_count",
        "market_brag_count",
        "china_count",
        "taiwan_count",
        "chips_count",
        "ai_count",
        "iran_count",
        "russia_count",
        "night_post_count",
        "pre_post_count",
        "open_post_count",
        "total_excl",
        "total_caps",
        "total_alpha",
        "avg_post_len",
        "all_caps_post_count",
        "exclaim_post_count",
        "uppercase_phrase_count",
        "sig_djt_count",
        "sig_potus_count",
        "sig_tyfa_count",
        "pre_tariff_count",
        "pre_deal_count",
        "pre_relief_count",
        "pre_action_count",
        "open_tariff_count",
        "open_deal_count",
    ]
    for col in base_count_cols:
        daily[col] = 0.0

    if post_df.empty or len(decision_times) == 0:
        return binary_features_from_interval_counts(daily)

    post_times = post_df["Timestamp"].sort_values()
    decision_ns = decision_times.to_numpy(dtype="datetime64[ns]")
    post_ns = post_times.to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(decision_ns, post_ns, side="left") - 1
    valid = (positions >= 0) & (positions < len(decision_times))
    if not valid.any():
        return binary_features_from_interval_counts(daily)

    assigned = post_df.loc[post_times.index[valid]].copy()
    assigned["sample_date"] = decision_times.index.to_numpy()[positions[valid]]

    required = [
        "ev_tariff",
        "ev_deal",
        "ev_relief",
        "ev_action",
        "ev_attack",
        "ev_positive",
        "ev_market_brag",
        "ev_china",
        "ev_taiwan",
        "ev_chips",
        "ev_ai",
        "ev_iran",
        "ev_russia",
        "is_night_post",
        "is_pre_market_post",
        "is_market_open_post",
        "exclamation_count",
        "caps_count",
        "alpha_count",
        "is_all_caps_post",
        "has_exclamation_mark",
        "has_uppercase_phrase",
        "post_len",
        "sig_djt",
        "sig_potus",
        "sig_tyfa",
    ]
    for col in required:
        if col not in assigned.columns:
            assigned[col] = 0

    grouped = assigned.groupby("sample_date")
    daily["post_count"] = grouped.size().reindex(out_index, fill_value=0).astype(float)
    mappings = {
        "tariff_count": "ev_tariff",
        "deal_count": "ev_deal",
        "relief_count": "ev_relief",
        "action_count": "ev_action",
        "attack_count": "ev_attack",
        "positive_count": "ev_positive",
        "market_brag_count": "ev_market_brag",
        "china_count": "ev_china",
        "taiwan_count": "ev_taiwan",
        "chips_count": "ev_chips",
        "ai_count": "ev_ai",
        "iran_count": "ev_iran",
        "russia_count": "ev_russia",
        "night_post_count": "is_night_post",
        "pre_post_count": "is_pre_market_post",
        "open_post_count": "is_market_open_post",
        "total_excl": "exclamation_count",
        "total_caps": "caps_count",
        "total_alpha": "alpha_count",
        "all_caps_post_count": "is_all_caps_post",
        "exclaim_post_count": "has_exclamation_mark",
        "uppercase_phrase_count": "has_uppercase_phrase",
        "sig_djt_count": "sig_djt",
        "sig_potus_count": "sig_potus",
        "sig_tyfa_count": "sig_tyfa",
    }
    for out_col, src_col in mappings.items():
        daily[out_col] = grouped[src_col].sum().reindex(out_index, fill_value=0).astype(float)
    daily["avg_post_len"] = grouped["post_len"].mean().reindex(out_index, fill_value=0).astype(float)

    special = grouped.apply(lambda g: pd.Series({
        "pre_tariff_count": ((g["is_pre_market_post"] == 1) & (g["ev_tariff"] == 1)).sum(),
        "pre_deal_count": ((g["is_pre_market_post"] == 1) & (g["ev_deal"] == 1)).sum(),
        "pre_relief_count": ((g["is_pre_market_post"] == 1) & (g["ev_relief"] == 1)).sum(),
        "pre_action_count": ((g["is_pre_market_post"] == 1) & (g["ev_action"] == 1)).sum(),
        "open_tariff_count": ((g["is_market_open_post"] == 1) & (g["ev_tariff"] == 1)).sum(),
        "open_deal_count": ((g["is_market_open_post"] == 1) & (g["ev_deal"] == 1)).sum(),
    }))
    for col in special.columns:
        daily[col] = special[col].reindex(out_index, fill_value=0).astype(float)

    return binary_features_from_interval_counts(daily)


def binary_features_from_interval_counts(daily):
    out = pd.DataFrame(index=daily.index)

    out["post_count"] = daily["post_count"]
    out["tariff_count"] = daily["tariff_count"]
    out["deal_count"] = daily["deal_count"]
    out["relief_count"] = daily["relief_count"]
    out["china_count"] = daily["china_count"]
    out["taiwan_count"] = daily["taiwan_count"]
    out["chips_count"] = daily["chips_count"]
    out["ai_count"] = daily["ai_count"]
    out["night_post_count"] = daily["night_post_count"]
    out["pre_post_count"] = daily["pre_post_count"]
    out["open_post_count"] = daily["open_post_count"]
    out["total_excl"] = daily["total_excl"]
    out["avg_post_len"] = daily["avg_post_len"]
    out["all_caps_post_count"] = daily["all_caps_post_count"]
    out["exclaim_post_count"] = daily["exclaim_post_count"]
    out["uppercase_phrase_count"] = daily["uppercase_phrase_count"]

    out["posts_high"] = (daily["post_count"] >= 20).astype(int)
    out["posts_low"] = (daily["post_count"] <= 5).astype(int)
    out["posts_very_high"] = (daily["post_count"] >= 35).astype(int)
    out["silence_day"] = (daily["post_count"] == 0).astype(int)
    out["has_tariff"] = (daily["tariff_count"] >= 1).astype(int)
    out["tariff_heavy"] = (daily["tariff_count"] >= 3).astype(int)
    out["has_deal"] = (daily["deal_count"] >= 1).astype(int)
    out["deal_heavy"] = (daily["deal_count"] >= 2).astype(int)
    out["has_relief"] = (daily["relief_count"] >= 1).astype(int)
    out["has_action"] = (daily["action_count"] >= 1).astype(int)
    out["has_attack"] = (daily["attack_count"] >= 1).astype(int)
    out["attack_heavy"] = (daily["attack_count"] >= 3).astype(int)
    out["has_positive"] = (daily["positive_count"] >= 1).astype(int)
    out["positive_heavy"] = (daily["positive_count"] >= 3).astype(int)
    out["has_market_brag"] = (daily["market_brag_count"] >= 1).astype(int)
    out["brag_heavy"] = (daily["market_brag_count"] >= 2).astype(int)
    out["has_china"] = (daily["china_count"] >= 1).astype(int)
    out["has_iran"] = (daily["iran_count"] >= 1).astype(int)
    out["has_russia"] = (daily["russia_count"] >= 1).astype(int)
    out["pre_tariff"] = (daily["pre_tariff_count"] >= 1).astype(int)
    out["pre_deal"] = (daily["pre_deal_count"] >= 1).astype(int)
    out["pre_relief"] = (daily["pre_relief_count"] >= 1).astype(int)
    out["pre_action"] = (daily["pre_action_count"] >= 1).astype(int)
    out["open_tariff"] = (daily["open_tariff_count"] >= 1).astype(int)
    out["open_tariff_heavy"] = (daily["open_tariff_count"] >= 2).astype(int)
    out["open_deal"] = (daily["open_deal_count"] >= 1).astype(int)
    out["has_night_post"] = (daily["night_post_count"] >= 1).astype(int)
    out["sig_djt"] = (daily["sig_djt_count"] >= 1).astype(int)
    out["sig_potus"] = (daily["sig_potus_count"] >= 1).astype(int)
    out["sig_tyfa"] = (daily["sig_tyfa_count"] >= 1).astype(int)

    caps_ratio = daily["total_caps"] / daily["total_alpha"].replace(0, np.nan)
    out["high_emotion"] = (caps_ratio.fillna(0) > 0.2).astype(int)
    out["has_all_caps"] = (daily["all_caps_post_count"] >= 1).astype(int)
    out["all_caps_heavy"] = (daily["all_caps_post_count"] >= 3).astype(int)
    out["has_uppercase_phrase"] = (daily["uppercase_phrase_count"] >= 1).astype(int)
    out["uppercase_phrase_heavy"] = (daily["uppercase_phrase_count"] >= 3).astype(int)
    out["has_exclamation"] = (daily["exclaim_post_count"] >= 1).astype(int)
    out["exclamation_heavy"] = (daily["exclaim_post_count"] >= 3).astype(int)
    out["lots_of_excl"] = (daily["total_excl"] >= 5).astype(int)
    out["long_posts"] = (daily["avg_post_len"] > 400).astype(int)
    out["short_posts"] = ((daily["avg_post_len"] < 150) & (daily["post_count"] > 0)).astype(int)
    out["deal_over_tariff"] = (
        (daily["deal_count"] > daily["tariff_count"]) & (daily["deal_count"] >= 1)
    ).astype(int)
    out["tariff_only"] = ((daily["tariff_count"] >= 1) & (daily["deal_count"] == 0)).astype(int)
    out["deal_only"] = ((daily["deal_count"] >= 1) & (daily["tariff_count"] == 0)).astype(int)
    out["has_taiwan"] = (daily["taiwan_count"] >= 1).astype(int)
    out["has_chips"] = (daily["chips_count"] >= 1).astype(int)
    out["chips_heavy"] = (daily["chips_count"] >= 2).astype(int)
    out["has_ai"] = (daily["ai_count"] >= 1).astype(int)
    out["china_taiwan_combo"] = ((daily["china_count"] >= 1) & (daily["taiwan_count"] >= 1)).astype(int)
    out["china_chips_combo"] = ((daily["china_count"] >= 1) & (daily["chips_count"] >= 1)).astype(int)
    out["taiwan_chips_combo"] = ((daily["taiwan_count"] >= 1) & (daily["chips_count"] >= 1)).astype(int)

    tariff_active = (daily["tariff_count"] >= 1).astype(int)
    china_active = (daily["china_count"] >= 1).astype(int)
    chips_active = (daily["chips_count"] >= 1).astype(int)
    out["tariff_streak_3d"] = (tariff_active.shift(1).rolling(3, min_periods=3).sum() >= 3).fillna(False).astype(int)
    out["tariff_rising"] = (
        (tariff_active.shift(1).rolling(3, min_periods=3).sum() >= 2)
        & (daily["tariff_count"] >= 1)
    ).fillna(False).astype(int)
    out["china_streak_3d"] = (china_active.shift(1).rolling(3, min_periods=3).sum() >= 3).fillna(False).astype(int)
    out["chips_streak_3d"] = (chips_active.shift(1).rolling(3, min_periods=3).sum() >= 3).fillna(False).astype(int)
    prev_7_post_avg = daily["post_count"].shift(1).rolling(7, min_periods=3).mean()
    out["volume_spike"] = (daily["post_count"] > prev_7_post_avg * 2).fillna(False).astype(int)
    out["volume_drop"] = (daily["post_count"] < prev_7_post_avg * 0.4).fillna(False).astype(int)
    return out


def make_event_combos(event_df, max_combo_size=2):
    event_cols = binary_event_columns(event_df)
    events = event_df[event_cols].astype("int8")
    combo_data = {col: events[col] for col in event_cols}

    if max_combo_size >= 2:
        for a, b in combinations(event_cols, 2):
            combo_data[f"{a}&{b}"] = ((events[a] == 1) & (events[b] == 1)).astype("int8")

    combo_df = pd.DataFrame(combo_data, index=events.index)
    return combo_df, event_cols


def brute_force_events(combo_df, price, hold, min_n, min_abs_mean_ret, min_hit_rate, min_score, top_k):
    future_ret = price.shift(-hold) / price - 1
    data = combo_df.join(future_ret.rename("future_ret"), how="inner")

    rows = []
    selected = []
    for col in combo_df.columns:
        mask = data[col] == 1
        n = int(mask.sum())
        if n < min_n:
            continue

        returns = data.loc[mask, "future_ret"].dropna()
        if returns.empty:
            continue

        mean_ret = float(returns.mean())
        hit_up = float((returns > 0).mean())
        hit_down = float((returns < 0).mean())
        direction = 1 if mean_ret >= 0 else -1
        directional_hit = hit_up if direction > 0 else hit_down
        score = abs(mean_ret) * np.sqrt(n) * directional_hit

        row = {
            "event": col,
            "n": n,
            "mean_ret": mean_ret,
            "median_ret": float(returns.median()),
            "std": float(returns.std()),
            "hit_up": hit_up,
            "hit_down": hit_down,
            "direction": "long" if direction > 0 else "short",
            "directional_hit": directional_hit,
            "score": float(score),
        }
        rows.append(row)

        if (
            abs(mean_ret) >= min_abs_mean_ret
            and directional_hit >= min_hit_rate
            and score >= min_score
        ):
            selected.append(row)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("No event combo has enough samples. Lower --min-n.")

    result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)
    selected_df = pd.DataFrame(selected).sort_values("score", ascending=False)
    selected_before_top_k = int(len(selected_df))
    if selected_df.empty:
        fallback = result_df[
            (result_df["score"] >= min_score)
            & (result_df["directional_hit"] >= min_hit_rate)
        ].copy()
        if fallback.empty:
            fallback = result_df.head(min(top_k, 20)).copy()
        selected_df = fallback.head(top_k).copy()
    else:
        selected_df = selected_df.head(top_k).copy()

    selection_stats = {
        "eligible_before_top_k": selected_before_top_k,
        "selected_after_top_k": int(len(selected_df)),
        "top_k_events": int(top_k),
        "min_n": int(min_n),
        "min_abs_mean_ret": float(min_abs_mean_ret),
        "min_hit_rate": float(min_hit_rate),
        "min_score": float(min_score),
    }
    return selected_df["event"].tolist(), result_df, selected_df, selection_stats


def add_institution_features(base, path, target):
    stock_id = target.split(".")[0]
    p = Path(path)
    if not p.exists():
        return base

    inst = pd.read_csv(p)
    inst = inst[inst["stock_id"].astype(str) == stock_id].copy()
    if inst.empty:
        return base

    inst["date"] = pd.to_datetime(inst["date"])
    inst["net_buy"] = inst["buy"] - inst["sell"]
    wide = inst.pivot_table(index="date", columns="name", values="net_buy", aggfunc="sum")
    wide = wide.add_prefix("inst_net_")
    wide["inst_net_total"] = wide.sum(axis=1)
    return base.join(wide, how="left")


def add_margin_features(base, path, target):
    stock_id = target.split(".")[0]
    p = Path(path)
    if not p.exists():
        return base

    margin = pd.read_csv(p)
    margin = margin[margin["stock_id"].astype(str) == stock_id].copy()
    if margin.empty:
        return base

    margin["date"] = pd.to_datetime(margin["date"])
    margin = margin.set_index("date").sort_index()
    cols = [
        "MarginPurchaseTodayBalance",
        "ShortSaleTodayBalance",
        "MarginPurchaseBuy",
        "MarginPurchaseSell",
        "ShortSaleBuy",
        "ShortSaleSell",
    ]
    cols = [c for c in cols if c in margin.columns]
    margin = margin[cols].add_prefix("margin_")
    margin["margin_balance_chg"] = margin["margin_MarginPurchaseTodayBalance"].pct_change()
    margin["short_balance_chg"] = margin["margin_ShortSaleTodayBalance"].pct_change()
    return base.join(margin, how="left")


def add_futures_night_features(base, path, target=None):
    p = Path(path)
    if not p.exists():
        return base

    futures = pd.read_csv(p)
    if target is not None and "stock_id" in futures.columns:
        futures = futures[futures["stock_id"].astype(str) == target].copy()
    if futures.empty:
        return base

    futures["date"] = pd.to_datetime(futures["date"])
    futures = futures.sort_values("date").set_index("date")
    cols = [c for c in ["spread", "spread_per", "volume"] if c in futures.columns]
    if not cols:
        return base
    futures = futures[cols].add_prefix("tx_night_")
    if "tx_night_volume" in futures.columns:
        futures["tx_night_volume_z20"] = (
            futures["tx_night_volume"] - futures["tx_night_volume"].rolling(20).mean()
        ) / futures["tx_night_volume"].rolling(20).std()
    return base.join(futures, how="left")


def make_market_features(price_df, target):
    sample_dates = real_price_series(price_df, target).index
    decision_times = decision_times_for_samples(sample_dates, target)
    out = pd.DataFrame(index=sample_dates)

    def add_source(source):
        nonlocal out
        if source is None or source.empty:
            return
        feature_cols = [c for c in source.columns if c != "available_at"]
        aligned = asof_join_features(sample_dates, decision_times, source, feature_cols)
        out = out.join(aligned, how="left")

    add_source(price_feature_source(price_df, target, "target", "target"))

    for col in ["TSM", "^SOX", "^NDX", "^GSPC", "TWD=X"]:
        # 修改 #1:標的本身就是領先指標之一時跳過,避免產生與 target_ret_* 完全重複、
        # 且語意錯亂 (「領先指標」變「標的自我特徵」) 的欄位。
        if col == target:
            continue
        if col in price_df.columns:
            add_source(price_feature_source(price_df, col, col, "lead"))

    if "^VIX" in price_df.columns:
        add_source(price_feature_source(price_df, "^VIX", "vix", "vix"))

    if "^TNX" in price_df.columns:
        add_source(price_feature_source(price_df, "^TNX", "tnx", "tnx"))

    if target.endswith(".TW"):
        add_source(institution_feature_source(INSTITUTION_PATH, target))
        add_source(margin_feature_source(MARGIN_PATH, target))
        add_source(futures_night_feature_source(FUTURES_NIGHT_PATH, target=target))
    else:
        # 美股使用與台股同語意/schema 的對應資料檔:
        # us_institutional_investors.csv: date, stock_id, name, buy, sell
        # us_margin_trading.csv: date, stock_id, MarginPurchase*/ShortSale*
        # us_futures_night.csv: date, stock_id, spread, spread_per, volume
        add_source(institution_feature_source(US_INSTITUTION_PATH, target))
        add_source(margin_feature_source(US_MARGIN_PATH, target))
        add_source(futures_night_feature_source(US_FUTURES_NIGHT_PATH, target=target))
    return out


def add_regime_features(df):
    out = df.copy()
    vix_high = out.get("vix_high", pd.Series(0, index=out.index)).fillna(0) > 0
    vix_low = out.get("vix_low", pd.Series(0, index=out.index)).fillna(0) > 0
    bull = out["target_ma_gap_60"].fillna(0) > 0
    weak = out["target_ma_gap_20"].fillna(0) < -0.03
    oversold = out["target_drawdown_60"].fillna(0) < -0.08
    recent_drop = out["target_ret_5d"].fillna(0) < -0.04

    out["regime_calm_bull"] = (bull & vix_low).astype(float)
    out["regime_risk_off"] = (vix_high & (weak | recent_drop)).astype(float)
    out["regime_oversold"] = (oversold | recent_drop).astype(float)
    out["regime_neutral"] = (
        (out["regime_calm_bull"] == 0)
        & (out["regime_risk_off"] == 0)
        & (out["regime_oversold"] == 0)
    ).astype(float)

    out["regime_label"] = 3
    out.loc[out["regime_calm_bull"] == 1, "regime_label"] = 0
    out.loc[out["regime_risk_off"] == 1, "regime_label"] = 1
    out.loc[(out["regime_oversold"] == 1) & (out["regime_risk_off"] == 0), "regime_label"] = 2
    return out


def add_event_regime_interactions(df, selected_events):
    out = df.copy()
    regimes = ["regime_calm_bull", "regime_risk_off", "regime_oversold"]
    seed_events = [
        c
        for c in selected_events
        if any(key in c for key in ["tariff", "china", "taiwan", "chips", "deal", "relief"])
    ][:30]
    for event_col in seed_events:
        for regime_col in regimes:
            out[f"{event_col}__x__{regime_col}"] = out[event_col] * out[regime_col]
    return out


def build_model_frame(
    price_df,
    event_df,
    selected_events,
    combo_df,
    target,
    hold,
    binary_threshold,
    presidential_terms_only,
):
    market = make_market_features(price_df, target)
    events = combo_df[selected_events].copy()
    counts = event_df[[c for c in COUNT_EVENT_COLS if c in event_df.columns]].copy()
    frame = market.join(counts, how="left").join(events, how="left")
    event_like_cols = list(counts.columns) + selected_events
    frame[event_like_cols] = frame[event_like_cols].fillna(0)

    frame = add_regime_features(frame)
    frame = add_event_regime_interactions(frame, selected_events)

    # 修改 #3:用去 stale 後的價格序列算 future_ret,shift(-hold) 才會指向下一個「真正
    # 有交易」的收盤;stale (ffill) 日不在此序列 → future_ret 為 NaN,下方 dropna 會剔除。
    price = real_price_series(price_df, target)
    future_ret = price.shift(-hold) / price - 1
    frame["future_ret"] = future_ret.reindex(frame.index)
    frame["direction_label"] = (frame["future_ret"] > binary_threshold).astype(int)

    frame = frame.replace([np.inf, -np.inf], np.nan).sort_index()
    frame = frame.dropna(subset=["future_ret", "direction_label", "regime_label"])
    if presidential_terms_only:
        frame = filter_presidential_terms(frame)
    else:
        frame["term_segment"] = 1
    return frame


class FusionDataset(Dataset):
    def __init__(self, df, market_cols, event_cols, window, model_type):
        self.market_x = df[market_cols].astype(np.float32).values
        self.event_x = df[event_cols].astype(np.float32).values
        self.y = df["direction_label"].astype(np.int64).values
        self.regime_y = df["regime_label"].astype(np.int64).values
        self.future_ret = df["future_ret"].astype(np.float32).values
        self.dates = df.index.astype(str).to_numpy()
        self.term_segment = df["term_segment"].astype(np.int64).values
        self.window = window
        self.model_type = model_type
        self.indices = self._valid_start_indices()

    def _valid_start_indices(self):
        if self.model_type == "gated_mlp":
            return np.arange(len(self.y), dtype=np.int64)

        indices = []
        for segment in pd.unique(self.term_segment):
            positions = np.flatnonzero(self.term_segment == segment)
            if len(positions) <= self.window:
                continue
            for offset in range(0, len(positions) - self.window):
                start = positions[offset]
                end = positions[offset + self.window]
                if end - start == self.window:
                    indices.append(start)
        return np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def target_indices(self):
        if self.model_type == "gated_mlp":
            return self.indices
        return self.indices + self.window

    def __getitem__(self, idx):
        if self.model_type == "gated_mlp":
            end = int(self.indices[idx])
            market_x = self.market_x[end]
        else:
            start = int(self.indices[idx])
            end = start + self.window
            market_x = self.market_x[start:end]

        return {
            "market_x": torch.tensor(market_x),
            "event_x": torch.tensor(self.event_x[end]),
            "y": torch.tensor(self.y[end]),
            "regime_y": torch.tensor(self.regime_y[end]),
            "future_ret": torch.tensor(self.future_ret[end]),
            "date": self.dates[end],
        }


class RegimeFusionLSTM(nn.Module):
    def __init__(self, market_dim, event_dim, hidden_dim, dropout):
        super().__init__()
        self.market_lstm = nn.LSTM(
            market_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.regime_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, len(REGIME_LABELS)),
        )
        self.event_encoder = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.direction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(DIRECTION_LABELS)),
        )

    def forward(self, market_x, event_x):
        seq_out, _ = self.market_lstm(market_x)
        market_state = seq_out[:, -1, :]
        event_state = self.event_encoder(event_x)
        gate = self.gate(torch.cat([market_state, event_state], dim=1))
        fused_event = gate * event_state
        logits = self.direction_head(torch.cat([market_state, fused_event], dim=1))
        regime_logits = self.regime_head(market_state)
        return logits, regime_logits, gate


class RegimeFusionMLP(nn.Module):
    def __init__(self, market_dim, event_dim, hidden_dim, dropout):
        super().__init__()
        self.market_encoder = nn.Sequential(
            nn.Linear(market_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.event_encoder = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.regime_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, len(REGIME_LABELS)),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.direction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(DIRECTION_LABELS)),
        )

    def forward(self, market_x, event_x):
        market_state = self.market_encoder(market_x)
        event_state = self.event_encoder(event_x)
        gate = self.gate(torch.cat([market_state, event_state], dim=1))
        fused_event = gate * event_state
        logits = self.direction_head(torch.cat([market_state, fused_event], dim=1))
        regime_logits = self.regime_head(market_state)
        return logits, regime_logits, gate


def make_model(args, market_dim, event_dim):
    if args.model_type == "lstm":
        return RegimeFusionLSTM(
            market_dim=market_dim,
            event_dim=event_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )

    return RegimeFusionMLP(
        market_dim=market_dim,
        event_dim=event_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )


def split_and_scale(frame, market_cols, event_cols, window):
    n = len(frame)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = frame.iloc[:train_end].copy()
    val_df = frame.iloc[train_end - window : val_end].copy()
    test_df = frame.iloc[val_end - window :].copy()

    market_scaler = StandardScaler()
    event_scaler = StandardScaler()

    train_df[market_cols] = market_scaler.fit_transform(train_df[market_cols].fillna(0))
    val_df[market_cols] = market_scaler.transform(val_df[market_cols].fillna(0))
    test_df[market_cols] = market_scaler.transform(test_df[market_cols].fillna(0))

    train_df[event_cols] = event_scaler.fit_transform(train_df[event_cols].fillna(0))
    val_df[event_cols] = event_scaler.transform(val_df[event_cols].fillna(0))
    test_df[event_cols] = event_scaler.transform(test_df[event_cols].fillna(0))

    return train_df, val_df, test_df, market_scaler, event_scaler


def batch_to_device(batch, device):
    return {
        "market_x": batch["market_x"].to(device),
        "event_x": batch["event_x"].to(device),
        "y": batch["y"].to(device),
        "regime_y": batch["regime_y"].to(device),
        "future_ret": batch["future_ret"].to(device),
        "date": batch["date"],
    }


def evaluate(model, loader, criterion, regime_criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    preds = []
    labels = []
    regime_preds = []
    regime_labels = []
    returns = []
    dates = []
    probs = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = batch_to_device(raw_batch, device)
            logits, regime_logits, _ = model(batch["market_x"], batch["event_x"])
            loss = criterion(logits, batch["y"]) + 0.25 * regime_criterion(
                regime_logits, batch["regime_y"]
            )
            prob = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            regime_pred = regime_logits.argmax(dim=1)

            total_loss += float(loss.item()) * len(batch["y"])
            total += len(batch["y"])
            preds.extend(pred.cpu().numpy().tolist())
            labels.extend(batch["y"].cpu().numpy().tolist())
            regime_preds.extend(regime_pred.cpu().numpy().tolist())
            regime_labels.extend(batch["regime_y"].cpu().numpy().tolist())
            returns.extend(batch["future_ret"].cpu().numpy().tolist())
            dates.extend(list(raw_batch["date"]))
            probs.extend(prob.cpu().numpy().tolist())

    acc = accuracy_score(labels, preds) if labels else 0.0
    regime_acc = accuracy_score(regime_labels, regime_preds) if regime_labels else 0.0
    return {
        "loss": total_loss / max(total, 1),
        "acc": acc,
        "regime_acc": regime_acc,
        "preds": preds,
        "labels": labels,
        "regime_preds": regime_preds,
        "regime_labels": regime_labels,
        "returns": returns,
        "dates": dates,
        "probs": probs,
    }


def class_weight_tensor(dataset, num_classes, device):
    labels = dataset.y[dataset.target_indices()]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model(args, train_ds, val_ds, market_dim, event_dim, device):
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = make_model(args, market_dim, event_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    direction_weights = None
    if args.use_class_weights:
        direction_weights = class_weight_tensor(train_ds, len(DIRECTION_LABELS), device)
        print(f"Direction class weights: {direction_weights.detach().cpu().numpy().round(4).tolist()}")

    criterion = nn.CrossEntropyLoss(weight=direction_weights)
    regime_criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for raw_batch in train_loader:
            batch = batch_to_device(raw_batch, device)
            optimizer.zero_grad()
            logits, regime_logits, _ = model(batch["market_x"], batch["event_x"])
            loss = criterion(logits, batch["y"]) + 0.25 * regime_criterion(
                regime_logits, batch["regime_y"]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["y"])
            total += len(batch["y"])

        train_loss = total_loss / max(total, 1)
        val_metrics = evaluate(model, val_loader, criterion, regime_criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_regime_acc": val_metrics["regime_acc"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={row['val_loss']:.4f} | val_acc={row['val_acc']:.4f} | "
            f"val_regime_acc={row['val_regime_acc']:.4f}"
        )

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stop at epoch {epoch}; best_val_loss={best_val_loss:.4f}")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predictions_frame(metrics, trade_edge_threshold, trade_mode):
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(metrics["dates"]),
            "actual_label": metrics["labels"],
            "pred_label": metrics["preds"],
            "actual_regime": metrics["regime_labels"],
            "pred_regime": metrics["regime_preds"],
            "future_ret": metrics["returns"],
        }
    )
    probs = pd.DataFrame(
        metrics["probs"],
        columns=[f"prob_{DIRECTION_LABELS[i]}" for i in range(len(DIRECTION_LABELS))],
    )
    out = pd.concat([out, probs], axis=1)
    out["actual_label_name"] = out["actual_label"].map(DIRECTION_LABELS)
    out["pred_label_name"] = out["pred_label"].map(DIRECTION_LABELS)
    out["actual_regime_name"] = out["actual_regime"].map(REGIME_LABELS)
    out["pred_regime_name"] = out["pred_regime"].map(REGIME_LABELS)
    out["direction_edge"] = out["prob_up"] - out["prob_down"]
    raw_signal = np.where(out["direction_edge"] > 0, 1.0, -1.0)
    confident = out["direction_edge"].abs() >= trade_edge_threshold
    if trade_mode == "long_cash":
        signal = np.where(confident & (raw_signal > 0), 1.0, 0.0)
    elif trade_mode == "short_cash":
        signal = np.where(confident & (raw_signal < 0), -1.0, 0.0)
    else:
        signal = np.where(confident, raw_signal, 0.0)
    out["trade_signal"] = signal
    out["trade_decision"] = out["trade_signal"].map({-1.0: "SHORT", 0.0: "NEUTRAL", 1.0: "LONG"})
    out["strategy_ret_no_cost"] = out["trade_signal"] * out["future_ret"]
    return out


def strategy_metrics(pred_df):
    return {
        "trade_coverage": float((pred_df["trade_signal"] != 0).mean()),
        "neutral_count": int((pred_df["trade_signal"] == 0).sum()),
        "long_count": int((pred_df["trade_signal"] > 0).sum()),
        "short_count": int((pred_df["trade_signal"] < 0).sum()),
        "strategy_mean_return_no_cost": float(pred_df["strategy_ret_no_cost"].mean()),
        "strategy_total_return_no_cost": float((1 + pred_df["strategy_ret_no_cost"]).prod() - 1),
    }


def tune_trade_threshold(metrics, trade_mode, fallback_threshold, min_coverage):
    rows = []
    for threshold in np.round(np.arange(0.0, 0.501, 0.025), 3):
        pred_df = predictions_frame(metrics, threshold, trade_mode)
        row = {"trade_edge_threshold": float(threshold), **strategy_metrics(pred_df)}
        rows.append(row)

    search_df = pd.DataFrame(rows)
    eligible = search_df[search_df["trade_coverage"] >= min_coverage].copy()
    if eligible.empty:
        return fallback_threshold, search_df

    eligible = eligible.sort_values(
        ["strategy_total_return_no_cost", "trade_coverage"],
        ascending=[False, False],
    )
    return float(eligible.iloc[0]["trade_edge_threshold"]), search_df


def regime_conditioned_stats(frame, selected_events):
    rows = []
    regime_cols = ["regime_calm_bull", "regime_risk_off", "regime_oversold", "regime_neutral"]
    focus_events = [
        c
        for c in selected_events
        if c in frame.columns
        and any(key in c for key in ["tariff", "china", "taiwan", "chips", "deal", "relief"])
    ][:40]

    for event_col in focus_events:
        for regime_col in regime_cols:
            mask = (frame[event_col] > 0) & (frame[regime_col] > 0)
            n = int(mask.sum())
            if n < 5:
                continue
            r = frame.loc[mask, "future_ret"]
            rows.append(
                {
                    "event": event_col,
                    "regime": regime_col.replace("regime_", ""),
                    "n": n,
                    "mean_ret": float(r.mean()),
                    "median_ret": float(r.median()),
                    "hit_up": float((r > 0).mean()),
                    "hit_down": float((r < 0).mean()),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["event", "regime"]).reset_index(drop=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    price_df = read_price_data(PRICE_PATH, args.target)
    event_path = event_path_for_target(args.target)
    post_df = read_event_data(event_path)
    sample_dates = real_price_series(price_df, args.target).index
    decision_times = decision_times_for_samples(sample_dates, args.target)
    event_df = aggregate_posts_for_decisions(post_df, sample_dates, decision_times)
    print(f"Post-level event features: {event_path}")
    print("Trump events are aggregated from decision_time_D < Timestamp <= decision_time_D+1.")
    if args.presidential_terms_only:
        event_segment = presidential_term_segment(event_df.index)
        event_df = event_df.loc[event_segment > 0].copy()
    # 用去 stale 後的價格選事件,避免假日的假 0 報酬汙染 brute-force 篩選。
    price = real_price_series(price_df, args.target)
    combo_df, base_event_cols = make_event_combos(event_df)
    selected_events, brute_df, selected_df, selection_stats = brute_force_events(
        combo_df=combo_df,
        price=price,
        hold=args.hold,
        min_n=args.min_n,
        min_abs_mean_ret=args.min_abs_mean_ret,
        min_hit_rate=args.min_hit_rate,
        min_score=args.min_score,
        top_k=args.top_k_events,
    )

    print(f"Base binary event count: {len(base_event_cols)}")
    print(f"Event + combo count: {combo_df.shape[1]}")
    print(f"Eligible selected events before top-k: {selection_stats['eligible_before_top_k']}")
    print(f"Selected events for DL: {len(selected_events)}")
    print("\nTop selected brute-force events:")
    print(selected_df.head(20).to_string(index=False))

    frame = build_model_frame(
        price_df=price_df,
        event_df=event_df,
        selected_events=selected_events,
        combo_df=combo_df,
        target=args.target,
        hold=args.hold,
        binary_threshold=args.binary_threshold,
        presidential_terms_only=args.presidential_terms_only,
    )

    full_event_cols = selected_events + [
        c for c in frame.columns if "__x__regime_" in c
    ] + [c for c in COUNT_EVENT_COLS if c in frame.columns]
    full_event_cols = list(dict.fromkeys(full_event_cols))

    if args.feature_set == "market_only":
        frame["event_dummy_zero"] = 0.0
        all_event_cols = ["event_dummy_zero"]
    else:
        all_event_cols = full_event_cols

    excluded = set(
        full_event_cols
        + all_event_cols
        + ["future_ret", "direction_label", "regime_label", "term_segment"]
    )
    market_cols = [c for c in frame.columns if c not in excluded]

    usable = frame[
        market_cols + all_event_cols + ["future_ret", "direction_label", "regime_label", "term_segment"]
    ].copy()
    usable[market_cols + all_event_cols] = usable[market_cols + all_event_cols].fillna(0)
    usable = usable.dropna()

    if len(usable) <= args.window + 50:
        raise ValueError(
            f"Not enough usable rows ({len(usable)}) for window={args.window}. "
            "Lower --window or check input data."
        )

    train_df, val_df, test_df, market_scaler, event_scaler = split_and_scale(
        usable, market_cols, all_event_cols, args.window
    )
    train_ds = FusionDataset(train_df, market_cols, all_event_cols, args.window, args.model_type)
    val_ds = FusionDataset(val_df, market_cols, all_event_cols, args.window, args.model_type)
    test_ds = FusionDataset(test_df, market_cols, all_event_cols, args.window, args.model_type)

    print("\nDataset summary:")
    print(f"Rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Feature set: {args.feature_set}")
    print(f"Market features: {len(market_cols)} | Event/regime features: {len(all_event_cols)}")
    print("Direction label counts:")
    print(usable["direction_label"].map(DIRECTION_LABELS).value_counts().to_string())
    print("Regime label counts:")
    print(usable["regime_label"].map(REGIME_LABELS).value_counts().to_string())
    print("Term segment counts:")
    print(usable["term_segment"].value_counts().sort_index().to_string())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on device: {device}")
    print(f"Model type: {args.model_type}")
    model, history = train_model(
        args=args,
        train_ds=train_ds,
        val_ds=val_ds,
        market_dim=len(market_cols),
        event_dim=len(all_event_cols),
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    regime_criterion = nn.CrossEntropyLoss()
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    val_metrics = evaluate(model, val_loader, criterion, regime_criterion, device)
    selected_trade_edge_threshold = args.trade_edge_threshold
    threshold_search_df = pd.DataFrame()
    if args.auto_trade_threshold:
        selected_trade_edge_threshold, threshold_search_df = tune_trade_threshold(
            metrics=val_metrics,
            trade_mode=args.trade_mode,
            fallback_threshold=args.trade_edge_threshold,
            min_coverage=args.min_val_trade_coverage,
        )

    val_pred_df = predictions_frame(
        val_metrics,
        trade_edge_threshold=selected_trade_edge_threshold,
        trade_mode=args.trade_mode,
    )
    test_metrics = evaluate(model, test_loader, criterion, regime_criterion, device)
    pred_df = predictions_frame(
        test_metrics,
        trade_edge_threshold=selected_trade_edge_threshold,
        trade_mode=args.trade_mode,
    )

    y_true = pred_df["actual_label"].to_numpy()
    y_pred = pred_df["pred_label"].to_numpy()
    report = classification_report(
        y_true,
        y_pred,
        labels=list(DIRECTION_LABELS.keys()),
        target_names=[DIRECTION_LABELS[i] for i in DIRECTION_LABELS],
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(DIRECTION_LABELS.keys()))

    brute_df.to_csv(output_dir / "brute_force_all_events.csv", index=False)
    selected_df.to_csv(output_dir / "brute_force_selected_events.csv", index=False)
    history.to_csv(output_dir / "training_history.csv", index=False)
    val_pred_df.to_csv(output_dir / "validation_predictions.csv", index=False)
    threshold_search_df.to_csv(output_dir / "validation_trade_threshold_search.csv", index=False)
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
    regime_stats = regime_conditioned_stats(usable, selected_events)
    regime_stats.to_csv(output_dir / "event_regime_conditioned_stats.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "model_type": args.model_type,
            "feature_set": args.feature_set,
            "market_cols": market_cols,
            "event_cols": all_event_cols,
            "direction_labels": DIRECTION_LABELS,
            "regime_labels": REGIME_LABELS,
        },
        output_dir / "regime_fusion_model.pt",
    )
    joblib.dump(market_scaler, output_dir / "market_scaler.joblib")
    joblib.dump(event_scaler, output_dir / "event_scaler.joblib")

    summary = {
        "target": args.target,
        "hold": args.hold,
        "presidential_terms_only": args.presidential_terms_only,
        "presidential_terms": PRESIDENTIAL_TERMS,
        "binary_threshold": args.binary_threshold,
        "window": args.window,
        "model_type": args.model_type,
        "feature_set": args.feature_set,
        "trade_mode": args.trade_mode,
        "trade_edge_threshold": args.trade_edge_threshold,
        "selected_trade_edge_threshold": selected_trade_edge_threshold,
        "auto_trade_threshold": args.auto_trade_threshold,
        "min_val_trade_coverage": args.min_val_trade_coverage,
        "use_class_weights": args.use_class_weights,
        "device": device,
        "rows": {
            "usable": int(len(usable)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "term_segment_counts": {
            str(k): int(v) for k, v in usable["term_segment"].value_counts().sort_index().items()
        },
        "features": {
            "market": len(market_cols),
            "event_regime": len(all_event_cols),
            "selected_events": len(selected_events),
        },
        "event_selection": selection_stats,
        "validation_strategy": strategy_metrics(val_pred_df),
        "test": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["acc"],
            "regime_accuracy": test_metrics["regime_acc"],
            **strategy_metrics(pred_df),
        },
        "classification_report": report,
        "confusion_matrix_labels": [DIRECTION_LABELS[i] for i in DIRECTION_LABELS],
        "confusion_matrix": cm.tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n==============================")
    print(f"Target: {args.target}")
    print(f"Selected brute-force events: {len(selected_events)}")
    print(f"Test acc: {test_metrics['acc']:.4f}")
    print(f"Test regime acc: {test_metrics['regime_acc']:.4f}")
    print(
        f"Selected trade edge threshold: {selected_trade_edge_threshold:.3f} "
        f"(auto={args.auto_trade_threshold})"
    )
    print(
        f"Trade mode: {args.trade_mode} | "
        f"coverage={summary['test']['trade_coverage']:.2%} | "
        f"long={summary['test']['long_count']} | short={summary['test']['short_count']} | "
        f"neutral={summary['test']['neutral_count']}"
    )
    print(
        "Strategy return no cost: "
        f"mean={summary['test']['strategy_mean_return_no_cost']:.6f}, "
        f"total={summary['test']['strategy_total_return_no_cost']:.4f}"
    )
    print("Confusion matrix rows=true, cols=pred [down, up]:")
    print(cm)
    print(f"Saved outputs to: {output_dir}")
    print("==============================")


if __name__ == "__main__":
    main()
