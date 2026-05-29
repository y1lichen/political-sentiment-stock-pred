from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_TARGET, Paths
from src.utils.calendar import assign_effective_trade_date
from src.utils.io import ensure_dir, safe_name, write_json


TRUMP_SUM_COLS = [
    "kw_china",
    "kw_taiwan",
    "kw_tariffs",
    "kw_sanctions",
    "kw_chips",
    "kw_tech",
    "kw_ai",
    "kw_market",
    "kw_supply_chain",
    "kw_military",
    "tc_tariff",
    "tc_deal",
    "tc_relief",
    "tc_action",
    "tc_attack",
    "tc_positive",
    "tc_market_brag",
    "tc_iran",
    "tc_russia",
    "tc_fed",
    "tc_energy",
    "tc_pre_tariff",
    "tc_pre_deal",
    "tc_pre_relief",
    "tc_pre_action",
    "tc_open_tariff",
    "tc_open_deal",
    "tc_night_tariff",
    "tc_deal_over_tariff_post",
    "tc_tariff_only_post",
    "tc_relief_positive_post",
    "tc_attack_market_post",
    "is_night_post",
    "is_market_hours",
    "is_pre_market",
    "is_after_market",
    "sig_djt",
    "sig_potus",
    "sig_tyfa",
    "is_retweet_text",
    "exclamation_count",
    "question_count",
    "hashtag_count",
    "mention_count",
    "url_count",
    "repeated_exclamation",
    "repeated_question",
]

TRUMP_MEAN_COLS = [
    "vader_compound",
    "weighted_vader",
    "emotion_score",
    "char_count",
    "word_count",
    "avg_word_length",
    "caps_ratio",
    "uppercase_char_ratio",
    "exclamation_ratio",
    "question_ratio",
    "emotional_intensity",
    "vader_pos",
    "vader_neg",
    "vader_neu",
    "log_likes",
    "log_retweets",
    "engagement_score",
    "viral_score",
    "keyword_density",
    "event_score",
    "tc_directional_pressure",
    "tc_event_intensity",
]

CONTROL_TICKERS = ["TSM", "TWD=X", "^GSPC", "^NDX", "^SOX", "^TNX", "^VIX"]
TARGET_TICKERS = ["2330.TW", "2454.TW", "0050.TW"]


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    df["is_president"] = (
        ((d >= "2017-01-20") & (d <= "2021-01-20"))
        | (d >= "2025-01-20")
    ).astype(int)
    df["first_term"] = ((d >= "2017-01-20") & (d <= "2021-01-20")).astype(int)
    df["post_presidency"] = ((d >= "2021-01-21") & (d <= "2025-01-19")).astype(int)
    df["second_term"] = (d >= "2025-01-20").astype(int)
    df["campaign_period"] = (
        ((d >= "2020-06-01") & (d <= "2020-11-30"))
        | ((d >= "2024-06-01") & (d <= "2024-11-30"))
    ).astype(int)
    df["covid_crash_period"] = ((d >= "2020-02-01") & (d <= "2020-04-30")).astype(int)
    df["covid_recovery_liquidity_period"] = (
        (d >= "2020-05-01") & (d <= "2021-12-31")
    ).astype(int)
    df["covid_policy_period"] = ((d >= "2020-02-01") & (d <= "2021-12-31")).astype(int)
    df["policy_power_score"] = df["is_president"].astype(float)
    df.loc[df["campaign_period"].eq(1) & df["is_president"].eq(0), "policy_power_score"] = 0.4
    return df


def load_trading_calendar(market_dir: Path) -> list[pd.Timestamp]:
    prices = pd.read_csv(market_dir / "global_prices.csv", usecols=["Date"])
    dates = pd.to_datetime(prices["Date"]).dropna().drop_duplicates().sort_values()
    return [pd.Timestamp(d).normalize() for d in dates]


def build_trump_daily(trump_path: Path, trading_dates: list[pd.Timestamp]) -> pd.DataFrame:
    df = pd.read_csv(trump_path)
    df["effective_trade_date"] = assign_effective_trade_date(df["Timestamp"], trading_dates)
    df = df.dropna(subset=["effective_trade_date"]).copy()
    df["effective_trade_date"] = pd.to_datetime(df["effective_trade_date"]).dt.normalize()

    sum_cols = [c for c in TRUMP_SUM_COLS if c in df.columns]
    mean_cols = [c for c in TRUMP_MEAN_COLS if c in df.columns]
    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "mean" for c in mean_cols})
    daily = df.groupby("effective_trade_date").agg(agg)
    daily.columns = [
        f"trump_sum_{c}" if c in sum_cols else f"trump_avg_{c}"
        for c in daily.columns
    ]
    daily["trump_post_count"] = df.groupby("effective_trade_date").size()

    if "emotion_label" in df.columns:
        emo = pd.crosstab(df["effective_trade_date"], df["emotion_label"])
        emo = emo.add_prefix("trump_emotion_count_")
        daily = daily.join(emo, how="left")

    if "Platform" in df.columns:
        plat = pd.crosstab(df["effective_trade_date"], df["Platform"])
        plat = plat.add_prefix("trump_platform_count_")
        daily = daily.join(plat, how="left")

    out = pd.DataFrame({"date": trading_dates})
    out = out.merge(daily.reset_index().rename(columns={"effective_trade_date": "date"}), on="date", how="left")
    trump_cols = [c for c in out.columns if c != "date"]
    out[trump_cols] = out[trump_cols].fillna(0.0)
    out["trump_has_event"] = (out["trump_post_count"] > 0).astype(int)

    event_components = [
        "trump_sum_tc_tariff",
        "trump_sum_tc_action",
        "trump_sum_tc_attack",
        "trump_sum_tc_deal",
        "trump_sum_kw_china",
        "trump_sum_kw_taiwan",
        "trump_sum_kw_chips",
    ]
    present = [c for c in event_components if c in out.columns]
    out["trump_policy_event_count"] = out[present].sum(axis=1) if present else 0.0
    out["event_gate_default"] = (
        (out["trump_policy_event_count"] > 0)
        | (out.get("trump_avg_tc_event_intensity", 0) > 0)
    ).astype(int)
    return out


def add_market_features(market_dir: Path, target: str) -> pd.DataFrame:
    prices = pd.read_csv(market_dir / "global_prices.csv")
    prices["date"] = pd.to_datetime(prices.pop("Date")).dt.normalize()
    prices = prices.sort_values("date")

    vols = pd.read_csv(market_dir / "global_volumes.csv")
    vols["date"] = pd.to_datetime(vols.pop("Date")).dt.normalize()
    vols = vols.sort_values("date")

    out = prices[["date", target]].rename(columns={target: "target_close"}).copy()
    target_ret = prices[target].pct_change()
    out["target_return_1d"] = target_ret
    out["target_direction_1d"] = (out["target_return_1d"] > 0).astype(int)
    out["target_big_move_1d"] = (
        out["target_return_1d"].abs()
        > target_ret.rolling(60, min_periods=20).std().shift(1)
    ).astype(int)

    tickers = [c for c in TARGET_TICKERS + CONTROL_TICKERS if c in prices.columns]
    for ticker in tickers:
        s = safe_name(ticker)
        ret = prices[ticker].pct_change()
        out[f"mkt_{s}_ret_lag1"] = ret.shift(1)
        out[f"mkt_{s}_ret_3d_lag1"] = prices[ticker].pct_change(3).shift(1)
        out[f"mkt_{s}_ret_5d_lag1"] = prices[ticker].pct_change(5).shift(1)
        out[f"mkt_{s}_volatility_20d_lag1"] = ret.rolling(20, min_periods=5).std().shift(1)

    for ticker in tickers:
        if ticker in vols.columns:
            s = safe_name(ticker)
            v = vols[ticker].replace(0, np.nan)
            out[f"vol_{s}_change_5d_lag1"] = v.pct_change(5).shift(1)

    if "^VIX" in prices.columns:
        vix = prices["^VIX"]
        roll_q = vix.rolling(756, min_periods=100).quantile(0.8).shift(1)
        out["high_vix_regime"] = (vix.shift(1) > roll_q).astype(int)
    else:
        out["high_vix_regime"] = 0

    stress_parts = [c for c in ["mkt_idx_VIX_ret_lag1", "mkt_idx_SOX_ret_lag1", "mkt_idx_GSPC_ret_lag1"] if c in out.columns]
    if stress_parts:
        out["market_stress_score"] = out[stress_parts].fillna(0).abs().sum(axis=1)
    else:
        out["market_stress_score"] = 0.0
    return out


def add_institutional_features(market_dir: Path, target: str, base: pd.DataFrame) -> pd.DataFrame:
    inst_path = market_dir / "institutional_investors.csv"
    stock_id = target.split(".")[0]
    if not inst_path.exists():
        return base
    inst = pd.read_csv(inst_path)
    inst = inst[inst["stock_id"].astype(str).eq(stock_id)].copy()
    if inst.empty:
        return base
    inst["date"] = pd.to_datetime(inst["date"]).dt.normalize()
    inst["net_buy"] = inst["buy"] - inst["sell"]
    piv = inst.pivot_table(index="date", columns="name", values="net_buy", aggfunc="sum").add_prefix("inst_net_")
    piv = piv.sort_index()
    for c in piv.columns:
        piv[f"{c}_lag1"] = piv[c].shift(1)
        piv[f"{c}_sum5_lag1"] = piv[c].rolling(5, min_periods=1).sum().shift(1)
    piv = piv[[c for c in piv.columns if c.endswith("_lag1")]]
    return base.merge(piv.reset_index(), on="date", how="left")


def add_margin_features(market_dir: Path, target: str, base: pd.DataFrame) -> pd.DataFrame:
    path = market_dir / "margin_trading.csv"
    stock_id = target.split(".")[0]
    if not path.exists():
        return base
    m = pd.read_csv(path)
    m = m[m["stock_id"].astype(str).eq(stock_id)].copy()
    if m.empty:
        return base
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    keep = [
        "date",
        "MarginPurchaseTodayBalance",
        "ShortSaleTodayBalance",
        "MarginPurchaseBuy",
        "MarginPurchaseSell",
        "ShortSaleBuy",
        "ShortSaleSell",
    ]
    m = m[[c for c in keep if c in m.columns]].sort_values("date")
    for c in [c for c in m.columns if c != "date"]:
        m[f"margin_{c}_lag1"] = m[c].shift(1)
        m[f"margin_{c}_chg5_lag1"] = m[c].replace(0, np.nan).pct_change(5).shift(1)
    m = m[["date"] + [c for c in m.columns if c.startswith("margin_")]]
    return base.merge(m, on="date", how="left")


def add_tx_features(market_dir: Path, base: pd.DataFrame) -> pd.DataFrame:
    path = market_dir / "tx_futures_night.csv"
    if not path.exists():
        return base
    tx = pd.read_csv(path)
    tx["date"] = pd.to_datetime(tx["date"]).dt.normalize()
    tx = tx.sort_values("date")
    cols = ["spread", "spread_per", "volume", "open", "max", "min", "close"]
    tx = tx[["date"] + [c for c in cols if c in tx.columns]]
    for c in [c for c in tx.columns if c != "date"]:
        tx[f"tx_night_{c}_lag1"] = tx[c].shift(1)
    tx = tx[["date"] + [c for c in tx.columns if c.startswith("tx_night_")]]
    return base.merge(tx, on="date", how="left")


def build_modeling_table(trump_path: Path, market_dir: Path, target: str) -> pd.DataFrame:
    trading_dates = load_trading_calendar(market_dir)
    trump_daily = build_trump_daily(trump_path, trading_dates)
    market = add_market_features(market_dir, target)
    df = market.merge(trump_daily, on="date", how="left")
    df = add_regime_features(df)
    df = add_institutional_features(market_dir, target, df)
    df = add_margin_features(market_dir, target, df)
    df = add_tx_features(market_dir, df)

    if "trump_sum_tc_tariff" in df.columns:
        df["tariff_regime_intensity"] = (
            df["trump_sum_tc_tariff"].rolling(20, min_periods=1).sum().shift(1).fillna(0)
        )
    else:
        df["tariff_regime_intensity"] = 0.0

    df = df.sort_values("date").replace([np.inf, -np.inf], np.nan)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Deep Trump Code modeling table.")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--trump-path", type=Path, default=Paths().trump_posts)
    parser.add_argument("--market-dir", type=Path, default=Paths().market_dir)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or (Paths().datasets_dir / f"modeling_table_{safe_name(args.target)}.csv")
    ensure_dir(output.parent)

    df = build_modeling_table(args.trump_path, args.market_dir, args.target)
    df.to_csv(output, index=False)
    write_json(
        {
            "target": args.target,
            "rows": int(len(df)),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "columns": list(df.columns),
            "output": str(output),
        },
        output.with_suffix(".metadata.json"),
    )
    print(f"Wrote {output} with shape {df.shape}")


if __name__ == "__main__":
    main()

