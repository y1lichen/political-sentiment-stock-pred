import re
import numpy as np
import pandas as pd


INPUT_FILE = "./data/text/trump_posts_features_2017_2026.csv"
PRICE_FILE = "./data/taiwan_market_data/global_prices.csv"
POST_OUTPUT = "./data/output/trump_posts_with_event_features.csv"
DAILY_OUTPUT = "./data/output/trump_daily_binary_event_features.csv"
POST_OUTPUT_US = "./data/output/trump_posts_with_event_features_us.csv"
DAILY_OUTPUT_US = "./data/output/trump_daily_binary_event_features_us.csv"
POST_OUTPUT_TW = "./data/output/trump_posts_with_event_features_tw.csv"
DAILY_OUTPUT_TW = "./data/output/trump_daily_binary_event_features_tw.csv"


def contains_any(text, words):
    text = str(text).lower()
    return int(any(w in text for w in words))


def load_trading_calendar(market):
    try:
        price_df = pd.read_csv(PRICE_FILE)
    except FileNotFoundError:
        return None

    price_df["Date"] = pd.to_datetime(price_df["Date"]).dt.normalize()
    candidates = ["^GSPC", "TSM"] if market == "us" else ["0050.TW", "2330.TW"]
    ticker = next((c for c in candidates if c in price_df.columns), None)
    if ticker is None:
        return None

    series = price_df[["Date", ticker]].dropna().copy()
    # global_prices 用 ffill 對齊不同市場日曆; diff==0 多半是休市 carry-forward。
    real_trade = series[ticker].diff().fillna(1) != 0
    calendar = series.loc[real_trade, "Date"].drop_duplicates().sort_values()
    return calendar.reset_index(drop=True)


def map_to_next_trading_day(dates, trading_calendar):
    dates = pd.to_datetime(dates).dt.normalize()
    if trading_calendar is None or len(trading_calendar) == 0:
        return dates

    calendar = pd.to_datetime(trading_calendar).dt.normalize().sort_values().to_numpy()
    mapped = []
    for value in dates.to_numpy():
        if pd.isna(value):
            mapped.append(pd.NaT)
            continue
        pos = np.searchsorted(calendar, value, side="left")
        mapped.append(pd.NaT if pos >= len(calendar) else pd.Timestamp(calendar[pos]))
    return pd.Series(mapped, index=dates.index)


def add_post_level_features(df, market="us", trading_calendar=None):
    df = df.copy()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    if market == "tw":
        local_tz = "Asia/Taipei"
        open_hour, open_minute = 9, 0
        close_hour, close_minute = 13, 30
    else:
        local_tz = "America/New_York"
        open_hour, open_minute = 9, 30
        close_hour, close_minute = 16, 0

    df["Timestamp_Local"] = df["Timestamp"].dt.tz_convert(local_tz)
    df["hour_local"] = df["Timestamp_Local"].dt.hour
    df["minute_local"] = df["Timestamp_Local"].dt.minute
    local_date = df["Timestamp_Local"].dt.tz_localize(None).dt.normalize()
    # 嚴格時間對齊: 不論美股或台股，Trump 貼文都只影響下一個有效交易日。
    # 這避免同日盤中/盤前資訊被放進 close_t -> close_t+1 的訊號日。
    raw_effective_date = local_date + pd.Timedelta(days=1)
    effective_date = map_to_next_trading_day(raw_effective_date, trading_calendar)
    df["trump_date"] = effective_date.dt.date

    df["Content"] = df["Content"].astype(str)
    lower = df["Content"].str.lower()

    # 時間事件
    df["is_pre_market_post"] = (
        (df["hour_local"] < open_hour) |
        ((df["hour_local"] == open_hour) & (df["minute_local"] < open_minute))
    ).astype(int)

    df["is_market_open_post"] = (
        (
            (df["hour_local"] > open_hour) |
            ((df["hour_local"] == open_hour) & (df["minute_local"] >= open_minute))
        ) &
        (
            (df["hour_local"] < close_hour) |
            ((df["hour_local"] == close_hour) & (df["minute_local"] < close_minute))
        )
    ).astype(int)

    df["is_night_post"] = (
        (df["hour_local"] < 5) | (df["hour_local"] >= 23)
    ).astype(int)

    # 文字型態
    df["post_len"] = df["Content"].str.len()
    df["exclamation_count"] = df["Content"].str.count("!")
    df["question_count"] = df["Content"].str.count(r"\?")

    df["alpha_count"] = df["Content"].apply(lambda x: sum(ch.isalpha() for ch in x))
    df["caps_count"] = df["Content"].apply(lambda x: sum(ch.isupper() for ch in x))
    df["caps_alpha_ratio"] = df["caps_count"] / df["alpha_count"].replace(0, np.nan)
    df["caps_alpha_ratio"] = df["caps_alpha_ratio"].fillna(0.0)

    # trump_code 類別事件
    df["ev_tariff"] = (
        lower.str.contains(r"\btariffs?\b|\bdut(?:y|ies)\b", regex=True) |
        (df.get("kw_tariffs", 0) == 1)
    ).astype(int)

    df["ev_deal"] = lower.str.contains(
        r"\bdeal\b|\bagreement\b|\bsigned\b|\bnegotiate\b|\bnegotiation\b|\btrade deal\b",
        regex=True
    ).astype(int)

    df["ev_relief"] = lower.str.contains(
        r"\bpause\b|\bexempt\b|\bexemption\b|\bsuspend\b|\bdelay\b|\bdelayed\b|\brelief\b",
        regex=True
    ).astype(int)

    df["ev_action"] = lower.str.contains(
        r"\bimmediately\b|\bhereby\b|\bexecutive order\b|\bjust signed\b|\bordered\b|\bdeclare\b",
        regex=True
    ).astype(int)

    df["ev_attack"] = lower.str.contains(
        r"\bfake news\b|\bcorrupt\b|\bfraud\b|\bwitch hunt\b|\bhoax\b|\bliar\b|\bdisaster\b",
        regex=True
    ).astype(int)

    df["ev_positive"] = lower.str.contains(
        r"\bgreat\b|\btremendous\b|\bincredible\b|\bhistoric\b|\bbeautiful\b|\bstrong\b|\bwin\b",
        regex=True
    ).astype(int)

    df["ev_market_brag"] = lower.str.contains(
        r"\bstock market\b|\ball[- ]time high\b|\brecord high\b|\bdow\b|\bnasdaq\b|\bs&p\b",
        regex=True
    ).astype(int)

    # 你專案需要的延伸事件
    for col in [
        "kw_china", "kw_taiwan", "kw_chips", "kw_ai",
        "kw_tech", "kw_sanctions", "kw_supply_chain",
        "kw_military", "kw_market"
    ]:
        if col not in df.columns:
            df[col] = 0

    df["ev_china"] = ((df["kw_china"] == 1) | lower.str.contains(r"\bchina\b|\bchinese\b|\bbeijing\b", regex=True)).astype(int)
    df["ev_taiwan"] = ((df["kw_taiwan"] == 1) | lower.str.contains(r"\btaiwan\b|\btsmc\b|\btaipei\b", regex=True)).astype(int)
    df["ev_chips"] = ((df["kw_chips"] == 1) | lower.str.contains(r"\bchips?\b|\bsemiconductor\b|\bnvidia\b|\btsmc\b", regex=True)).astype(int)
    df["ev_ai"] = ((df["kw_ai"] == 1) | lower.str.contains(r"\bai\b|\bartificial intelligence\b|\bopenai\b|\bchatgpt\b", regex=True)).astype(int)

    df["ev_iran"] = lower.str.contains(r"\biran\b|\biranian\b", regex=True).astype(int)
    df["ev_russia"] = lower.str.contains(r"\brussia\b|\bputin\b|\bukraine\b", regex=True).astype(int)

    # 特殊 signature
    df["sig_djt"] = df["Content"].str.contains("President DJT", case=False, regex=False).astype(int)
    df["sig_potus"] = df["Content"].str.contains("PRESIDENT OF THE UNITED STATES", case=False, regex=False).astype(int)
    df["sig_tyfa"] = df["Content"].str.contains("Thank you for your attention", case=False, regex=False).astype(int)

    return df


def build_daily_binary_features(post_df):
    df = post_df.copy()

    start = pd.to_datetime(df["trump_date"].min())
    end = pd.to_datetime(df["trump_date"].max())
    full_dates = pd.date_range(start, end, freq="D").date

    agg = df.groupby("trump_date").agg(
        post_count=("Content", "size"),

        tariff_count=("ev_tariff", "sum"),
        deal_count=("ev_deal", "sum"),
        relief_count=("ev_relief", "sum"),
        action_count=("ev_action", "sum"),
        attack_count=("ev_attack", "sum"),
        positive_count=("ev_positive", "sum"),
        market_brag_count=("ev_market_brag", "sum"),

        china_count=("ev_china", "sum"),
        taiwan_count=("ev_taiwan", "sum"),
        chips_count=("ev_chips", "sum"),
        ai_count=("ev_ai", "sum"),
        iran_count=("ev_iran", "sum"),
        russia_count=("ev_russia", "sum"),

        pre_tariff_count=("ev_tariff", lambda x: 0),
        total_excl=("exclamation_count", "sum"),
        total_caps=("caps_count", "sum"),
        total_alpha=("alpha_count", "sum"),
        avg_post_len=("post_len", "mean"),

        night_post_count=("is_night_post", "sum"),
        pre_post_count=("is_pre_market_post", "sum"),
        open_post_count=("is_market_open_post", "sum"),

        sig_djt_count=("sig_djt", "sum"),
        sig_potus_count=("sig_potus", "sum"),
        sig_tyfa_count=("sig_tyfa", "sum"),
    )

    # 重新計算需要同時看時間與事件的 count
    special = df.groupby("trump_date").apply(lambda g: pd.Series({
        "pre_tariff_count": ((g["is_pre_market_post"] == 1) & (g["ev_tariff"] == 1)).sum(),
        "pre_deal_count": ((g["is_pre_market_post"] == 1) & (g["ev_deal"] == 1)).sum(),
        "pre_relief_count": ((g["is_pre_market_post"] == 1) & (g["ev_relief"] == 1)).sum(),
        "pre_action_count": ((g["is_pre_market_post"] == 1) & (g["ev_action"] == 1)).sum(),
        "open_tariff_count": ((g["is_market_open_post"] == 1) & (g["ev_tariff"] == 1)).sum(),
        "open_deal_count": ((g["is_market_open_post"] == 1) & (g["ev_deal"] == 1)).sum(),
    }))

    agg = agg.drop(columns=["pre_tariff_count"])
    agg = agg.join(special, how="left")

    daily = pd.DataFrame(index=full_dates)
    daily.index.name = "trump_date"
    daily = daily.join(agg, how="left").fillna(0)

    # trump_code 原始風格二元特徵
    out = pd.DataFrame(index=daily.index)

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

    out["lots_of_excl"] = (daily["total_excl"] >= 5).astype(int)
    out["long_posts"] = (daily["avg_post_len"] > 400).astype(int)
    out["short_posts"] = ((daily["avg_post_len"] < 150) & (daily["post_count"] > 0)).astype(int)

    # 組合型事件
    out["deal_over_tariff"] = (
        (daily["deal_count"] > daily["tariff_count"]) &
        (daily["deal_count"] >= 1)
    ).astype(int)

    out["tariff_only"] = (
        (daily["tariff_count"] >= 1) &
        (daily["deal_count"] == 0)
    ).astype(int)

    out["deal_only"] = (
        (daily["deal_count"] >= 1) &
        (daily["tariff_count"] == 0)
    ).astype(int)

    # 你專案額外需要的科技 / 台灣 / 半導體事件
    out["has_taiwan"] = (daily["taiwan_count"] >= 1).astype(int)
    out["has_chips"] = (daily["chips_count"] >= 1).astype(int)
    out["chips_heavy"] = (daily["chips_count"] >= 2).astype(int)
    out["has_ai"] = (daily["ai_count"] >= 1).astype(int)
    out["china_taiwan_combo"] = ((daily["china_count"] >= 1) & (daily["taiwan_count"] >= 1)).astype(int)
    out["china_chips_combo"] = ((daily["china_count"] >= 1) & (daily["chips_count"] >= 1)).astype(int)
    out["taiwan_chips_combo"] = ((daily["taiwan_count"] >= 1) & (daily["chips_count"] >= 1)).astype(int)

    # 時序事件：只用過去資料，避免 leakage
    tariff_active = (daily["tariff_count"] >= 1).astype(int)
    china_active = (daily["china_count"] >= 1).astype(int)
    chips_active = (daily["chips_count"] >= 1).astype(int)

    out["tariff_streak_3d"] = (
        tariff_active.shift(1).rolling(3, min_periods=3).sum() >= 3
    ).fillna(False).astype(int)

    out["tariff_rising"] = (
        (tariff_active.shift(1).rolling(3, min_periods=3).sum() >= 2) &
        (daily["tariff_count"] >= 1)
    ).fillna(False).astype(int)

    out["china_streak_3d"] = (
        china_active.shift(1).rolling(3, min_periods=3).sum() >= 3
    ).fillna(False).astype(int)

    out["chips_streak_3d"] = (
        chips_active.shift(1).rolling(3, min_periods=3).sum() >= 3
    ).fillna(False).astype(int)

    prev_7_post_avg = daily["post_count"].shift(1).rolling(7, min_periods=3).mean()
    out["volume_spike"] = (
        daily["post_count"] > prev_7_post_avg * 2
    ).fillna(False).astype(int)

    out["volume_drop"] = (
        daily["post_count"] < prev_7_post_avg * 0.4
    ).fillna(False).astype(int)

    # 保留一些 daily count，之後做深度學習也很有用
    count_cols = [
        "post_count", "tariff_count", "deal_count", "relief_count",
        "china_count", "taiwan_count", "chips_count", "ai_count",
        "night_post_count", "pre_post_count", "open_post_count",
        "total_excl", "avg_post_len"
    ]

    final = pd.concat([daily[count_cols], out], axis=1)
    final = final.reset_index()
    final["trump_date"] = pd.to_datetime(final["trump_date"])

    return final


def main():
    df = pd.read_csv(INPUT_FILE)
    us_calendar = load_trading_calendar("us")
    tw_calendar = load_trading_calendar("tw")

    post_df = add_post_level_features(df, market="us", trading_calendar=us_calendar)
    daily_df = build_daily_binary_features(post_df)
    post_df_us = post_df
    daily_df_us = daily_df
    post_df_tw = add_post_level_features(df, market="tw", trading_calendar=tw_calendar)
    daily_df_tw = build_daily_binary_features(post_df_tw)

    post_df.to_csv(POST_OUTPUT, index=False)
    daily_df.to_csv(DAILY_OUTPUT, index=False)
    post_df_us.to_csv(POST_OUTPUT_US, index=False)
    daily_df_us.to_csv(DAILY_OUTPUT_US, index=False)
    post_df_tw.to_csv(POST_OUTPUT_TW, index=False)
    daily_df_tw.to_csv(DAILY_OUTPUT_TW, index=False)

    binary_cols = [
        c for c in daily_df.columns
        if c not in [
            "trump_date", "post_count", "tariff_count", "deal_count",
            "relief_count", "china_count", "taiwan_count", "chips_count",
            "ai_count", "night_post_count", "pre_post_count",
            "open_post_count", "total_excl", "avg_post_len"
        ]
    ]

    print(f"Saved post-level features to: {POST_OUTPUT}")
    print(f"Saved daily binary event features to: {DAILY_OUTPUT}")
    print(f"Saved US post-level features to: {POST_OUTPUT_US}")
    print(f"Saved US daily binary event features to: {DAILY_OUTPUT_US}")
    print(f"Saved TW post-level features to: {POST_OUTPUT_TW}")
    print(f"Saved TW daily binary event features to: {DAILY_OUTPUT_TW}")
    print(f"Daily rows: {len(daily_df)}")
    print(f"TW daily rows: {len(daily_df_tw)}")
    print(f"Binary event feature count: {len(binary_cols)}")
    print("\nTop event frequencies:")
    print(daily_df[binary_cols].sum().sort_values(ascending=False).head(30))


if __name__ == "__main__":
    main()
