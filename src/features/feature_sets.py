from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.utils.io import safe_name


CANONICAL_FEATURE_SETS = (
    "TW_self_only",
    "TW_market_only",
    "TW_plus_global_market",
    "Trump_text_only",
    "TW_plus_Trump",
    "Global_plus_Trump_no_gate",
    "Global_plus_Trump_with_gate",
)

FEATURE_SET_ALIASES = {
    "full": "Global_plus_Trump_with_gate",
    "market_only": "TW_plus_global_market",
}

TARGET_COLUMNS = {
    "target_return_1d",
    "target_direction_1d",
    "target_big_move_1d",
    "target_close",
}
DATE_COLUMNS = {"date", "Date", "timestamp", "Timestamp", "datetime", "Datetime"}

TW_MARKET_SYMBOLS = ("2330_TW", "2454_TW", "0050_TW")
GLOBAL_MARKET_SYMBOLS = (
    "TSM",
    "TWD_X",
    "idx_GSPC",
    "idx_NDX",
    "idx_SOX",
    "idx_TNX",
    "idx_VIX",
)

TRUMP_REGIME_COLUMNS = (
    "is_president",
    "first_term",
    "post_presidency",
    "second_term",
    "campaign_period",
    "policy_power_score",
    "tariff_regime_intensity",
)
EVENT_GATE_COLUMN = "event_gate_default"
MARKET_STATE_COLUMNS = ("high_vix_regime", "market_stress_score")

_CANONICAL_LOOKUP = {name.lower(): name for name in CANONICAL_FEATURE_SETS}


def resolve_feature_set(name: str) -> str:
    """Resolve a feature-set name or alias to its canonical name."""
    key = str(name)
    if key in CANONICAL_FEATURE_SETS:
        return key

    normalized = key.lower()
    if normalized in FEATURE_SET_ALIASES:
        return FEATURE_SET_ALIASES[normalized]
    if normalized in _CANONICAL_LOOKUP:
        return _CANONICAL_LOOKUP[normalized]

    valid = [*CANONICAL_FEATURE_SETS, *FEATURE_SET_ALIASES]
    raise ValueError(f"Unknown feature set: {name}. Valid names: {valid}")


def build_feature_set(
    df: pd.DataFrame,
    target: str,
    feature_set: str,
    feature_budget: int = 80,
    all_features: bool = False,
) -> tuple[str, list[str]]:
    """Build the ordered feature list for one ablation feature set."""
    canonical_name = resolve_feature_set(feature_set)
    legal_columns = _legal_numeric_columns(df, target)
    features = [
        column
        for column in legal_columns
        if _matches_feature_set(column, target, canonical_name)
    ]

    if all_features:
        return canonical_name, features
    return canonical_name, features[:feature_budget]


def audit_feature_set(features: Iterable[str]) -> dict[str, object]:
    """Return JSON-friendly counts and leakage flags for selected features."""
    feature_list = list(features)
    counts = {
        "total": len(feature_list),
        "trump_text": _count(feature_list, _is_trump_text_column),
        "tw_market": _count(feature_list, _is_tw_market_column),
        "global_market": _count(feature_list, _is_global_market_column),
        "tx_night": _count(feature_list, _is_tx_night_column),
        "institutional": _count(feature_list, _is_institutional_column),
        "margin": _count(feature_list, _is_margin_column),
        "market_state": _count(feature_list, _is_market_state_column),
        "trump_regime": _count(feature_list, _is_trump_regime_column),
        "event_gate": _count(feature_list, _is_event_gate_column),
        "target_columns": _count(feature_list, _is_target_column_name),
        "date_columns": _count(feature_list, _is_date_column_name),
    }
    counts["other"] = _count(feature_list, _is_other_column)

    return {
        "feature_count": len(feature_list),
        "counts": counts,
        "contamination_flags": {
            "has_trump_text": counts["trump_text"] > 0,
            "has_trump_regime": counts["trump_regime"] > 0,
            "has_event_gate": counts["event_gate"] > 0,
            "has_global_market": counts["global_market"] > 0,
            "has_tx_night": counts["tx_night"] > 0,
            "has_target_columns": counts["target_columns"] > 0,
            "has_date_columns": counts["date_columns"] > 0,
            "has_other_columns": counts["other"] > 0,
        },
    }


def _legal_numeric_columns(df: pd.DataFrame, target: str) -> list[str]:
    target_safe_name = safe_name(target)
    excluded = {*TARGET_COLUMNS, target, target_safe_name}
    return [
        column
        for column in df.columns
        if column not in excluded
        and not _is_date_column_name(column)
        and pd.api.types.is_numeric_dtype(df[column])
        and df[column].notna().any()
    ]


def _matches_feature_set(column: str, target: str, feature_set: str) -> bool:
    if feature_set == "TW_self_only":
        return _is_self_market_column(column, target)
    if feature_set == "TW_market_only":
        return _is_tw_market_group_column(column)
    if feature_set == "TW_plus_global_market":
        return _is_tw_plus_global_market_column(column)
    if feature_set == "Trump_text_only":
        return _is_trump_text_column(column)
    if feature_set == "TW_plus_Trump":
        return _is_tw_market_group_column(column) or _is_trump_text_column(column)
    if feature_set == "Global_plus_Trump_no_gate":
        return (
            _is_tw_plus_global_market_column(column)
            or _is_trump_text_column(column)
            or _is_trump_regime_column(column)
        )
    if feature_set == "Global_plus_Trump_with_gate":
        return (
            _is_tw_plus_global_market_column(column)
            or _is_trump_text_column(column)
            or _is_trump_regime_column(column)
            or _is_event_gate_column(column)
        )
    return False


def _is_self_market_column(column: str, target: str) -> bool:
    target_symbol = safe_name(target)
    return _is_market_or_volume_column(column, (target_symbol,))


def _is_tw_market_group_column(column: str) -> bool:
    return (
        _is_tw_market_column(column)
        or _is_institutional_column(column)
        or _is_margin_column(column)
    )


def _is_tw_plus_global_market_column(column: str) -> bool:
    return (
        _is_tw_market_group_column(column)
        or _is_global_market_column(column)
        or _is_tx_night_column(column)
        or _is_market_state_column(column)
    )


def _is_market_or_volume_column(column: str, symbols: tuple[str, ...]) -> bool:
    return any(
        column.startswith(f"mkt_{symbol}_") or column.startswith(f"vol_{symbol}_")
        for symbol in symbols
    )


def _is_tw_market_column(column: str) -> bool:
    return _is_market_or_volume_column(column, TW_MARKET_SYMBOLS)


def _is_global_market_column(column: str) -> bool:
    return _is_market_or_volume_column(column, GLOBAL_MARKET_SYMBOLS)


def _is_trump_text_column(column: str) -> bool:
    return column.startswith("trump_")


def _is_institutional_column(column: str) -> bool:
    return column.startswith("inst_")


def _is_margin_column(column: str) -> bool:
    return column.startswith("margin_")


def _is_tx_night_column(column: str) -> bool:
    return column.startswith("tx_night_")


def _is_market_state_column(column: str) -> bool:
    return column in MARKET_STATE_COLUMNS


def _is_trump_regime_column(column: str) -> bool:
    return column in TRUMP_REGIME_COLUMNS


def _is_event_gate_column(column: str) -> bool:
    return column == EVENT_GATE_COLUMN


def _is_target_column_name(column: str) -> bool:
    return column in TARGET_COLUMNS


def _is_date_column_name(column: str) -> bool:
    return column in DATE_COLUMNS


def _is_known_ablation_column(column: str) -> bool:
    return (
        _is_trump_text_column(column)
        or _is_tw_market_column(column)
        or _is_global_market_column(column)
        or _is_tx_night_column(column)
        or _is_institutional_column(column)
        or _is_margin_column(column)
        or _is_market_state_column(column)
        or _is_trump_regime_column(column)
        or _is_event_gate_column(column)
        or _is_target_column_name(column)
        or _is_date_column_name(column)
    )


def _is_other_column(column: str) -> bool:
    return not _is_known_ablation_column(column)


def _count(features: Iterable[str], predicate) -> int:
    return sum(1 for column in features if predicate(column))
