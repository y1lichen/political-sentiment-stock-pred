from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import DEFAULT_TARGET, RANDOM_SEED, Paths
from src.evaluation.metrics import (
    classification_metrics,
    event_signal_metrics,
    regression_metrics,
    signal_metrics,
)
from src.features.feature_sets import CANONICAL_FEATURE_SETS, audit_feature_set
from src.utils.io import ensure_dir, safe_name


SPLITS = ("train", "val", "test")
BASELINE_THRESHOLD = 0.55
GATED_FEATURE_SET = "Global_plus_Trump_with_gate"
COMPARISONS = (
    ("Trump_text_only", "TW_self_only", "Trump raw signal"),
    ("TW_plus_Trump", "TW_market_only", "Trump over Taiwan local"),
    ("TW_plus_global_market", "TW_market_only", "Global market transmission"),
    ("Global_plus_Trump_no_gate", "TW_plus_global_market", "Trump beyond global market"),
    ("Global_plus_Trump_with_gate", "Global_plus_Trump_no_gate", "Event gate value"),
)
PLACEBO_OFFSETS = (-5, -3, -1, 0, 1, 3, 5)
MATCH_COLS = (
    "mkt_2330_TW_ret_lag1",
    "mkt_2330_TW_ret_5d_lag1",
    "mkt_2330_TW_volatility_20d_lag1",
    "mkt_0050_TW_ret_lag1",
    "mkt_idx_VIX_ret_lag1",
    "mkt_idx_SOX_ret_lag1",
    "mkt_idx_NDX_ret_lag1",
    "tx_night_close_lag1",
    "market_stress_score",
)


def artifact_stem(model: str, target: str, split: str, feature_set: str) -> str:
    return f"{model}_{safe_name(target)}_{split}_{feature_set}"


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    values = np.arange(start, stop + step / 2, step)
    return [round(float(v), 2) for v in values]


def load_prediction(paths: Paths, model: str, target: str, split: str, feature_set: str) -> pd.DataFrame:
    path = paths.predictions_dir / f"predictions_{artifact_stem(model, target, split, feature_set)}.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["feature_set"] = feature_set
    frame["event_gate_default"] = frame["event_gate_default"].fillna(0).astype(int)
    return frame


def add_diagnostic_gate(frame: pd.DataFrame, feature_set: str, gate_col: str = "event_gate_default") -> pd.DataFrame:
    out = frame.copy()
    out["_diagnostic_gate"] = out[gate_col] if feature_set == GATED_FEATURE_SET else 1
    return out


def evaluate_frame(
    frame: pd.DataFrame,
    threshold: float,
    transaction_cost: float,
    slippage: float,
) -> dict[str, float | int]:
    y_true = frame["target_direction_1d"].astype(int).to_numpy()
    proba = frame["pred_direction_proba"].to_numpy()
    ret_pred = frame["pred_return"].to_numpy()
    return {
        "rows": int(len(frame)),
        "event_days": int(frame["event_gate_default"].fillna(0).astype(int).eq(1).sum()),
        **classification_metrics(y_true, proba),
        **regression_metrics(frame["target_return_1d"].to_numpy(), ret_pred),
        **signal_metrics(
            frame,
            gate_col="_diagnostic_gate",
            threshold=threshold,
            transaction_cost=transaction_cost,
            slippage=slippage,
        ),
        **event_signal_metrics(frame, gate_col="_diagnostic_gate", threshold=threshold),
    }


def build_sweep(
    predictions: dict[str, pd.DataFrame],
    thresholds: list[float],
    transaction_cost: float,
    slippage: float,
) -> pd.DataFrame:
    rows = []
    for feature_set, frame in predictions.items():
        gated = add_diagnostic_gate(frame, feature_set)
        for split_name in SPLITS:
            split_frame = gated[gated["split"].eq(split_name)].copy()
            for threshold in thresholds:
                rows.append(
                    {
                        "feature_set": feature_set,
                        "split": split_name,
                        "threshold": threshold,
                        "uses_event_gate": feature_set == GATED_FEATURE_SET,
                        **evaluate_frame(split_frame, threshold, transaction_cost, slippage),
                    }
                )
    return pd.DataFrame(rows)


def select_thresholds(sweep: pd.DataFrame, min_val_signals: int) -> dict[str, float]:
    selected = {}
    for feature_set in CANONICAL_FEATURE_SETS:
        candidates = sweep[
            sweep["feature_set"].eq(feature_set)
            & sweep["split"].eq("val")
            & (sweep["signal_count"] >= min_val_signals)
        ].copy()
        if candidates.empty:
            raise RuntimeError(f"No validation threshold candidate for {feature_set}")
        candidates = candidates.sort_values(
            ["avg_signal_return", "hit_rate", "signal_count", "threshold"],
            ascending=[False, False, False, True],
            na_position="last",
        )
        selected[feature_set] = float(candidates.iloc[0]["threshold"])
    return selected


def selected_summary(sweep: pd.DataFrame, selected: dict[str, float]) -> pd.DataFrame:
    rows = []
    for feature_set, threshold in selected.items():
        for method, selected_threshold in (
            ("baseline_0.55", BASELINE_THRESHOLD),
            ("validation_selected", threshold),
        ):
            matched = sweep[
                sweep["feature_set"].eq(feature_set)
                & sweep["threshold"].eq(round(selected_threshold, 2))
            ].copy()
            matched["selection_method"] = method
            rows.append(matched)
    return pd.concat(rows, ignore_index=True)


def test_row(summary: pd.DataFrame, feature_set: str, method: str = "validation_selected") -> pd.Series:
    rows = summary[
        summary["feature_set"].eq(feature_set)
        & summary["selection_method"].eq(method)
        & summary["split"].eq("test")
    ]
    return rows.iloc[0]


def pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def decimal(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.4f}"


def format_cell(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(format_cell(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def metric_table(rows: list[dict[str, object]]) -> str:
    frame = pd.DataFrame(rows)
    return markdown_table(frame)


def read_feature_audits(paths: Paths, model: str, target: str, split: str) -> pd.DataFrame:
    rows = []
    for feature_set in CANONICAL_FEATURE_SETS:
        path = paths.features_dir / f"selected_features_{artifact_stem(model, target, split, feature_set)}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = payload["features"]
        audit = audit_feature_set(features)
        counts = audit["counts"]
        flags = audit["contamination_flags"]
        rows.append(
            {
                "feature_set": feature_set,
                "feature_count": len(features),
                "trump_text": counts["trump_text"],
                "tw_market": counts["tw_market"],
                "global_market": counts["global_market"],
                "tx_night": counts["tx_night"],
                "institutional": counts["institutional"],
                "margin": counts["margin"],
                "market_state": counts["market_state"],
                "trump_regime": counts["trump_regime"],
                "event_gate": counts["event_gate"],
                "other": counts["other"],
                "has_trump_text": flags["has_trump_text"],
                "has_global_market": flags["has_global_market"],
                "has_tx_night": flags["has_tx_night"],
                "has_event_gate": flags["has_event_gate"],
                "has_other_columns": flags["has_other_columns"],
            }
        )
    return pd.DataFrame(rows)


def build_incremental_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left, right, label in COMPARISONS:
        lrow = test_row(summary, left)
        rrow = test_row(summary, right)
        rows.append(
            {
                "comparison": label,
                "left": left,
                "right": right,
                "delta_auc": float(lrow["auc"] - rrow["auc"]),
                "delta_hit_rate": float(lrow["hit_rate"] - rrow["hit_rate"]),
                "delta_avg_signal_return": float(lrow["avg_signal_return"] - rrow["avg_signal_return"]),
                "delta_cumulative_return": float(lrow["cumulative_return"] - rrow["cumulative_return"]),
                "delta_signal_count": int(lrow["signal_count"] - rrow["signal_count"]),
            }
        )
    return pd.DataFrame(rows)


def main_test_rows(summary: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for feature_set in CANONICAL_FEATURE_SETS:
        row = test_row(summary, feature_set)
        rows.append(
            {
                "feature_set": feature_set,
                "threshold": f"{row['threshold']:.2f}",
                "auc": decimal(row["auc"]),
                "accuracy": decimal(row["accuracy"]),
                "balanced_accuracy": decimal(row["balanced_accuracy"]),
                "precision": decimal(row["precision"]),
                "recall": decimal(row["recall"]),
                "f1": decimal(row["f1"]),
                "signals": int(row["signal_count"]),
                "coverage": pct(row["coverage"]),
                "hit_rate": pct(row["hit_rate"]),
                "avg_signal_return": pct(row["avg_signal_return"]),
                "compound_nav": decimal(row["compound_nav"]),
                "cumulative_return": pct(row["cumulative_return"]),
                "max_drawdown": pct(row["max_drawdown"]),
                "event_signals": int(row["event_signal_count"]),
                "non_event_signals": int(row["non_event_signal_count"]),
            }
        )
    return rows


def build_markdown(
    target: str,
    model: str,
    split: str,
    selected: dict[str, float],
    feature_audit: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    transaction_cost: float,
    slippage: float,
) -> str:
    selected_lines = "\n".join(
        f"- `{feature_set}`: `{threshold:.2f}`" for feature_set, threshold in selected.items()
    )
    threshold_rows = []
    for feature_set in CANONICAL_FEATURE_SETS:
        baseline = test_row(summary, feature_set, "baseline_0.55")
        selected_row = test_row(summary, feature_set)
        threshold_rows.append(
            {
                "feature_set": feature_set,
                "baseline_hit": pct(baseline["hit_rate"]),
                "baseline_avg_ret": pct(baseline["avg_signal_return"]),
                "selected_threshold": f"{selected_row['threshold']:.2f}",
                "selected_hit": pct(selected_row["hit_rate"]),
                "selected_avg_ret": pct(selected_row["avg_signal_return"]),
            }
        )
    comparison_rows = [
        {
            "comparison": row["comparison"],
            "left_minus_right_auc": decimal(row["delta_auc"]),
            "left_minus_right_hit": pct(row["delta_hit_rate"]),
            "left_minus_right_avg_ret": pct(row["delta_avg_signal_return"]),
            "left_minus_right_cum_ret": pct(row["delta_cumulative_return"]),
            "signal_delta": int(row["delta_signal_count"]),
        }
        for _, row in comparison.iterrows()
    ]
    return f"""# Ablation Diagnostics

Experiment: `{target} / {model} / {split}`

## Selected Thresholds

Thresholds are selected on validation only. Candidate threshold requires validation `signal_count >= 50`, then maximizes validation `avg_signal_return`.

{selected_lines}

## Feature Audit

{markdown_table(feature_audit)}

## Test Summary

{metric_table(main_test_rows(summary))}

## Threshold 0.55 vs Validation-Selected

{metric_table(threshold_rows)}

## Incremental Comparisons

{metric_table(comparison_rows)}

## Interpretation Guide

- `Trump_text_only` vs `TW_self_only`: tests whether Trump text has raw signal.
- `TW_plus_Trump` vs `TW_market_only`: tests whether Trump features add signal beyond Taiwan local information.
- `TW_plus_global_market` vs `TW_market_only`: tests whether global and overnight market reaction explains performance.
- `Global_plus_Trump_no_gate` vs `TW_plus_global_market`: tests whether Trump text adds information beyond global market reaction.
- `Global_plus_Trump_with_gate` vs `Global_plus_Trump_no_gate`: tests whether `event_gate_default` helps or over-restricts the trading universe.

Transaction cost setting: `{transaction_cost}` per position change. Slippage setting: `{slippage}` per position change. If both are zero, cost/slippage-adjusted returns equal the raw strategy return. `cumulative_return` is arithmetic sum; `compound_nav` is compounded NAV.
"""


def shuffled_gate(frame: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    shuffled_parts = []
    for split_name in SPLITS:
        split_frame = frame[frame["split"].eq(split_name)]
        values = split_frame["event_gate_default"].to_numpy().copy()
        rng.shuffle(values)
        shuffled_parts.append(pd.Series(values, index=split_frame.index))
    return pd.concat(shuffled_parts).sort_index()


def shifted_gate(frame: pd.DataFrame, offset: int) -> pd.Series:
    shifted_parts = []
    for split_name in SPLITS:
        split_frame = frame[frame["split"].eq(split_name)]
        shifted = split_frame["event_gate_default"].shift(offset, fill_value=0).astype(int)
        shifted_parts.append(shifted)
    return pd.concat(shifted_parts).sort_index()


def placebo_metric(
    frame: pd.DataFrame,
    threshold: float,
    gate: pd.Series,
    label: str,
) -> dict[str, object]:
    placebo = frame.copy()
    placebo["_diagnostic_gate"] = gate.to_numpy()
    metrics = signal_metrics(placebo, gate_col="_diagnostic_gate", threshold=threshold)
    events = event_signal_metrics(placebo, gate_col="_diagnostic_gate", threshold=threshold)
    return {
        "placebo": label,
        "threshold": threshold,
        "signal_count": metrics["signal_count"],
        "coverage": metrics["coverage"],
        "hit_rate": metrics["hit_rate"],
        "avg_signal_return": metrics["avg_signal_return"],
        "cumulative_return": metrics["cumulative_return"],
        "event_signal_count": events["event_signal_count"],
        "non_event_signal_count": events["non_event_signal_count"],
    }


def random_event_placebo(frame: pd.DataFrame, threshold: float, permutations: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = [placebo_metric(frame, threshold, frame["event_gate_default"], "real_event_days")]
    for i in range(permutations):
        rows.append(placebo_metric(frame, threshold, shuffled_gate(frame, rng), f"random_{i + 1:03d}"))
    return pd.DataFrame(rows)


def shifted_event_placebo(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = [
        placebo_metric(frame, threshold, shifted_gate(frame, offset), f"shift_{offset:+d}")
        for offset in PLACEBO_OFFSETS
    ]
    return pd.DataFrame(rows)


def matched_non_event_test(
    predictions: pd.DataFrame,
    model_table: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    available = [c for c in MATCH_COLS if c in model_table.columns]
    frame = predictions[predictions["split"].eq("test")].merge(
        model_table[["date", *available]],
        on="date",
        how="left",
    )
    frame = frame.dropna(subset=available).copy()
    events = frame[frame["event_gate_default"].eq(1)].copy()
    non_events = frame[frame["event_gate_default"].eq(0)].copy()
    means = frame[available].mean()
    stds = frame[available].std(ddof=0).replace(0, 1)
    non_event_matrix = ((non_events[available] - means) / stds).to_numpy()
    matched_indices = []
    for _, event_row in events.iterrows():
        event_vector = ((event_row[available] - means) / stds).to_numpy(dtype=float)
        distances = ((non_event_matrix - event_vector) ** 2).sum(axis=1)
        matched_indices.append(non_events.index[int(distances.argmin())])
    matched = non_events.loc[matched_indices].copy()
    events_eval = events.copy()
    matched_eval = matched.copy()
    events_eval["_diagnostic_gate"] = 1
    matched_eval["_diagnostic_gate"] = 1
    event_metrics = signal_metrics(events_eval, gate_col="_diagnostic_gate", threshold=threshold)
    matched_metrics = signal_metrics(matched_eval, gate_col="_diagnostic_gate", threshold=threshold)
    return pd.DataFrame(
        [
            {
                "sample": "real_event_days",
                "rows": int(len(events_eval)),
                "signal_count": event_metrics["signal_count"],
                "hit_rate": event_metrics["hit_rate"],
                "avg_signal_return": event_metrics["avg_signal_return"],
                "cumulative_return": event_metrics["cumulative_return"],
            },
            {
                "sample": "matched_non_event_days",
                "rows": int(len(matched_eval)),
                "signal_count": matched_metrics["signal_count"],
                "hit_rate": matched_metrics["hit_rate"],
                "avg_signal_return": matched_metrics["avg_signal_return"],
                "cumulative_return": matched_metrics["cumulative_return"],
            },
        ]
    )


def build_placebo_markdown(
    target: str,
    model: str,
    split: str,
    random_placebo: pd.DataFrame,
    shifted_placebo: pd.DataFrame,
    matched: pd.DataFrame,
) -> str:
    random_summary = random_placebo.copy()
    random_summary["kind"] = np.where(random_summary["placebo"].eq("real_event_days"), "real", "random")
    random_agg = random_summary.groupby("kind")[
        ["signal_count", "hit_rate", "avg_signal_return", "cumulative_return"]
    ].mean().reset_index()
    return f"""# Placebo Tests

Experiment: `{target} / {model} / {split}`

## Random Event-Date Placebo

The random placebo shuffles `event_gate_default` inside each split, preserving each split's event-day count.

{markdown_table(random_agg)}

## Shifted Event-Date Placebo

Offsets are trading-row shifts, not calendar-day shifts.

{markdown_table(shifted_placebo)}

## Matched Non-Event Day Test

Matching uses available lagged market-state columns from the modeling table.

{markdown_table(matched)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seven-way ablation diagnostics.")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--model", default="event_gated_mlp")
    parser.add_argument("--split", default="regime_aware")
    parser.add_argument("--threshold-start", type=float, default=0.50)
    parser.add_argument("--threshold-stop", type=float, default=0.75)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--min-val-signals", type=int, default=50)
    parser.add_argument("--transaction-cost", type=float, default=0.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--random-placebo-permutations", type=int, default=100)
    args = parser.parse_args()

    paths = Paths()
    predictions = {
        feature_set: load_prediction(paths, args.model, args.target, args.split, feature_set)
        for feature_set in CANONICAL_FEATURE_SETS
    }
    thresholds = threshold_grid(args.threshold_start, args.threshold_stop, args.threshold_step)
    sweep = build_sweep(predictions, thresholds, args.transaction_cost, args.slippage)
    selected = select_thresholds(sweep, args.min_val_signals)
    summary = selected_summary(sweep, selected)
    comparison = build_incremental_comparison(summary)
    feature_audit = read_feature_audits(paths, args.model, args.target, args.split)

    ensure_dir(paths.reports_dir)
    report_stem = f"{args.model}_{safe_name(args.target)}_{args.split}"
    sweep_path = paths.reports_dir / f"threshold_sweep_ablation_{report_stem}.csv"
    summary_path = paths.reports_dir / f"ablation_summary_{report_stem}.csv"
    comparison_path = paths.reports_dir / f"ablation_comparisons_{report_stem}.csv"
    diagnostics_path = paths.reports_dir / f"diagnostics_ablation_{report_stem}.md"
    placebo_path = paths.reports_dir / f"placebo_tests_{report_stem}.md"
    random_placebo_path = paths.reports_dir / f"placebo_random_event_{report_stem}.csv"
    shifted_placebo_path = paths.reports_dir / f"placebo_shifted_event_{report_stem}.csv"
    matched_path = paths.reports_dir / f"placebo_matched_non_event_{report_stem}.csv"

    sweep.to_csv(sweep_path, index=False)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    diagnostics_path.write_text(
        build_markdown(
            args.target,
            args.model,
            args.split,
            selected,
            feature_audit,
            summary,
            comparison,
            args.transaction_cost,
            args.slippage,
        ),
        encoding="utf-8",
    )

    gated_frame = predictions[GATED_FEATURE_SET]
    gated_threshold = selected[GATED_FEATURE_SET]
    random_placebo = random_event_placebo(gated_frame, gated_threshold, args.random_placebo_permutations)
    shifted_placebo = shifted_event_placebo(gated_frame, gated_threshold)
    model_table = pd.read_csv(paths.datasets_dir / f"modeling_table_{safe_name(args.target)}.csv", parse_dates=["date"])
    no_gate_frame = predictions["Global_plus_Trump_no_gate"]
    matched = matched_non_event_test(no_gate_frame, model_table, selected["Global_plus_Trump_no_gate"])

    random_placebo.to_csv(random_placebo_path, index=False)
    shifted_placebo.to_csv(shifted_placebo_path, index=False)
    matched.to_csv(matched_path, index=False)
    placebo_path.write_text(
        build_placebo_markdown(args.target, args.model, args.split, random_placebo, shifted_placebo, matched),
        encoding="utf-8",
    )

    print(f"Wrote threshold sweep: {sweep_path}")
    print(f"Wrote ablation summary: {summary_path}")
    print(f"Wrote ablation comparisons: {comparison_path}")
    print(f"Wrote diagnostics report: {diagnostics_path}")
    print(f"Wrote placebo report: {placebo_path}")


if __name__ == "__main__":
    main()
