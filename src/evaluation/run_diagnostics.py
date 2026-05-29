from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_TARGET, Paths
from src.evaluation.metrics import signal_metrics
from src.utils.io import ensure_dir, safe_name, write_json


SPLITS = ("train", "val", "test")
BASELINE_THRESHOLD = 0.55


def artifact_stem(model: str, target: str, split: str, feature_set: str = "full") -> str:
    suffix = "" if feature_set == "full" else f"_{feature_set}"
    return f"{model}_{safe_name(target)}_{split}{suffix}"


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    values = np.arange(start, stop + step / 2, step)
    return [round(float(v), 2) for v in values]


def with_diagnostic_gate(frame: pd.DataFrame, event_days_only: bool, use_event_gate: bool) -> pd.DataFrame:
    out = frame.copy()
    out["event_gate_default"] = out["event_gate_default"].fillna(0).astype(int)
    if event_days_only:
        out = out[out["event_gate_default"].eq(1)].copy()
    out["_diagnostic_gate"] = out["event_gate_default"] if use_event_gate else 1
    return out


def signal_counts(frame: pd.DataFrame, threshold: float) -> dict[str, int]:
    gate = frame["_diagnostic_gate"].fillna(0).astype(int).eq(1)
    long_signal = gate & (frame["pred_direction_proba"] >= threshold)
    short_signal = gate & (frame["pred_direction_proba"] <= 1 - threshold)
    signal_mask = long_signal | short_signal
    non_event_signal = signal_mask & frame["event_gate_default"].fillna(0).astype(int).eq(0)
    return {
        "long_count": int(long_signal.sum()),
        "short_count": int(short_signal.sum()),
        "non_event_signal_count": int(non_event_signal.sum()),
    }


def evaluate_frame(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    metrics = signal_metrics(frame, gate_col="_diagnostic_gate", threshold=threshold)
    counts = signal_counts(frame, threshold)
    return {
        "rows": int(len(frame)),
        "event_days": int(frame["event_gate_default"].fillna(0).astype(int).eq(1).sum()),
        **counts,
        **metrics,
    }


def build_sweep(
    scenarios: dict[str, pd.DataFrame],
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    for scenario, frame in scenarios.items():
        for split_name in SPLITS:
            split_frame = frame[frame["split"].eq(split_name)].copy()
            for threshold in thresholds:
                rows.append(
                    {
                        "scenario": scenario,
                        "split": split_name,
                        "threshold": threshold,
                        **evaluate_frame(split_frame, threshold),
                    }
                )
    return pd.DataFrame(rows)


def select_thresholds(sweep: pd.DataFrame, min_val_signals: int) -> dict[str, float]:
    selected = {}
    for scenario in sorted(sweep["scenario"].unique()):
        candidates = sweep[
            sweep["scenario"].eq(scenario)
            & sweep["split"].eq("val")
            & (sweep["signal_count"] >= min_val_signals)
        ].copy()
        if candidates.empty:
            raise RuntimeError(f"No validation threshold candidate for {scenario}")
        candidates = candidates.sort_values(
            ["avg_signal_return", "hit_rate", "signal_count", "threshold"],
            ascending=[False, False, False, True],
            na_position="last",
        )
        selected[scenario] = float(candidates.iloc[0]["threshold"])
    return selected


def scenario_rows(sweep: pd.DataFrame, selected: dict[str, float]) -> pd.DataFrame:
    rows = []
    for scenario, threshold in selected.items():
        for selection_method, selected_threshold in [
            ("baseline_0.55", BASELINE_THRESHOLD),
            ("validation_selected", threshold),
        ]:
            matched = sweep[
                sweep["scenario"].eq(scenario)
                & sweep["threshold"].eq(round(selected_threshold, 2))
            ].copy()
            matched["selection_method"] = selection_method
            rows.append(matched)
    return pd.concat(rows, ignore_index=True)


def test_row(summary: pd.DataFrame, scenario: str, selection_method: str) -> pd.Series:
    rows = summary[
        summary["scenario"].eq(scenario)
        & summary["split"].eq("test")
        & summary["selection_method"].eq(selection_method)
    ]
    return rows.iloc[0]


def pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def number(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.4f}"


def markdown_metric_row(label: str, row: pd.Series) -> str:
    return (
        f"| {label} | {row['threshold']:.2f} | {int(row['signal_count'])} | "
        f"{pct(row['coverage'])} | {pct(row['hit_rate'])} | "
        f"{pct(row['avg_signal_return'])} | {pct(row['cumulative_return'])} | "
        f"{pct(row['max_drawdown'])} | {int(row['non_event_signal_count'])} |"
    )


def build_markdown(
    target: str,
    model: str,
    split: str,
    selected: dict[str, float],
    summary: pd.DataFrame,
    event_payload: dict,
) -> str:
    full_base = test_row(summary, "full", "baseline_0.55")
    full_selected = test_row(summary, "full", "validation_selected")
    full_event = test_row(summary, "full_event_days", "validation_selected")
    market_event = test_row(summary, "market_only_on_event_days", "validation_selected")
    pure_market = test_row(summary, "pure_market", "validation_selected")

    rows = [
        markdown_metric_row("Full baseline threshold", full_base),
        markdown_metric_row("Full validation-selected threshold", full_selected),
        markdown_metric_row("Full event days only", full_event),
        markdown_metric_row("Market-only on event days", market_event),
        markdown_metric_row("Pure market, no event gate", pure_market),
    ]
    selected_lines = "\n".join(
        f"- `{scenario}`: `{threshold:.2f}`" for scenario, threshold in sorted(selected.items())
    )
    row_counts = event_payload["event_day_row_counts"]
    return f"""# Threshold Sweep and Ablation Diagnostics

Experiment: `{target} / {model} / {split}`

## Selected Thresholds

Thresholds are selected on validation only. The rule is highest validation average signal return with at least `{event_payload["min_val_signals"]}` validation signals.

{selected_lines}

## Test Summary

| Scenario | Threshold | Signals | Coverage | Hit rate | Avg signal return | Cumulative return | Max drawdown | Non-event signals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Event-Day-Only Row Counts

| Split | Event-day rows |
|---|---:|
| Train | {row_counts["train"]} |
| Validation | {row_counts["val"]} |
| Test | {row_counts["test"]} |

## Interpretation

- `full` keeps the current production-style gate: only rows with `event_gate_default == 1` may trade.
- `full_event_days` changes the denominator to event days only, so coverage is measured among event days.
- `market_only_on_event_days` removes Trump inputs from the model but evaluates on the same event-day subset, testing whether market features alone explain the event-day performance.
- `pure_market` removes the event gate during evaluation, so non-event days can trade. Its `non_event_signal_count` should be greater than zero when the model emits signals outside event days.

These returns do not include transaction costs or slippage. `cumulative_return` is the arithmetic sum of daily strategy returns, not a compounded NAV.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run threshold sweep and ablation diagnostics.")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--model", default="event_gated_mlp")
    parser.add_argument("--split", default="regime_aware")
    parser.add_argument("--threshold-start", type=float, default=0.50)
    parser.add_argument("--threshold-stop", type=float, default=0.75)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--min-val-signals", type=int, default=50)
    args = parser.parse_args()

    paths = Paths()
    base_stem = artifact_stem(args.model, args.target, args.split)
    market_stem = artifact_stem(args.model, args.target, args.split, "market_only")
    full = pd.read_csv(paths.predictions_dir / f"predictions_{base_stem}.csv", parse_dates=["date"])
    market_only = pd.read_csv(paths.predictions_dir / f"predictions_{market_stem}.csv", parse_dates=["date"])

    scenarios = {
        "full": with_diagnostic_gate(full, event_days_only=False, use_event_gate=True),
        "full_event_days": with_diagnostic_gate(full, event_days_only=True, use_event_gate=True),
        "market_only_on_event_days": with_diagnostic_gate(market_only, event_days_only=True, use_event_gate=True),
        "pure_market": with_diagnostic_gate(market_only, event_days_only=False, use_event_gate=False),
    }

    thresholds = threshold_grid(args.threshold_start, args.threshold_stop, args.threshold_step)
    sweep = build_sweep(scenarios, thresholds)
    selected = select_thresholds(sweep, args.min_val_signals)
    summary = scenario_rows(sweep, selected)

    ensure_dir(paths.reports_dir)
    sweep_path = paths.reports_dir / f"threshold_sweep_{base_stem}.csv"
    event_path = paths.reports_dir / f"event_day_only_{base_stem}.json"
    ablation_path = paths.reports_dir / f"ablation_summary_{base_stem}.csv"
    markdown_path = paths.reports_dir / f"diagnostics_{base_stem}.md"

    sweep.to_csv(sweep_path, index=False)
    summary.to_csv(ablation_path, index=False)

    event_payload = {
        "target": args.target,
        "model": args.model,
        "split": args.split,
        "min_val_signals": args.min_val_signals,
        "selected_thresholds": selected,
        "event_day_row_counts": {
            split_name: int(scenarios["full_event_days"]["split"].eq(split_name).sum())
            for split_name in SPLITS
        },
        "baseline_threshold_event_day_metrics": summary[
            summary["scenario"].isin(["full_event_days", "market_only_on_event_days"])
            & summary["selection_method"].eq("baseline_0.55")
        ].to_dict(orient="records"),
        "selected_threshold_event_day_metrics": summary[
            summary["scenario"].isin(["full_event_days", "market_only_on_event_days"])
            & summary["selection_method"].eq("validation_selected")
        ].to_dict(orient="records"),
    }
    write_json(event_payload, event_path)
    markdown_path.write_text(
        build_markdown(args.target, args.model, args.split, selected, summary, event_payload),
        encoding="utf-8",
    )

    print(f"Wrote threshold sweep: {sweep_path}")
    print(f"Wrote event-day metrics: {event_path}")
    print(f"Wrote ablation summary: {ablation_path}")
    print(f"Wrote diagnostics report: {markdown_path}")


if __name__ == "__main__":
    main()
