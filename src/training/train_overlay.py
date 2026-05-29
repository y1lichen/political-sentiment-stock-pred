from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import DEFAULT_TARGET, RANDOM_SEED, Paths
from src.data.build_dataset import build_modeling_table
from src.features.feature_sets import (
    EVENT_GATE_COLUMN,
    TRUMP_REGIME_COLUMNS,
    build_feature_set,
)
from src.training.common import (
    build_sample_weights,
    fit_sklearn_model,
    split_by_mode,
)
from src.utils.io import ensure_dir, safe_name, write_json, write_pickle


MARKET_FEATURE_SET = "TW_plus_global_market"


class ConstantProbabilityModel:
    """Small sklearn-like fallback when a sparse overlay target has one class."""

    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, X):
        p = np.full(len(X), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass(frozen=True)
class StrategyMetrics:
    signal_count: int
    coverage: float
    hit_rate: float
    avg_signal_return: float
    cumulative_return: float
    compound_nav: float
    max_drawdown: float
    sharpe: float
    turnover: float
    return_after_costs: float


def build_overlay_features(df: pd.DataFrame, target: str, all_features: bool) -> list[str]:
    _, trump_features = build_feature_set(
        df,
        target,
        "Trump_text_only",
        all_features=True,
    )
    candidates = list(trump_features)
    candidates.extend(c for c in TRUMP_REGIME_COLUMNS if c in df.columns)
    if EVENT_GATE_COLUMN in df.columns:
        candidates.append(EVENT_GATE_COLUMN)
    candidates = [
        c
        for c in dict.fromkeys(candidates)
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
    ]
    if all_features:
        return candidates
    return candidates[:96]


def fit_overlay_classifier(model_name: str, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray):
    unique = np.unique(y)
    if len(unique) < 2:
        return ConstantProbabilityModel(float(unique[0]) if len(unique) else 0.5)
    return fit_sklearn_model(model_name, X, y.astype(int), sample_weight)


def positions_from_proba(proba: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(proba >= threshold, 1, np.where(proba <= 1.0 - threshold, -1, 0))


def strategy_metrics(
    returns: np.ndarray,
    positions: np.ndarray,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
) -> StrategyMetrics:
    returns = np.nan_to_num(returns.astype(float), nan=0.0)
    positions = positions.astype(float)
    gross = positions * returns
    signal = positions != 0
    position_change = np.abs(np.diff(np.r_[0.0, positions]))
    costs = position_change * (float(transaction_cost) + float(slippage))
    net = gross - costs
    hit = np.where(positions > 0, returns > 0, returns < 0)
    nav = pd.Series(1.0 + net).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    active_ret = net[signal]
    std = active_ret.std(ddof=1) if len(active_ret) > 1 else 0.0
    return StrategyMetrics(
        signal_count=int(signal.sum()),
        coverage=float(signal.mean()) if len(signal) else 0.0,
        hit_rate=float(hit[signal].mean()) if signal.any() else float("nan"),
        avg_signal_return=float(gross[signal].mean()) if signal.any() else float("nan"),
        cumulative_return=float(gross.sum()),
        compound_nav=float(nav.iloc[-1]) if len(nav) else 1.0,
        max_drawdown=float(drawdown.min()) if len(drawdown) else 0.0,
        sharpe=float(active_ret.mean() / std * np.sqrt(252)) if std > 0 else float("nan"),
        turnover=float(position_change.sum()),
        return_after_costs=float(net.sum()),
    )


def metrics_dict(metrics: StrategyMetrics) -> dict[str, float | int]:
    return metrics.__dict__.copy()


def choose_market_threshold(
    returns: np.ndarray,
    proba: np.ndarray,
    thresholds: Iterable[float],
    min_signals: int,
    objective: str,
    transaction_cost: float,
    slippage: float,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in thresholds:
        pos = positions_from_proba(proba, float(threshold))
        m = strategy_metrics(returns, pos, transaction_cost, slippage)
        rows.append({"threshold": float(threshold), **metrics_dict(m)})
    sweep = pd.DataFrame(rows)
    candidates = sweep[sweep["signal_count"] >= min_signals].copy()
    if candidates.empty:
        candidates = sweep.copy()
    best = candidates.sort_values(objective, ascending=False).iloc[0]
    return float(best["threshold"]), sweep


def apply_overlay(
    market_proba: np.ndarray,
    profit_proba: np.ndarray,
    direction_proba: np.ndarray,
    event_gate: np.ndarray,
    market_threshold: float,
    veto_threshold: float,
    boost_threshold: float,
    boost_size: float,
    override_threshold: float,
    event_only: bool,
) -> np.ndarray:
    base = positions_from_proba(market_proba, market_threshold).astype(float)
    out = base.copy()
    allowed = event_gate.astype(bool) if event_only else np.ones(len(base), dtype=bool)

    veto = allowed & (base != 0) & (profit_proba < veto_threshold)
    out[veto] = 0

    boost = allowed & (base != 0) & (profit_proba >= boost_threshold)
    out[boost] = base[boost] * float(boost_size)

    can_override = allowed & (base == 0) & (override_threshold < 1.0)
    out[can_override & (direction_proba >= override_threshold)] = 1
    out[can_override & (direction_proba <= 1.0 - override_threshold)] = -1
    return out


def choose_overlay_thresholds(
    returns: np.ndarray,
    market_proba: np.ndarray,
    profit_proba: np.ndarray,
    direction_proba: np.ndarray,
    event_gate: np.ndarray,
    market_threshold: float,
    veto_grid: Iterable[float],
    boost_grid: Iterable[float],
    override_grid: Iterable[float],
    boost_size: float,
    min_signals: int,
    objective: str,
    transaction_cost: float,
    slippage: float,
    event_only: bool,
) -> tuple[float, float, pd.DataFrame]:
    rows = []
    for veto in veto_grid:
        for boost in boost_grid:
            for override in override_grid:
                pos = apply_overlay(
                    market_proba,
                    profit_proba,
                    direction_proba,
                    event_gate,
                    market_threshold,
                    float(veto),
                    float(boost),
                    boost_size,
                    float(override),
                    event_only,
                )
                m = strategy_metrics(returns, pos, transaction_cost, slippage)
                rows.append(
                    {
                        "veto_threshold": float(veto),
                        "boost_threshold": float(boost),
                        "override_threshold": float(override),
                        **metrics_dict(m),
                    }
                )
    sweep = pd.DataFrame(rows)
    candidates = sweep[sweep["signal_count"] >= min_signals].copy()
    if candidates.empty:
        candidates = sweep.copy()
    best = candidates.sort_values(objective, ascending=False).iloc[0]
    return float(best["veto_threshold"]), float(best["boost_threshold"]), float(best["override_threshold"]), sweep


def train_overlay(args: argparse.Namespace) -> None:
    np.random.seed(RANDOM_SEED)
    paths = Paths()
    dataset_path = args.dataset or (paths.datasets_dir / f"modeling_table_{safe_name(args.target)}.csv")
    if args.rebuild_dataset or not dataset_path.exists():
        df = build_modeling_table(paths.trump_posts, paths.market_dir, args.target)
        ensure_dir(dataset_path.parent)
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path, parse_dates=["date"])

    train_mask, val_mask, test_mask = split_by_mode(df, args.split)
    clean = df.dropna(subset=["target_return_1d", "target_direction_1d"]).copy()
    clean = clean.sort_values("date").reset_index(drop=True)
    train_mask, val_mask, test_mask = split_by_mode(clean, args.split)

    _, market_features = build_feature_set(
        clean,
        args.target,
        MARKET_FEATURE_SET,
        all_features=True,
    )
    overlay_features = build_overlay_features(clean, args.target, all_features=args.all_overlay_features)

    X_market = clean[market_features].replace([np.inf, -np.inf], np.nan)
    X_overlay = clean[overlay_features].replace([np.inf, -np.inf], np.nan)
    y = clean["target_direction_1d"].astype(int).to_numpy()
    returns = clean["target_return_1d"].astype(float).to_numpy()
    event_gate = clean.get(EVENT_GATE_COLUMN, pd.Series(0, index=clean.index)).fillna(0).astype(int).to_numpy()
    weights = build_sample_weights(clean, args.split)

    market_model = fit_sklearn_model(
        args.market_model,
        X_market[train_mask],
        y[train_mask],
        weights[train_mask],
    )
    market_proba = market_model.predict_proba(X_market)[:, 1]

    market_thresholds = np.round(np.arange(args.market_threshold_min, args.market_threshold_max + 1e-9, args.threshold_step), 4)
    market_threshold, market_sweep = choose_market_threshold(
        returns[val_mask],
        market_proba[val_mask],
        market_thresholds,
        args.min_val_signals,
        args.objective,
        args.transaction_cost,
        args.slippage,
    )

    train_base_pos = positions_from_proba(market_proba[train_mask], market_threshold)
    train_event = event_gate[train_mask].astype(bool)
    overlay_train_mask_local = train_base_pos != 0
    if args.overlay_event_only:
        overlay_train_mask_local &= train_event
    if overlay_train_mask_local.sum() < args.min_overlay_train_samples:
        overlay_train_mask_local = train_base_pos != 0
    if overlay_train_mask_local.sum() < args.min_overlay_train_samples:
        overlay_train_mask_local = np.ones(train_base_pos.shape, dtype=bool)

    train_indices = np.flatnonzero(train_mask)
    overlay_indices = train_indices[overlay_train_mask_local]
    base_profit = train_base_pos[overlay_train_mask_local] * returns[overlay_indices] > 0

    profit_model = fit_overlay_classifier(
        args.overlay_model,
        X_overlay.iloc[overlay_indices],
        base_profit.astype(int),
        weights[overlay_indices],
    )

    direction_train_mask = train_mask & (event_gate.astype(bool) if args.overlay_event_only else np.ones(len(clean), dtype=bool))
    if direction_train_mask.sum() < args.min_overlay_train_samples:
        direction_train_mask = train_mask
    direction_model = fit_overlay_classifier(
        args.overlay_model,
        X_overlay[direction_train_mask],
        y[direction_train_mask],
        weights[direction_train_mask],
    )

    profit_proba = profit_model.predict_proba(X_overlay)[:, 1]
    direction_proba = direction_model.predict_proba(X_overlay)[:, 1]

    veto_grid = np.round(np.arange(args.veto_min, args.veto_max + 1e-9, args.threshold_step), 4)
    boost_grid = list(np.round(np.arange(args.boost_min, args.boost_max + 1e-9, args.threshold_step), 4))
    boost_grid.append(1.01)  # no boost option
    override_grid = list(np.round(np.arange(args.override_min, args.override_max + 1e-9, args.threshold_step), 4))
    override_grid.append(1.01)  # no override option
    veto_threshold, boost_threshold, override_threshold, overlay_sweep = choose_overlay_thresholds(
        returns[val_mask],
        market_proba[val_mask],
        profit_proba[val_mask],
        direction_proba[val_mask],
        event_gate[val_mask],
        market_threshold,
        veto_grid,
        boost_grid,
        override_grid,
        args.boost_size,
        args.min_val_signals,
        args.objective,
        args.transaction_cost,
        args.slippage,
        args.overlay_event_only,
    )

    rows = []
    pred_parts = []
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        market_pos = positions_from_proba(market_proba[mask], market_threshold)
        overlay_pos = apply_overlay(
            market_proba[mask],
            profit_proba[mask],
            direction_proba[mask],
            event_gate[mask],
            market_threshold,
            veto_threshold,
            boost_threshold,
            args.boost_size,
            override_threshold,
            args.overlay_event_only,
        )
        for strategy, pos in [("market_only", market_pos), ("market_plus_trump_overlay", overlay_pos)]:
            m = strategy_metrics(returns[mask], pos, args.transaction_cost, args.slippage)
            rows.append({"split": split_name, "strategy": strategy, **metrics_dict(m)})

        part = clean.loc[mask, ["date", "target_return_1d", "target_direction_1d"]].copy()
        part["split"] = split_name
        part["event_gate_default"] = event_gate[mask]
        part["market_proba"] = market_proba[mask]
        part["market_position"] = market_pos
        part["overlay_profit_proba"] = profit_proba[mask]
        part["overlay_direction_proba"] = direction_proba[mask]
        part["overlay_position"] = overlay_pos
        part["market_strategy_return"] = market_pos * returns[mask]
        part["overlay_strategy_return"] = overlay_pos * returns[mask]
        pred_parts.append(part)

    stem = f"overlay_{safe_name(args.target)}_{args.split}_{args.market_model}_{args.overlay_model}"
    model_path = paths.models_dir / f"{stem}.pkl"
    pred_path = paths.predictions_dir / f"predictions_{stem}.csv"
    summary_path = paths.reports_dir / f"summary_{stem}.csv"
    report_path = paths.reports_dir / f"report_{stem}.md"
    market_sweep_path = paths.reports_dir / f"market_threshold_sweep_{stem}.csv"
    overlay_sweep_path = paths.reports_dir / f"overlay_threshold_sweep_{stem}.csv"

    ensure_dir(paths.models_dir)
    ensure_dir(paths.predictions_dir)
    ensure_dir(paths.reports_dir)
    write_pickle(
        {
            "market_model": market_model,
            "profit_model": profit_model,
            "direction_model": direction_model,
            "market_features": market_features,
            "overlay_features": overlay_features,
            "market_threshold": market_threshold,
            "veto_threshold": veto_threshold,
            "boost_threshold": boost_threshold,
            "boost_size": args.boost_size,
            "override_threshold": override_threshold,
            "args": vars(args),
        },
        model_path,
    )
    predictions = pd.concat(pred_parts, ignore_index=True)
    summary = pd.DataFrame(rows)
    predictions.to_csv(pred_path, index=False)
    summary.to_csv(summary_path, index=False)
    market_sweep.to_csv(market_sweep_path, index=False)
    overlay_sweep.to_csv(overlay_sweep_path, index=False)

    test = summary[summary["split"].eq("test")].set_index("strategy")
    deltas = {
        key: float(test.loc["market_plus_trump_overlay", key] - test.loc["market_only", key])
        for key in [
            "signal_count",
            "coverage",
            "hit_rate",
            "avg_signal_return",
            "cumulative_return",
            "compound_nav",
            "max_drawdown",
            "sharpe",
            "return_after_costs",
        ]
    }
    write_json(
        {
            "target": args.target,
            "split": args.split,
            "market_model": args.market_model,
            "overlay_model": args.overlay_model,
            "market_threshold": market_threshold,
            "veto_threshold": veto_threshold,
            "boost_threshold": boost_threshold,
            "boost_size": args.boost_size,
            "override_threshold": override_threshold,
            "market_feature_count": len(market_features),
            "overlay_feature_count": len(overlay_features),
            "overlay_train_samples": int(len(overlay_indices)),
            "test_deltas_overlay_minus_market": deltas,
            "paths": {
                "model": str(model_path),
                "predictions": str(pred_path),
                "summary": str(summary_path),
                "report": str(report_path),
            },
        },
        report_path.with_suffix(".json"),
    )

    report = [
        "# Market Baseline + Trump Overlay Report",
        "",
        f"Target: `{args.target}`",
        f"Split: `{args.split}`",
        f"Market model: `{args.market_model}` using `{MARKET_FEATURE_SET}`",
        f"Overlay model: `{args.overlay_model}` using Trump text/regime features",
        f"Validation-selected market threshold: `{market_threshold:.2f}`",
        f"Validation-selected veto threshold: `{veto_threshold:.2f}`",
        f"Validation-selected boost threshold: `{boost_threshold:.2f}`",
        f"Boost size: `{args.boost_size:.2f}`",
        f"Validation-selected override threshold: `{override_threshold:.2f}`",
        "",
        "## Test Comparison",
        "",
        "```text",
        test.reset_index().to_string(index=False),
        "```",
        "",
        "## Overlay Minus Market Deltas",
        "",
        "```text",
        pd.DataFrame([deltas]).to_string(index=False),
        "```",
        "",
        "Positive `cumulative_return`, `compound_nav`, `avg_signal_return`, or `return_after_costs` deltas mean the Trump overlay improved over the market-only baseline under this validation-selected configuration.",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote model: {model_path}")
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report: {report_path}")
    print(test.reset_index().to_string(index=False))
    print("Overlay minus market test deltas:")
    for key, value in deltas.items():
        print(f"  {key}: {value:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train market baseline plus Trump overlay strategy.")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--split", default="regime_aware", choices=["regime_matched", "all_history", "regime_aware"])
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--market-model", default="lightgbm", choices=["logistic", "elasticnet", "random_forest", "lightgbm"])
    parser.add_argument("--overlay-model", default="elasticnet", choices=["logistic", "elasticnet", "random_forest", "lightgbm"])
    parser.add_argument("--all-overlay-features", action="store_true")
    parser.add_argument("--overlay-event-only", action="store_true", default=True)
    parser.add_argument("--overlay-all-days", action="store_true", help="Allow Trump overlay decisions on all days, not only event days.")
    parser.add_argument("--objective", default="return_after_costs", choices=[
        "avg_signal_return",
        "cumulative_return",
        "compound_nav",
        "sharpe",
        "return_after_costs",
    ])
    parser.add_argument("--market-threshold-min", type=float, default=0.50)
    parser.add_argument("--market-threshold-max", type=float, default=0.90)
    parser.add_argument("--veto-min", type=float, default=0.00)
    parser.add_argument("--veto-max", type=float, default=0.70)
    parser.add_argument("--boost-min", type=float, default=0.60)
    parser.add_argument("--boost-max", type=float, default=0.90)
    parser.add_argument("--boost-size", type=float, default=1.5)
    parser.add_argument("--override-min", type=float, default=0.60)
    parser.add_argument("--override-max", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--min-val-signals", type=int, default=50)
    parser.add_argument("--min-overlay-train-samples", type=int, default=50)
    parser.add_argument("--transaction-cost", type=float, default=0.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    args = parser.parse_args()
    if args.overlay_all_days:
        args.overlay_event_only = False
    train_overlay(args)


if __name__ == "__main__":
    main()
