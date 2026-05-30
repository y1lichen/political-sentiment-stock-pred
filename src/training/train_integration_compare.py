from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TARGETS
from src.evaluation.metrics import classification_metrics, signal_metrics
from src.training.train import train

LEGACY_TARGETS = ("0050.TW", "00632R.TW", "2303.TW", "2308.TW", "2317.TW", "2330.TW", "2376.TW", "2377.TW", "2382.TW", "2454.TW", "3711.TW")


def _build_args(
    target: str,
    model: str,
    split: str,
    feature_set: str,
    feature_budget: int,
    all_features: bool,
    rebuild_dataset: bool,
    signal_threshold: float,
):
    return argparse.Namespace(
        target=target,
        model=model,
        split=split,
        dataset=None,
        rebuild_dataset=rebuild_dataset,
        feature_budget=feature_budget,
        all_features=all_features,
        hidden_dim=32,
        epochs=80,
        lr=1e-3,
        weight_decay=1e-3,
        signal_threshold=signal_threshold,
        feature_set=feature_set,
    )


def _artifact_paths(project_root: Path, model_name: str, target: str, split: str, feature_set: str):
    stem_target = target.replace(".", "_")
    metrics_path = project_root / "outputs" / "reports" / (
        f"metrics_{model_name}_{stem_target}_{split}_{feature_set}.json"
    )
    pred_path = project_root / "outputs" / "predictions" / (
        f"predictions_{model_name}_{stem_target}_{split}_{feature_set}.csv"
    )
    return metrics_path, pred_path


def _train_and_load(project_root: Path, cfg: dict, target: str, args: argparse.Namespace):
    train_args = _build_args(
        target=target,
        model=cfg["model"],
        split=args.split,
        feature_set=cfg["feature_set"],
        feature_budget=int(cfg["feature_budget"]),
        all_features=bool(cfg["all_features"]),
        rebuild_dataset=bool(args.rebuild_dataset),
        signal_threshold=float(cfg["signal_threshold"]),
    )
    train(train_args)

    metrics_path, pred_path = _artifact_paths(project_root, cfg["model"], target, args.split, cfg["feature_set"])
    if not metrics_path.exists() or not pred_path.exists():
        raise FileNotFoundError(f"Missing artifacts for {target}/{cfg['tag']}: {metrics_path} / {pred_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    pred = pd.read_csv(pred_path)
    return metrics, pred


def _to_row(target: str, cfg: dict, metrics: dict, selected_from: str = ""):
    val_metrics = metrics.get("val", {})
    test_metrics = metrics.get("test", {})
    return {
        "target": target,
        "config": cfg["tag"],
        "model": cfg["model"],
        "feature_set": cfg["feature_set"],
        "feature_budget": int(cfg["feature_budget"]),
        "all_features": bool(cfg["all_features"]),
        "signal_threshold": float(cfg["signal_threshold"]),
        "selected_from": selected_from,
        "val_macro_f1": val_metrics.get("f1_macro", val_metrics.get("f1")),
        "macro_f1": test_metrics.get("f1_macro", test_metrics.get("f1")),
        "precision": test_metrics.get("precision"),
        "recall": test_metrics.get("recall"),
        "accuracy": test_metrics.get("accuracy"),
        "auc": test_metrics.get("auc"),
        "signal_coverage": test_metrics.get("signal_coverage"),
        "signal_hit_rate": test_metrics.get("signal_hit_rate"),
        "signal_cumulative_return": test_metrics.get("signal_cumulative_return"),
        "signal_sharpe": test_metrics.get("signal_sharpe"),
        "signal_mdd": test_metrics.get("signal_compound_max_drawdown"),
        "signal_trades": test_metrics.get("signal_signal_count"),
    }


def _row_from_predframe(target: str, config: str, model: str, feature_set: str, threshold: float, frame: pd.DataFrame, selected_from: str):
    test = frame[frame["split"] == "test"].copy()
    y_true = test["target_direction_1d"].astype(int).to_numpy()
    proba = test["pred_direction_proba"].astype(float).to_numpy()

    cls = classification_metrics(y_true, proba)
    sig = signal_metrics(test, threshold=threshold)

    return {
        "target": target,
        "config": config,
        "model": model,
        "feature_set": feature_set,
        "feature_budget": np.nan,
        "all_features": np.nan,
        "signal_threshold": threshold,
        "selected_from": selected_from,
        "val_macro_f1": np.nan,
        "macro_f1": cls.get("f1"),
        "precision": cls.get("precision"),
        "recall": cls.get("recall"),
        "accuracy": cls.get("accuracy"),
        "auc": cls.get("auc"),
        "signal_coverage": sig.get("coverage"),
        "signal_hit_rate": sig.get("hit_rate"),
        "signal_cumulative_return": sig.get("cumulative_return"),
        "signal_sharpe": sig.get("sharpe"),
        "signal_mdd": sig.get("compound_max_drawdown"),
        "signal_trades": sig.get("signal_count"),
    }


def _build_moe_blend(
    market_pred: pd.DataFrame,
    step3_pred: pd.DataFrame,
    alpha_event: float,
    beta_non_event: float,
) -> pd.DataFrame:
    join_cols = ["date", "split", "target_direction_1d", "target_return_1d", "event_gate_default"]
    base = market_pred[join_cols + ["pred_direction_proba"]].rename(columns={"pred_direction_proba": "proba_market"})
    extra = step3_pred[["date", "split", "pred_direction_proba"]].rename(columns={"pred_direction_proba": "proba_step3"})
    fused = base.merge(extra, on=["date", "split"], how="inner")

    is_event = fused["event_gate_default"].fillna(0).astype(int).eq(1)
    blended_event = alpha_event * fused["proba_step3"] + (1.0 - alpha_event) * fused["proba_market"]
    blended_non_event = beta_non_event * fused["proba_step3"] + (1.0 - beta_non_event) * fused["proba_market"]
    fused["pred_direction_proba"] = np.where(is_event, blended_event, blended_non_event)
    return fused


def _safe_float(x, fallback: float = float("-inf")) -> float:
    if x is None:
        return fallback
    try:
        v = float(x)
    except Exception:
        return fallback
    if np.isnan(v):
        return fallback
    return v


def run_compare(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[2]
    legacy_output_dir = project_root / "output"
    legacy_output_dir.mkdir(parents=True, exist_ok=True)

    if args.targets == "default":
        targets = list(TARGETS)
    elif args.targets in {"all", "all_legacy"}:
        targets = list(LEGACY_TARGETS)
    else:
        targets = [x.strip() for x in args.targets.split(",") if x.strip()]

    fixed_configs = [
        {
            "tag": "market_only",
            "feature_set": args.market_feature_set,
            "model": args.model,
            "feature_budget": args.feature_budget,
            "all_features": args.all_features,
            "signal_threshold": args.signal_threshold,
        },
        {
            "tag": "trump_full",
            "feature_set": args.full_feature_set,
            "model": args.model,
            "feature_budget": args.feature_budget,
            "all_features": args.all_features,
            "signal_threshold": args.signal_threshold,
        },
    ]

    step3_candidates = [
        {"tag": "step3_integrated", "model": "event_gated_mlp", "feature_set": "Global_plus_Trump_no_gate", "feature_budget": 80, "all_features": False, "signal_threshold": 0.55},
        {"tag": "step3_integrated", "model": "event_gated_mlp", "feature_set": "Global_plus_Trump_no_gate", "feature_budget": 48, "all_features": False, "signal_threshold": 0.60},
        {"tag": "step3_integrated", "model": "small_mlp", "feature_set": "Global_plus_Trump_no_gate", "feature_budget": 48, "all_features": False, "signal_threshold": 0.60},
        {"tag": "step3_integrated", "model": "logistic", "feature_set": "Global_plus_Trump_no_gate", "feature_budget": 48, "all_features": False, "signal_threshold": 0.60},
        {"tag": "step3_integrated", "model": "elasticnet", "feature_set": "Global_plus_Trump_no_gate", "feature_budget": 48, "all_features": False, "signal_threshold": 0.60},
        {"tag": "step3_integrated", "model": "logistic", "feature_set": "Global_plus_Trump_no_gate", "feature_budget": 80, "all_features": False, "signal_threshold": 0.55},
        {"tag": "step3_integrated", "model": "elasticnet", "feature_set": "Global_plus_Trump_no_gate", "feature_budget": 80, "all_features": False, "signal_threshold": 0.55},
    ]

    alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    beta_grid = [0.0, 0.1, 0.2, 0.3]

    rows = []
    pred_frames = []

    for target in targets:
        fixed_pred = {}
        for cfg in fixed_configs:
            metrics, pred = _train_and_load(project_root, cfg, target, args)
            rows.append(_to_row(target, cfg, metrics))
            pred["target"] = target
            pred["config"] = cfg["tag"]
            pred["model"] = cfg["model"]
            pred["feature_set"] = cfg["feature_set"]
            pred_frames.append(pred)
            fixed_pred[cfg["tag"]] = pred.copy()

        best = None
        best_val_score = float("-inf")
        best_metrics = None
        best_pred = None
        # v7: 2454 走防過擬合與保守候選；其餘沿用 v6 目標
        lam = float(args.val_auc_weight)
        target_candidates = step3_candidates
        if target == "2454.TW":
            target_candidates = [c for c in step3_candidates if c["model"] in {"elasticnet", "logistic", "small_mlp"}]
            if not target_candidates:
                target_candidates = step3_candidates

        for cand in target_candidates:
            metrics, pred = _train_and_load(project_root, cand, target, args)
            train_m = metrics.get("train", {})
            val_m = metrics.get("val", {})

            train_f1 = _safe_float(train_m.get("f1_macro", train_m.get("f1")))
            val_f1 = _safe_float(val_m.get("f1_macro", val_m.get("f1")))
            val_auc = _safe_float(val_m.get("auc"), fallback=0.5)

            overfit_gap = max(0.0, train_f1 - val_f1)
            if target == "2454.TW":
                # 對 2454 加強泛化約束，避免 val 高但 test 崩
                score = val_f1 + (lam + 0.10) * val_auc - 0.50 * overfit_gap
            else:
                score = val_f1 + lam * val_auc

            if score > best_val_score:
                best_val_score = float(score)
                best = cand
                best_metrics = metrics
                best_pred = pred

        if best is None or best_metrics is None or best_pred is None:
            raise RuntimeError(f"No step3 candidate selected for target={target}")

        selected_label = (
            f"model={best['model']}|fs={best['feature_set']}|budget={best['feature_budget']}|"
            f"thr={best['signal_threshold']}|obj=v7_target_adaptive"
        )
        rows.append(_to_row(target, best, best_metrics, selected_from=selected_label))
        best_pred["target"] = target
        best_pred["config"] = "step3_integrated"
        best_pred["model"] = best["model"]
        best_pred["feature_set"] = best["feature_set"]
        pred_frames.append(best_pred)

        market_pred = fixed_pred["market_only"].copy()
        step3_pred = best_pred.copy()

        best_alpha = 0.0
        best_beta = 0.0
        best_mix_score = float("-inf")
        best_fused = None

        for alpha in alpha_grid:
            for beta in beta_grid:
                fused = _build_moe_blend(market_pred, step3_pred, alpha_event=float(alpha), beta_non_event=float(beta))
                val = fused[fused["split"] == "val"]
                if len(val) == 0:
                    continue
                y_true = val["target_direction_1d"].astype(int).to_numpy()
                y_proba = val["pred_direction_proba"].astype(float).to_numpy()
                cls = classification_metrics(y_true, y_proba)
                score = _safe_float(cls.get("f1")) + lam * _safe_float(cls.get("auc"), fallback=0.5)
                if score > best_mix_score:
                    best_mix_score = float(score)
                    best_alpha = float(alpha)
                    best_beta = float(beta)
                    best_fused = fused

        if best_fused is None:
            best_fused = _build_moe_blend(market_pred, step3_pred, alpha_event=0.0, beta_non_event=0.0)

        moe_row = _row_from_predframe(
            target=target,
            config="step3_moe_blend",
            model="mixture_blend(market_only,step3_integrated)",
            feature_set="event_gate_alpha_beta_blend",
            threshold=float(args.signal_threshold),
            frame=best_fused,
            selected_from=f"alpha_event={best_alpha}|beta_non_event={best_beta}|obj=v7_target_adaptive",
        )
        rows.append(moe_row)

        fused_out = best_fused.copy()
        fused_out["target"] = target
        fused_out["config"] = "step3_moe_blend"
        fused_out["model"] = "mixture_blend(market_only,step3_integrated)"
        fused_out["feature_set"] = "event_gate_alpha_beta_blend"
        fused_out["alpha_event"] = best_alpha
        fused_out["beta_non_event"] = best_beta
        pred_frames.append(fused_out)

    metrics_df = pd.DataFrame(rows).sort_values(["target", "config"]).reset_index(drop=True)
    metrics_df.to_csv(legacy_output_dir / args.metrics_output, index=False)
    if pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(legacy_output_dir / args.predictions_output, index=False)

    print(f"Wrote compare metrics: {legacy_output_dir / args.metrics_output}")
    print(f"Wrote compare predictions: {legacy_output_dir / args.predictions_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Integration runner v7: target-adaptive selection + alpha/beta blend.")
    parser.add_argument("--targets", default="default", help="Comma-separated tickers, or 'default', 'all'/ 'all_legacy' (11 tickers).")
    parser.add_argument("--model", default="event_gated_mlp")
    parser.add_argument("--split", default="regime_aware", choices=["regime_matched", "all_history", "regime_aware"])
    parser.add_argument("--market-feature-set", default="TW_plus_global_market")
    parser.add_argument("--full-feature-set", default="Global_plus_Trump_with_gate")
    parser.add_argument("--feature-budget", type=int, default=80)
    parser.add_argument("--all-features", action="store_true")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--signal-threshold", type=float, default=0.55)
    parser.add_argument("--val-auc-weight", type=float, default=0.15)
    parser.add_argument("--metrics-output", default="integration_compare_metrics_step3_v7.csv")
    parser.add_argument("--predictions-output", default="integration_compare_predictions_step3_v7.csv")
    args = parser.parse_args()
    run_compare(args)


if __name__ == "__main__":
    main()
