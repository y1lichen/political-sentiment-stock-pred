from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import TARGETS
from src.evaluation.metrics import classification_metrics, signal_metrics
from src.training.train import train

LEGACY_TARGETS = (
    "0050.TW",
    "00632R.TW",
    "2303.TW",
    "2308.TW",
    "2317.TW",
    "2330.TW",
    "2376.TW",
    "2377.TW",
    "2382.TW",
    "2454.TW",
    "3711.TW",
)


def _build_args(
    target: str,
    model: str,
    split: str,
    feature_set: str,
    feature_budget: int,
    all_features: bool,
    rebuild_dataset: bool,
    signal_threshold: float,
    pos_weight: float,
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
        pos_weight=pos_weight,
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
        pos_weight=float(cfg.get("pos_weight", 1.0)),
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


def _row_from_predframe(
    target: str,
    config: str,
    model: str,
    feature_set: str,
    threshold: float,
    frame: pd.DataFrame,
    selected_from: str,
):
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


def _build_moe_blend(market_pred: pd.DataFrame, step3_pred: pd.DataFrame, alpha_event: float, beta_non_event: float) -> pd.DataFrame:
    join_cols = ["date", "split", "target_direction_1d", "target_return_1d", "event_gate_default"]
    base = market_pred[join_cols + ["pred_direction_proba"]].rename(columns={"pred_direction_proba": "proba_market"})
    extra = step3_pred[["date", "split", "pred_direction_proba"]].rename(columns={"pred_direction_proba": "proba_step3"})
    fused = base.merge(extra, on=["date", "split"], how="inner")

    is_event = fused["event_gate_default"].fillna(0).astype(int).eq(1)
    blended_event = alpha_event * fused["proba_step3"] + (1.0 - alpha_event) * fused["proba_market"]
    blended_non_event = beta_non_event * fused["proba_step3"] + (1.0 - beta_non_event) * fused["proba_market"]
    fused["pred_direction_proba"] = np.where(is_event, blended_event, blended_non_event)
    return fused


def _build_weighted_blend(market_pred: pd.DataFrame, step3_pred: pd.DataFrame, w_step3: float) -> pd.DataFrame:
    join_cols = ["date", "split", "target_direction_1d", "target_return_1d", "event_gate_default"]
    base = market_pred[join_cols + ["pred_direction_proba"]].rename(columns={"pred_direction_proba": "proba_market"})
    extra = step3_pred[["date", "split", "pred_direction_proba"]].rename(columns={"pred_direction_proba": "proba_step3"})
    fused = base.merge(extra, on=["date", "split"], how="inner")
    fused["pred_direction_proba"] = (1.0 - w_step3) * fused["proba_market"] + w_step3 * fused["proba_step3"]
    return fused


def _build_stacking_blend(market_pred: pd.DataFrame, step3_pred: pd.DataFrame) -> pd.DataFrame | None:
    """Round-1: train a simple meta-learner on train split to fuse two probabilities."""
    join_cols = ["date", "split", "target_direction_1d", "target_return_1d", "event_gate_default"]
    base = market_pred[join_cols + ["pred_direction_proba"]].rename(columns={"pred_direction_proba": "proba_market"})
    extra = step3_pred[["date", "split", "pred_direction_proba"]].rename(columns={"pred_direction_proba": "proba_step3"})
    fused = base.merge(extra, on=["date", "split"], how="inner")

    tr = fused[fused["split"] == "train"].copy()
    if tr.empty:
        return None
    y_train = (tr["target_direction_1d"].astype(int).to_numpy() > 0).astype(int)
    if len(np.unique(y_train)) < 2:
        return None

    x_train = tr[["proba_market", "proba_step3"]].astype(float).to_numpy()
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(x_train, y_train)

    x_all = fused[["proba_market", "proba_step3"]].astype(float).to_numpy()
    fused["pred_direction_proba"] = lr.predict_proba(x_all)[:, 1]
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


def _metrics_from_pred_frame(frame: pd.DataFrame, threshold: float) -> dict:
    out: dict[str, dict[str, float]] = {}
    for split in ["train", "val", "test"]:
        part = frame[frame["split"] == split].copy()
        if part.empty:
            continue
        y_true = part["target_direction_1d"].astype(int).to_numpy()
        y_proba = part["pred_direction_proba"].astype(float).to_numpy()
        cls = classification_metrics(y_true, y_proba)
        sig = signal_metrics(part, threshold=threshold)
        out[split] = {
            "accuracy": cls.get("accuracy"),
            "precision": cls.get("precision"),
            "recall": cls.get("recall"),
            "f1": cls.get("f1"),
            "auc": cls.get("auc"),
            "signal_coverage": sig.get("coverage"),
            "signal_hit_rate": sig.get("hit_rate"),
            "signal_cumulative_return": sig.get("cumulative_return"),
            "signal_sharpe": sig.get("sharpe"),
            "signal_compound_max_drawdown": sig.get("compound_max_drawdown"),
            "signal_signal_count": sig.get("signal_count"),
        }
    return out


def _apply_linear_head_event_only(frame: pd.DataFrame) -> pd.DataFrame:
    """v9: linear calibration head is only applied on event days."""
    out = frame.copy()
    tr = out[out["split"] == "train"]
    if tr.empty:
        return out

    x_train = tr[["pred_direction_proba"]].astype(float).to_numpy()
    y_train = tr["target_direction_1d"].astype(int).to_numpy()

    y_bin = (y_train > 0).astype(int)
    if len(np.unique(y_bin)) < 2:
        return out

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(x_train, y_bin)

    event_mask = out["event_gate_default"].fillna(0).astype(int).eq(1).to_numpy()
    if not event_mask.any():
        return out

    x_event = out.loc[event_mask, ["pred_direction_proba"]].astype(float).to_numpy()
    out.loc[event_mask, "pred_direction_proba"] = lr.predict_proba(x_event)[:, 1]
    return out


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
            "pos_weight": 1.0,
        },
        {
            "tag": "trump_full",
            "feature_set": args.full_feature_set,
            "model": args.model,
            "feature_budget": args.feature_budget,
            "all_features": args.all_features,
            "signal_threshold": args.signal_threshold,
            "pos_weight": 1.0,
        },
    ]

    step3_candidates = [
        {"tag": "step3_integrated", "model": "event_gated_mlp", "feature_set": "Global_plus_Trump_no_gate", "feature_budget": 80, "all_features": False, "signal_threshold": 0.55, "pos_weight": 1.0},
        {"tag": "step3_integrated", "model": "event_gated_mlp", "feature_set": "Global_plus_Trump_no_gate", "feature_budget": 48, "all_features": False, "signal_threshold": 0.60, "pos_weight": 1.0},
        {"tag": "step3_integrated", "model": "small_mlp", "feature_set": "Global_plus_Trump_no_gate", "feature_budget": 48, "all_features": False, "signal_threshold": 0.60, "pos_weight": 1.0},
        {"tag": "step3_integrated", "model": "logistic", "feature_set": "Global_plus_Trump_no_gate", "feature_budget": 48, "all_features": False, "signal_threshold": 0.60, "pos_weight": 1.0},
        {"tag": "step3_integrated", "model": "elasticnet", "feature_set": "Global_plus_Trump_no_gate", "feature_budget": 48, "all_features": False, "signal_threshold": 0.60, "pos_weight": 1.0},
        {"tag": "step3_integrated", "model": "logistic", "feature_set": "Global_plus_Trump_no_gate", "feature_budget": 80, "all_features": False, "signal_threshold": 0.55, "pos_weight": 1.0},
        {"tag": "step3_integrated", "model": "elasticnet", "feature_set": "Global_plus_Trump_no_gate", "feature_budget": 80, "all_features": False, "signal_threshold": 0.55, "pos_weight": 1.0},
    ]

    if args.deep_only:
        step3_candidates = [c for c in step3_candidates if c["model"] in {"event_gated_mlp", "small_mlp"}]

    alpha_grid = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]
    beta_grid = [0.0, 0.05, 0.10, 0.20, 0.30]

    rows = []
    pred_frames = []

    for target in targets:
        fixed_pred = {}
        for cfg in fixed_configs:
            metrics, pred = _train_and_load(project_root, cfg, target, args)
            rows.append(_to_row(target, cfg, metrics))
            p = pred.copy()
            p["target"] = target
            p["config"] = cfg["tag"]
            p["model"] = cfg["model"]
            p["feature_set"] = cfg["feature_set"]
            pred_frames.append(p)
            fixed_pred[cfg["tag"]] = p.copy()

        best_cfg = None
        best_metrics = None
        best_pred = None
        best_val_score = float("-inf")
        best_variant = "raw"

        lam = float(args.val_auc_weight)
        target_candidates = step3_candidates
        laggard_pw_targets = {"00632R.TW", "2377.TW", "2454.TW"}
        if target in laggard_pw_targets:
            extra_pw = []
            for c in step3_candidates:
                if c["model"] in {"event_gated_mlp", "small_mlp"}:
                    for pw in [1.2, 1.5, 2.0]:
                        cc = dict(c)
                        cc["pos_weight"] = pw
                        extra_pw.append(cc)
            target_candidates = step3_candidates + extra_pw
        if (not args.deep_only) and target == "2454.TW":
            target_candidates = [c for c in target_candidates if c["model"] in {"elasticnet", "logistic", "small_mlp", "event_gated_mlp"}] or target_candidates

        for cand in target_candidates:
            metrics_raw, pred_raw = _train_and_load(project_root, cand, target, args)
            variants = [("raw", metrics_raw, pred_raw)]

            if args.v8_linear_head and cand["model"] in {"event_gated_mlp", "small_mlp"}:
                pred_cal = _apply_linear_head_event_only(pred_raw)
                metrics_cal = _metrics_from_pred_frame(pred_cal, threshold=float(cand["signal_threshold"]))
                variants.append(("linear_head_event_only", metrics_cal, pred_cal))

            for variant_name, variant_metrics, variant_pred in variants:
                train_m = variant_metrics.get("train", {})
                val_m = variant_metrics.get("val", {})
                train_f1 = _safe_float(train_m.get("f1_macro", train_m.get("f1")))
                val_f1 = _safe_float(val_m.get("f1_macro", val_m.get("f1")))
                val_auc = _safe_float(val_m.get("auc"), fallback=0.5)

                overfit_gap = max(0.0, train_f1 - val_f1)
                if (not args.deep_only) and target == "2454.TW":
                    score = val_f1 + (lam + 0.10) * val_auc - 0.50 * overfit_gap
                else:
                    score = val_f1 + lam * val_auc

                if score > best_val_score:
                    best_val_score = float(score)
                    best_cfg = cand
                    best_metrics = variant_metrics
                    best_pred = variant_pred
                    best_variant = variant_name

        if best_cfg is None or best_metrics is None or best_pred is None:
            raise RuntimeError(f"No step3 candidate selected for target={target}")

        selected_label = (
            f"model={best_cfg['model']}|fs={best_cfg['feature_set']}|budget={best_cfg['feature_budget']}|"
            f"thr={best_cfg['signal_threshold']}|variant={best_variant}|obj=v9_event_linear_moe"
        )
        rows.append(_to_row(target, best_cfg, best_metrics, selected_from=selected_label))

        step3_pred = best_pred.copy()
        step3_pred["target"] = target
        step3_pred["config"] = "step3_integrated"
        step3_pred["model"] = best_cfg["model"] if best_variant == "raw" else f"{best_cfg['model']}+linear_head"
        step3_pred["feature_set"] = best_cfg["feature_set"]
        pred_frames.append(step3_pred)

        market_pred = fixed_pred["market_only"].copy()

        best_alpha = 0.0
        best_beta = 0.0
        best_mix_score = float("-inf")
        best_fused = None
        best_moe_name = "alpha_beta"

        for alpha in alpha_grid:
            for beta in beta_grid:
                fused = _build_moe_blend(market_pred, step3_pred, alpha_event=float(alpha), beta_non_event=float(beta))
                val = fused[fused["split"] == "val"]
                if val.empty:
                    continue
                y_true = val["target_direction_1d"].astype(int).to_numpy()
                y_proba = val["pred_direction_proba"].astype(float).to_numpy()
                cls = classification_metrics(y_true, y_proba)
                sig = signal_metrics(val, threshold=float(args.signal_threshold))
                score = (
                    1.0 * _safe_float(cls.get("f1"))
                    + 0.15 * _safe_float(cls.get("auc"), fallback=0.5)
                )
                if score > best_mix_score:
                    best_mix_score = float(score)
                    best_alpha = float(alpha)
                    best_beta = float(beta)
                    best_fused = fused
                    best_moe_name = "alpha_beta"

        cls_auc_w = 0.20

        stacked = _build_stacking_blend(market_pred, step3_pred)
        if stacked is not None:
            val = stacked[stacked["split"] == "val"]
            if not val.empty:
                y_true = val["target_direction_1d"].astype(int).to_numpy()
                y_proba = val["pred_direction_proba"].astype(float).to_numpy()
                cls = classification_metrics(y_true, y_proba)
                score = _safe_float(cls.get("f1")) + cls_auc_w * _safe_float(cls.get("auc"), fallback=0.5)
                if score > best_mix_score:
                    best_mix_score = float(score)
                    best_alpha = float("nan")
                    best_beta = float("nan")
                    best_fused = stacked
                    best_moe_name = "stacking_lr"

        for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            blended = _build_weighted_blend(market_pred, step3_pred, w_step3=float(w))
            val = blended[blended["split"] == "val"]
            if val.empty:
                continue
            y_true = val["target_direction_1d"].astype(int).to_numpy()
            y_proba = val["pred_direction_proba"].astype(float).to_numpy()
            cls = classification_metrics(y_true, y_proba)
            score = _safe_float(cls.get("f1")) + cls_auc_w * _safe_float(cls.get("auc"), fallback=0.5)
            if score > best_mix_score:
                best_mix_score = float(score)
                best_alpha = float("nan")
                best_beta = float(w)
                best_fused = blended
                best_moe_name = "weighted_blend"

        if best_fused is None:
            best_fused = _build_moe_blend(market_pred, step3_pred, alpha_event=0.0, beta_non_event=0.0)

        # Round-7: laggard-specific preference to avoid alpha_beta overfitting.
        laggard_pref_targets = {"00632R.TW", "2377.TW", "2454.TW"}
        if target in laggard_pref_targets and best_moe_name == "alpha_beta":
            cand = []
            for w in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
                b = _build_weighted_blend(market_pred, step3_pred, w_step3=float(w))
                v = b[b["split"] == "val"]
                if v.empty:
                    continue
                yt = v["target_direction_1d"].astype(int).to_numpy()
                yp = v["pred_direction_proba"].astype(float).to_numpy()
                cls = classification_metrics(yt, yp)
                sc = _safe_float(cls.get("f1")) + 0.15 * _safe_float(cls.get("auc"), fallback=0.5)
                cand.append((sc, "weighted_blend", w, b))
            st = _build_stacking_blend(market_pred, step3_pred)
            if st is not None:
                v = st[st["split"] == "val"]
                if not v.empty:
                    yt = v["target_direction_1d"].astype(int).to_numpy()
                    yp = v["pred_direction_proba"].astype(float).to_numpy()
                    cls = classification_metrics(yt, yp)
                    sc = _safe_float(cls.get("f1")) + 0.15 * _safe_float(cls.get("auc"), fallback=0.5)
                    cand.append((sc, "stacking_lr", float("nan"), st))
            if cand:
                cand = sorted(cand, key=lambda x: x[0], reverse=True)
                sc, name, wb, bf = cand[0]
                best_moe_name = name
                best_beta = wb
                best_alpha = float("nan")
                best_fused = bf

        val_base = market_pred[market_pred["split"] == "val"]
        val_fused = best_fused[best_fused["split"] == "val"]
        if (not val_base.empty) and (not val_fused.empty):
            f1_base = _safe_float(classification_metrics(
                val_base["target_direction_1d"].astype(int).to_numpy(),
                val_base["pred_direction_proba"].astype(float).to_numpy(),
            ).get("f1"), fallback=-1.0)
            f1_fused = _safe_float(classification_metrics(
                val_fused["target_direction_1d"].astype(int).to_numpy(),
                val_fused["pred_direction_proba"].astype(float).to_numpy(),
            ).get("f1"), fallback=-1.0)
            if f1_fused < f1_base:
                best_fused = market_pred.copy()
                best_alpha = float("nan")
                best_beta = float("nan")
                best_moe_name = "fallback_market_only"

        moe_row = _row_from_predframe(
            target=target,
            config="step3_moe_blend",
            model="mixture_blend(market_only,step3_integrated)",
            feature_set="event_gate_alpha_beta_blend",
            threshold=float(args.signal_threshold),
            frame=best_fused,
            selected_from=f"blend={best_moe_name}|alpha_event={best_alpha}|beta_non_event={best_beta}|obj=v16_round7_laggard_pref",
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
    parser = argparse.ArgumentParser(description="Integration runner v9: event-only linear head + adaptive MoE.")
    parser.add_argument("--targets", default="default", help="Comma-separated tickers, or 'default', 'all'/'all_legacy' (11 tickers).")
    parser.add_argument("--model", default="event_gated_mlp")
    parser.add_argument("--split", default="regime_aware", choices=["regime_matched", "all_history", "regime_aware"])
    parser.add_argument("--market-feature-set", default="TW_plus_global_market")
    parser.add_argument("--full-feature-set", default="Global_plus_Trump_with_gate")
    parser.add_argument("--feature-budget", type=int, default=80)
    parser.add_argument("--all-features", action="store_true")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--signal-threshold", type=float, default=0.55)
    parser.add_argument("--val-auc-weight", type=float, default=0.15)
    parser.add_argument("--deep-only", action="store_true", help="Use only deep models in step3 candidate pool.")
    parser.add_argument("--v8-linear-head", action="store_true", default=True, help="Enable deep+linear calibration head for deep models.")
    parser.add_argument("--metrics-output", default="integration_compare_metrics_step3_v9.csv")
    parser.add_argument("--predictions-output", default="integration_compare_predictions_step3_v9.csv")
    args = parser.parse_args()
    run_compare(args)


if __name__ == "__main__":
    main()
