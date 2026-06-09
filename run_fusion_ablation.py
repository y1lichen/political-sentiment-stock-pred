import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

FUSION_MODES = ["gated_concat", "raw_concat", "ungated_concat", "add"]


def total_return(returns):
    returns = pd.Series(returns).fillna(0.0)
    return float((1.0 + returns).prod() - 1.0)


def sharpe_like(returns):
    returns = pd.Series(returns).fillna(0.0)
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(252) * returns.mean() / std)


def prediction_metrics(pred_path):
    df = pd.read_csv(pred_path)
    actual = df["actual_label"].astype(int).to_numpy()
    pred = df["pred_label"].astype(int).to_numpy()
    prob_up = df["prob_up"].astype(float).to_numpy()
    strategy_ret = df["strategy_ret_no_cost"].astype(float).to_numpy()
    trades = int((df["trade_signal"] != 0).sum())
    try:
        auc = roc_auc_score(actual, prob_up)
    except ValueError:
        auc = 0.5
    return {
        "accuracy": float(accuracy_score(actual, pred)),
        "macro_f1": float(f1_score(actual, pred, average="macro")),
        "precision": float(precision_score(actual, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(actual, pred, pos_label=1, zero_division=0)),
        "auc": float(auc),
        "cumret": total_return(strategy_ret),
        "sharpe": sharpe_like(strategy_ret),
        "trades": trades,
        "n_test": int(len(df)),
    }


def load_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_one(args, target, fusion_mode, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "event_combo.py",
        "--target", target,
        "--hold", str(args.hold),
        "--model-type", "gated_mlp",
        "--fusion-mode", fusion_mode,
        "--feature-set", "full",
        "--binary-threshold", "0.0",
        "--auto-trade-threshold",
        "--trade-mode", args.trade_mode,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--output-dir", str(out_dir),
    ]
    if args.presidential_terms_only:
        cmd.append("--presidential-terms-only")
    else:
        cmd.append("--no-presidential-terms-only")

    with (out_dir / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=args.repo_root, check=True, stdout=log, stderr=subprocess.STDOUT)

    metrics = prediction_metrics(out_dir / "test_predictions.csv")
    summary = load_json(out_dir / "summary.json")
    features = summary.get("features", {})
    test_event_days = summary.get("test_event_days", {})
    return {
        "target": target,
        "fusion_mode": fusion_mode,
        **metrics,
        "event_day_cumret": test_event_days.get("total_return"),
        "event_day_trades": test_event_days.get("long_count", 0) + test_event_days.get("short_count", 0),
        "selected_trade_edge_threshold": summary.get("selected_trade_edge_threshold"),
        "market_features": features.get("market"),
        "event_regime_features": features.get("event_regime"),
        "selected_events": features.get("selected_events"),
        "output_dir": str(out_dir),
    }


def write_report(df, output_path, args):
    lines = []
    lines.append("# Fusion Mode Ablation Report")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Targets: `{args.targets}`")
    lines.append(f"- Fusion modes: `{', '.join(args.fusion_modes)}`")
    lines.append(f"- Epochs: `{args.epochs}`, patience: `{args.patience}`")
    lines.append(f"- Seed: `{args.seed}`")
    lines.append("- Feature set: `full`")
    lines.append("- Proposed baseline fusion: `gated_concat`")
    lines.append("")
    lines.append("## Fusion Modes")
    lines.append("")
    lines.append("- `gated_concat`: original model, DirectionHead([market_state, gate * event_state]).")
    lines.append("- `raw_concat`: directly concatenates scaled raw market/event features into Direction Head.")
    lines.append("- `ungated_concat`: DirectionHead([market_state, event_state]) without gate modulation.")
    lines.append("- `add`: DirectionHead(market_state + gate * event_state).")
    lines.append("")
    for target in df["target"].unique():
        sub = df[df["target"] == target].copy()
        base = sub[sub["fusion_mode"] == "gated_concat"].iloc[0]
        lines.append(f"## {target}")
        lines.append("")
        lines.append("| fusion_mode | accuracy | d_acc vs gated | AUC | d_AUC vs gated | cumret | d_cumret vs gated | trades |")
        lines.append("| :-- | --: | --: | --: | --: | --: | --: | --: |")
        for _, row in sub.iterrows():
            lines.append(
                "| {fusion_mode} | {accuracy:.3f} | {d_acc:+.3f} | {auc:.3f} | {d_auc:+.3f} | {cumret:.3f} | {d_cumret:+.3f} | {trades:.0f} |".format(
                    fusion_mode=row["fusion_mode"],
                    accuracy=row["accuracy"],
                    d_acc=row["accuracy"] - base["accuracy"],
                    auc=row["auc"],
                    d_auc=row["auc"] - base["auc"],
                    cumret=row["cumret"],
                    d_cumret=row["cumret"] - base["cumret"],
                    trades=row["trades"],
                )
            )
        best_acc = sub.sort_values("accuracy", ascending=False).iloc[0]
        best_auc = sub.sort_values("auc", ascending=False).iloc[0]
        best_ret = sub.sort_values("cumret", ascending=False).iloc[0]
        lines.append("")
        lines.append(
            f"Best accuracy: `{best_acc['fusion_mode']}` ({best_acc['accuracy']:.3f}); "
            f"best AUC: `{best_auc['fusion_mode']}` ({best_auc['auc']:.3f}); "
            f"best cumret: `{best_ret['fusion_mode']}` ({best_ret['cumret']:.3f})."
        )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run fusion-mode ablation experiments.")
    parser.add_argument("--repo-root", default="/home/butter303062/political-sentiment-stock-pred")
    parser.add_argument("--targets", default="2330.TW,TSM")
    parser.add_argument("--fusion-modes", default=",".join(FUSION_MODES))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trade-mode", choices=["long_short", "long_cash", "short_cash"], default="long_short")
    parser.add_argument("--presidential-terms-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="data/output/fusion_ablation_core")
    parser.add_argument("--output-csv", default="data/output/fusion_ablation_core/fusion_ablation_results.csv")
    parser.add_argument("--output-md", default="data/output/fusion_ablation_core/fusion_ablation_report.md")
    args = parser.parse_args()
    args.repo_root = str(Path(args.repo_root))
    args.targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    args.fusion_modes = [x.strip() for x in args.fusion_modes.split(",") if x.strip()]

    base_output = Path(args.repo_root) / args.output_dir
    rows = []
    for target in args.targets:
        for fusion_mode in args.fusion_modes:
            out_dir = base_output / target.replace("/", "_") / fusion_mode
            print(f"[run] target={target} fusion_mode={fusion_mode}")
            rows.append(run_one(args, target, fusion_mode, out_dir))
            df = pd.DataFrame(rows)
            output_csv = Path(args.repo_root) / args.output_csv
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)

    df = pd.DataFrame(rows)
    output_csv = Path(args.repo_root) / args.output_csv
    output_md = Path(args.repo_root) / args.output_md
    df.to_csv(output_csv, index=False)
    write_report(df, output_md, args)
    print(f"Saved results: {output_csv}")
    print(f"Saved report: {output_md}")
    print(df[["target", "fusion_mode", "accuracy", "auc", "cumret", "trades"]].to_string(index=False))


if __name__ == "__main__":
    main()
