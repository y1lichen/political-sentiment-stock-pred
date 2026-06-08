import os
import subprocess
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score
)

# === 設定檔 ===
# 台美股合併實驗: 4 個美股標的 + 11 個台股標的
# event_combo.py 會依 target 格式自動挑選對應的市場特徵集
TARGETS = [
    # === 美股標的 ===
    "TSM",     # 台積電 ADR
    "^SOX",    # 費城半導體
    "^NDX",    # Nasdaq 100
    "^GSPC",   # S&P 500
    
    # === 台股標的 ===
    "0050.TW",   # 台灣 50
    "00632R.TW", # 台灣 50 反 1
    "2303.TW",   # 聯電
    "2308.TW",   # 台達電
    "2317.TW",   # 鴻海
    "2330.TW",   # 台積電
    "2376.TW",   # 技嘉
    "2377.TW",   # 微星
    "2382.TW",   # 廣達
    "2454.TW",   # 聯發科
    "3711.TW",   # 日月光投控
]

OUTPUT_CSV_NAME = "my_model_vs_baseline_all_markets.csv"
DEFAULT_HORIZONS = "3:1,5:1,10:3,20:5,20:10,60:20"

EVENT_FEATURE_FILES = [
    "data/output/trump_posts_with_event_features_us.csv",
    "data/output/trump_posts_with_event_features_tw.csv",
]

BASE_CMD = [
    "--presidential-terms-only",
    "--binary-threshold", "0.0",
    "--auto-trade-threshold",
    "--trade-mode", "long_short",   # 開啟雙向交易
    "--min-score", "0.03",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run full event+market vs market-only baseline across targets and "
            "multiple lookback/holding horizons."
        )
    )
    parser.add_argument(
        "--horizons",
        default=DEFAULT_HORIZONS,
        help=(
            "Comma-separated window:hold pairs. Example: '3:1,20:10'. "
            "Default tests short, medium, and longer event horizons."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=["lstm", "gated_mlp"],
        default="lstm",
        help=(
            "Use lstm for true N-day lookback sequences. gated_mlp is faster, "
            "but --window only affects split overlap."
        ),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-csv", default=OUTPUT_CSV_NAME)
    parser.add_argument("--output-root", default="data/output/run_all_compare")
    parser.add_argument(
        "--targets",
        default=",".join(TARGETS),
        help="Comma-separated targets to run. Defaults to all configured TW/US targets.",
    )
    parser.add_argument(
        "--python-bin",
        default="python",
        help="Python executable used to launch event_combo.py on the server.",
    )
    return parser.parse_args()


def parse_horizons(text):
    horizons = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid horizon spec {raw!r}; expected window:hold, e.g. 20:10.")
        window, hold = [int(x) for x in parts]
        if window <= 0 or hold <= 0:
            raise ValueError(f"Invalid horizon spec {raw!r}; window and hold must be positive.")
        if hold > window:
            print(
                f"Warning: horizon {window}:{hold} has hold > window. "
                "This is allowed, but usually harder to model."
            )
        horizons.append({"window": window, "hold": hold, "horizon": f"{window}d_to_{hold}d"})
    if not horizons:
        raise ValueError("No valid horizons were provided.")
    return horizons


def parse_targets(text):
    targets = [x.strip() for x in text.split(",") if x.strip()]
    return targets or TARGETS


def ensure_event_features():
    """Ensure event_combo.py will use the relabeled per-market Trump event data."""
    if all(os.path.exists(path) for path in EVENT_FEATURE_FILES):
        return
    print("Relabeled Trump event files missing; running data/data_preprocess.py first.")
    subprocess.run(["python", "data/data_preprocess.py"], check=True)

def total_return(returns):
    returns = pd.Series(returns).fillna(0.0)
    return float((1.0 + returns).prod() - 1.0)

def sharpe_like(returns, hold=1):
    returns = pd.Series(returns).fillna(0.0)
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    periods_per_year = 252 / max(int(hold), 1)
    return float(np.sqrt(periods_per_year) * returns.mean() / std)

def prediction_metrics(pred_path, hold=1):
    df = pd.read_csv(pred_path)

    actual = df["actual_label"].astype(int).to_numpy()
    pred = df["pred_label"].astype(int).to_numpy()
    prob_up = df["prob_up"].astype(float).to_numpy()
    strategy_ret = df["strategy_ret_no_cost"].astype(float).to_numpy()
    trades = int((df["trade_signal"] != 0).sum())

    macro_f1 = f1_score(actual, pred, average="macro")
    precision = precision_score(actual, pred, pos_label=1, zero_division=0)
    recall = recall_score(actual, pred, pos_label=1, zero_division=0)
    accuracy = accuracy_score(actual, pred)
    try:
        auc = roc_auc_score(actual, prob_up)
    except ValueError:
        auc = 0.5

    cumret = total_return(strategy_ret)
    sharpe = sharpe_like(strategy_ret, hold=hold)

    return {
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "auc": auc,
        "sharpe": sharpe,
        "cumret": cumret,
        "trades": trades,
    }


def calculate_metrics(target, full_pred_path, baseline_pred_path, window, hold, model_type):
    """計算 full event+market model 與 market-only model baseline 的差異。"""
    full = prediction_metrics(full_pred_path, hold=hold)
    baseline = prediction_metrics(baseline_pred_path, hold=hold)
    full_event_days = summary_event_days(os.path.join(os.path.dirname(full_pred_path), "summary.json"))
    baseline_event_days = summary_event_days(os.path.join(os.path.dirname(baseline_pred_path), "summary.json"))
    full_rule_event_days = summary_event_days(
        os.path.join(os.path.dirname(full_pred_path), "summary.json"),
        key="test_rule_event_days",
    )
    full_event_day_threshold = summary_event_days(
        os.path.join(os.path.dirname(full_pred_path), "summary.json"),
        key="test_model_event_day_threshold",
    )
    full_hybrid_event_days = summary_event_days(
        os.path.join(os.path.dirname(full_pred_path), "summary.json"),
        key="test_hybrid_event_days",
    )

    return {
        "window": window,
        "hold": hold,
        "horizon": f"{window}d_to_{hold}d",
        "model_type": model_type,
        "target": target,
        "strategy": f"full_event_market_{model_type}",
        "baseline": f"market_only_{model_type}",
        "macro_f1": full["macro_f1"],
        "d_macro_f1": full["macro_f1"] - baseline["macro_f1"],
        "precision": full["precision"],
        "d_precision": full["precision"] - baseline["precision"],
        "recall": full["recall"],
        "d_recall": full["recall"] - baseline["recall"],
        "accuracy": full["accuracy"],
        "d_accuracy": full["accuracy"] - baseline["accuracy"],
        "auc": full["auc"],
        "d_auc": full["auc"] - baseline["auc"],
        "sharpe": full["sharpe"],
        "d_sharpe": full["sharpe"] - baseline["sharpe"],
        "cumret": full["cumret"],
        "d_cumret": full["cumret"] - baseline["cumret"],
        "trades": full["trades"],
        "d_trades": full["trades"] - baseline["trades"],
        "baseline_macro_f1": baseline["macro_f1"],
        "baseline_precision": baseline["precision"],
        "baseline_recall": baseline["recall"],
        "baseline_accuracy": baseline["accuracy"],
        "baseline_auc": baseline["auc"],
        "baseline_sharpe": baseline["sharpe"],
        "baseline_cumret": baseline["cumret"],
        "baseline_trades": baseline["trades"],
        "event_days": full_event_days["event_days"],
        "event_coverage": full_event_days["event_coverage"],
        "event_days_macro_f1": full_event_days["macro_f1"],
        "d_event_days_macro_f1": full_event_days["macro_f1"] - baseline_event_days["macro_f1"],
        "event_days_up_precision": full_event_days["up_precision"],
        "d_event_days_up_precision": full_event_days["up_precision"] - baseline_event_days["up_precision"],
        "event_days_up_recall": full_event_days["up_recall"],
        "d_event_days_up_recall": full_event_days["up_recall"] - baseline_event_days["up_recall"],
        "event_days_up_f1": full_event_days["up_f1"],
        "d_event_days_up_f1": full_event_days["up_f1"] - baseline_event_days["up_f1"],
        "event_days_down_precision": full_event_days["down_precision"],
        "d_event_days_down_precision": full_event_days["down_precision"] - baseline_event_days["down_precision"],
        "event_days_down_recall": full_event_days["down_recall"],
        "d_event_days_down_recall": full_event_days["down_recall"] - baseline_event_days["down_recall"],
        "event_days_down_f1": full_event_days["down_f1"],
        "d_event_days_down_f1": full_event_days["down_f1"] - baseline_event_days["down_f1"],
        "event_days_precision": full_event_days["up_precision"],
        "d_event_days_precision": full_event_days["up_precision"] - baseline_event_days["up_precision"],
        "event_days_recall": full_event_days["up_recall"],
        "d_event_days_recall": full_event_days["up_recall"] - baseline_event_days["up_recall"],
        "event_days_accuracy": full_event_days["accuracy"],
        "d_event_days_accuracy": full_event_days["accuracy"] - baseline_event_days["accuracy"],
        "event_days_auc": full_event_days["auc"],
        "d_event_days_auc": full_event_days["auc"] - baseline_event_days["auc"],
        "event_days_sharpe": full_event_days["sharpe"],
        "d_event_days_sharpe": full_event_days["sharpe"] - baseline_event_days["sharpe"],
        "event_days_cumret": full_event_days["strategy_total_return_no_cost"],
        "d_event_days_cumret": (
            full_event_days["strategy_total_return_no_cost"]
            - baseline_event_days["strategy_total_return_no_cost"]
        ),
        "event_days_trade_accuracy": full_event_days["trade_accuracy"],
        "d_event_days_trade_accuracy": (
            full_event_days["trade_accuracy"] - baseline_event_days["trade_accuracy"]
        ),
        "event_days_trades": full_event_days["trade_count"],
        "d_event_days_trades": full_event_days["trade_count"] - baseline_event_days["trade_count"],
        "baseline_event_days": baseline_event_days["event_days"],
        "baseline_event_days_macro_f1": baseline_event_days["macro_f1"],
        "baseline_event_days_up_precision": baseline_event_days["up_precision"],
        "baseline_event_days_up_recall": baseline_event_days["up_recall"],
        "baseline_event_days_up_f1": baseline_event_days["up_f1"],
        "baseline_event_days_down_precision": baseline_event_days["down_precision"],
        "baseline_event_days_down_recall": baseline_event_days["down_recall"],
        "baseline_event_days_down_f1": baseline_event_days["down_f1"],
        "baseline_event_days_precision": baseline_event_days["up_precision"],
        "baseline_event_days_recall": baseline_event_days["up_recall"],
        "baseline_event_days_accuracy": baseline_event_days["accuracy"],
        "baseline_event_days_auc": baseline_event_days["auc"],
        "baseline_event_days_sharpe": baseline_event_days["sharpe"],
        "baseline_event_days_cumret": baseline_event_days["strategy_total_return_no_cost"],
        "baseline_event_days_trade_accuracy": baseline_event_days["trade_accuracy"],
        "baseline_event_days_trades": baseline_event_days["trade_count"],
        "rule_event_days_cumret": full_rule_event_days["strategy_total_return_no_cost"],
        "rule_event_days_trade_accuracy": full_rule_event_days["trade_accuracy"],
        "rule_event_days_trades": full_rule_event_days["trade_count"],
        "event_day_threshold_cumret": full_event_day_threshold["strategy_total_return_no_cost"],
        "event_day_threshold_trade_accuracy": full_event_day_threshold["trade_accuracy"],
        "event_day_threshold_trades": full_event_day_threshold["trade_count"],
        "hybrid_event_days_cumret": full_hybrid_event_days["strategy_total_return_no_cost"],
        "hybrid_event_days_trade_accuracy": full_hybrid_event_days["trade_accuracy"],
        "hybrid_event_days_trades": full_hybrid_event_days["trade_count"],
    }


def summary_event_days(summary_path, key="test_event_days"):
    defaults = {
        "event_days": 0,
        "event_coverage": 0.0,
        "macro_f1": 0.0,
        "up_precision": 0.0,
        "up_recall": 0.0,
        "up_f1": 0.0,
        "down_precision": 0.0,
        "down_recall": 0.0,
        "down_f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
        "auc": 0.5,
        "sharpe": 0.0,
        "strategy_total_return_no_cost": 0.0,
        "trade_accuracy": 0.0,
        "trade_count": 0,
    }
    if not os.path.exists(summary_path):
        return defaults
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {**defaults, **data.get(key, {})}

def main():
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    targets = parse_targets(args.targets)
    ensure_event_features()
    results = []

    print("Horizon settings:")
    for h in horizons:
        print(f"  - window={h['window']} days -> hold={h['hold']} days ({h['horizon']})")
    print(f"Model type: {args.model_type}")

    for horizon_cfg in horizons:
        window = horizon_cfg["window"]
        hold = horizon_cfg["hold"]
        horizon_name = horizon_cfg["horizon"]
        horizon_results = []

        for target in targets:
            print(f"\n{'-'*70}")
            print(
                f"🚀 開始訓練: {target} | window={window}d -> hold={hold}d | "
                f"model={args.model_type}"
            )
            print(f"{'-'*70}")

            target_out_dir = os.path.join(args.output_root, horizon_name, target)
            full_out_dir = os.path.join(target_out_dir, "full")
            baseline_out_dir = os.path.join(target_out_dir, "market_only")
            os.makedirs(full_out_dir, exist_ok=True)
            os.makedirs(baseline_out_dir, exist_ok=True)

            common_cmd = [
                args.python_bin,
                "event_combo.py",
                *BASE_CMD,
                "--hold", str(hold),
                "--window", str(window),
                "--model-type", args.model_type,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--target", target,
            ]
            full_cmd = common_cmd + [
                "--feature-set", "full",
                "--output-dir", full_out_dir,
            ]
            baseline_cmd = common_cmd + [
                "--feature-set", "market_only",
                "--output-dir", baseline_out_dir,
            ]

            try:
                subprocess.run(full_cmd, check=True)
                subprocess.run(baseline_cmd, check=True)

                full_pred_path = os.path.join(full_out_dir, "test_predictions.csv")
                baseline_pred_path = os.path.join(baseline_out_dir, "test_predictions.csv")
                if os.path.exists(full_pred_path) and os.path.exists(baseline_pred_path):
                    target_metrics = calculate_metrics(
                        target,
                        full_pred_path,
                        baseline_pred_path,
                        window=window,
                        hold=hold,
                        model_type=args.model_type,
                    )
                    results.append(target_metrics)
                    horizon_results.append(target_metrics)
                    print(
                        f"✅ {target} 完成! "
                        f"Accuracy: {target_metrics['accuracy']:.4f} "
                        f"(d={target_metrics['d_accuracy']:+.4f}), "
                        f"CumRet: {target_metrics['cumret']:.4f} "
                        f"(d={target_metrics['d_cumret']:+.4f})"
                    )
                else:
                    print(f"❌ 找不到預測檔: {full_pred_path} 或 {baseline_pred_path}")

            except subprocess.CalledProcessError as e:
                print(f"❌ 訓練 {target} 失敗: {e}")
                continue

        if horizon_results:
            horizon_path = args.output_csv.replace(".csv", f"_{horizon_name}.csv")
            pd.DataFrame(horizon_results).sort_values(by="target").to_csv(horizon_path, index=False)
            print(f"📄 已輸出 horizon 子表: {horizon_path}")

    if results:
        # Keep horizon metadata first, followed by the original model-vs-baseline metrics.
        columns_order = [
            'window', 'hold', 'horizon', 'model_type',
            'target', 'strategy', 'baseline', 'macro_f1', 'd_macro_f1', 'precision', 'd_precision',
            'recall', 'd_recall', 'accuracy', 'd_accuracy', 'auc', 'd_auc', 
            'sharpe', 'd_sharpe', 'cumret', 'd_cumret', 'trades', 'd_trades',
            'baseline_macro_f1', 'baseline_precision', 'baseline_recall',
            'baseline_accuracy', 'baseline_auc', 'baseline_sharpe',
            'baseline_cumret', 'baseline_trades',
            'event_days', 'event_coverage',
            'event_days_macro_f1', 'd_event_days_macro_f1',
            'event_days_up_precision', 'd_event_days_up_precision',
            'event_days_up_recall', 'd_event_days_up_recall',
            'event_days_up_f1', 'd_event_days_up_f1',
            'event_days_down_precision', 'd_event_days_down_precision',
            'event_days_down_recall', 'd_event_days_down_recall',
            'event_days_down_f1', 'd_event_days_down_f1',
            'event_days_precision', 'd_event_days_precision',
            'event_days_recall', 'd_event_days_recall',
            'event_days_accuracy', 'd_event_days_accuracy',
            'event_days_auc', 'd_event_days_auc',
            'event_days_sharpe', 'd_event_days_sharpe',
            'event_days_cumret', 'd_event_days_cumret',
            'event_days_trade_accuracy', 'd_event_days_trade_accuracy',
            'event_days_trades', 'd_event_days_trades',
            'baseline_event_days', 'baseline_event_days_macro_f1',
            'baseline_event_days_up_precision', 'baseline_event_days_up_recall',
            'baseline_event_days_up_f1', 'baseline_event_days_down_precision',
            'baseline_event_days_down_recall', 'baseline_event_days_down_f1',
            'baseline_event_days_precision', 'baseline_event_days_recall',
            'baseline_event_days_accuracy', 'baseline_event_days_auc',
            'baseline_event_days_sharpe', 'baseline_event_days_cumret',
            'baseline_event_days_trade_accuracy', 'baseline_event_days_trades',
            'rule_event_days_cumret', 'rule_event_days_trade_accuracy',
            'rule_event_days_trades', 'event_day_threshold_cumret',
            'event_day_threshold_trade_accuracy', 'event_day_threshold_trades',
            'hybrid_event_days_cumret', 'hybrid_event_days_trade_accuracy',
            'hybrid_event_days_trades'
        ]
        
        df_results = pd.DataFrame(results)[columns_order]
        # 為了視覺上好對齊，我們把美股和台股照名字排序
        df_results = df_results.sort_values(by=['window', 'hold', 'target'])
        
        df_results.to_csv(args.output_csv, index=False)
        print(f"\n🎉 評估完成！請查看跨市場報表: {args.output_csv}")

        horizon_summary_cols = [
            "accuracy",
            "d_accuracy",
            "macro_f1",
            "d_macro_f1",
            "auc",
            "d_auc",
            "sharpe",
            "d_sharpe",
            "cumret",
            "d_cumret",
            "event_days_accuracy",
            "d_event_days_accuracy",
            "event_days_cumret",
            "d_event_days_cumret",
            "hybrid_event_days_cumret",
            "event_day_threshold_cumret",
        ]
        horizon_summary = (
            df_results.groupby(["window", "hold", "horizon", "model_type"], as_index=False)[horizon_summary_cols]
            .mean()
            .sort_values(["window", "hold"])
        )
        summary_path = args.output_csv.replace(".csv", "_horizon_summary.csv")
        horizon_summary.to_csv(summary_path, index=False)
        print(f"📄 Horizon 平均表: {summary_path}")
        
        # 用 to_string() 避免 tabulate 報錯
        print("\n📊 你的模型表現 (依 CumRet 排序):")
        print(
            df_results.sort_values(by='cumret', ascending=False)[
                ['horizon', 'target', 'accuracy', 'd_accuracy', 'sharpe', 'cumret', 'd_cumret']
            ].head(10).to_string(index=False)
        )
    else:
        print("\n⚠️ 執行失敗，未產生結果。")

if __name__ == "__main__":
    main()
