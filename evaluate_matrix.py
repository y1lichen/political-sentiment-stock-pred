import os
import subprocess
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

STRATEGY_NAME = "full_event_market_gated_mlp"
BASELINE_STRATEGY_NAME = "market_only_gated_mlp"
OUTPUT_CSV_NAME = "my_model_vs_baseline_all_markets.csv"

# 你的 Gated MLP 啟動指令 (使用最好的設定)
BASE_CMD = [
    "python", "event_combo.py",
    "--hold", "1",
    "--presidential-terms-only",
    "--model-type", "gated_mlp",    
    "--binary-threshold", "0.0",
    "--auto-trade-threshold",
    "--trade-mode", "long_short",   # 開啟雙向交易
    "--epochs", "80",
    "--batch-size", "64"
]

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

    macro_f1 = f1_score(actual, pred, average="macro")
    precision = precision_score(actual, pred, pos_label=1, zero_division=0)
    recall = recall_score(actual, pred, pos_label=1, zero_division=0)
    accuracy = accuracy_score(actual, pred)
    try:
        auc = roc_auc_score(actual, prob_up)
    except ValueError:
        auc = 0.5

    cumret = total_return(strategy_ret)
    sharpe = sharpe_like(strategy_ret)

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


def calculate_metrics(target, full_pred_path, baseline_pred_path):
    """計算 full event+market model 與 market-only model baseline 的差異。"""
    full = prediction_metrics(full_pred_path)
    baseline = prediction_metrics(baseline_pred_path)

    return {
        "target": target,
        "strategy": STRATEGY_NAME,
        "baseline": BASELINE_STRATEGY_NAME,
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
    }

def main():
    results = []
    
    for target in TARGETS:
        print(f"\n{'-'*50}")
        print(f"🚀 開始訓練 full model 與 market-only baseline: {target}")
        print(f"{'-'*50}")
        
        target_out_dir = f"data/output/run_all_compare/{target}"
        full_out_dir = os.path.join(target_out_dir, "full")
        baseline_out_dir = os.path.join(target_out_dir, "market_only")
        os.makedirs(full_out_dir, exist_ok=True)
        os.makedirs(baseline_out_dir, exist_ok=True)
        
        full_cmd = BASE_CMD + [
            "--target", target,
            "--feature-set", "full",
            "--output-dir", full_out_dir,
        ]
        baseline_cmd = BASE_CMD + [
            "--target", target,
            "--feature-set", "market_only",
            "--output-dir", baseline_out_dir,
        ]
        
        try:
            subprocess.run(full_cmd, check=True)
            subprocess.run(baseline_cmd, check=True)
            
            full_pred_path = os.path.join(full_out_dir, "test_predictions.csv")
            baseline_pred_path = os.path.join(baseline_out_dir, "test_predictions.csv")
            if os.path.exists(full_pred_path) and os.path.exists(baseline_pred_path):
                target_metrics = calculate_metrics(target, full_pred_path, baseline_pred_path)
                results.append(target_metrics)
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

    if results:
        # 強制與同學報表的欄位順序一模一樣
        columns_order = [
            'target', 'strategy', 'baseline', 'macro_f1', 'd_macro_f1', 'precision', 'd_precision',
            'recall', 'd_recall', 'accuracy', 'd_accuracy', 'auc', 'd_auc', 
            'sharpe', 'd_sharpe', 'cumret', 'd_cumret', 'trades', 'd_trades',
            'baseline_macro_f1', 'baseline_precision', 'baseline_recall',
            'baseline_accuracy', 'baseline_auc', 'baseline_sharpe',
            'baseline_cumret', 'baseline_trades'
        ]
        
        df_results = pd.DataFrame(results)[columns_order]
        # 為了視覺上好對齊，我們把美股和台股照名字排序
        df_results = df_results.sort_values(by='target')
        
        df_results.to_csv(OUTPUT_CSV_NAME, index=False)
        print(f"\n🎉 評估完成！請查看跨市場報表: {OUTPUT_CSV_NAME}")
        
        # 用 to_string() 避免 tabulate 報錯
        print("\n📊 你的模型表現 (依 CumRet 排序):")
        print(df_results.sort_values(by='cumret', ascending=False)[['target', 'accuracy', 'd_accuracy', 'sharpe', 'cumret', 'd_cumret']].head(10).to_string(index=False))
    else:
        print("\n⚠️ 執行失敗，未產生結果。")

if __name__ == "__main__":
    main()