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
# 根據同學的 LEGACY_TARGETS
TARGETS = [
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
]

STRATEGY_NAME = "my_gated_mlp_long_short" 
OUTPUT_CSV_NAME = "my_model_vs_baseline.csv"

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

def calculate_metrics(target, pred_path):
    """計算與同學完全相同的指標和 Baseline (d_)"""
    df = pd.read_csv(pred_path)
    
    # 過濾出測試集 (假設 pred_path 裡已經是 test data)
    actual = df["actual_label"].astype(int).to_numpy()
    pred = df["pred_label"].astype(int).to_numpy()
    prob_up = df["prob_up"].astype(float).to_numpy()
    
    # 策略績效
    strategy_ret = df["strategy_ret_no_cost"].astype(float).to_numpy()
    trades = int((df["trade_signal"] != 0).sum())
    
    # Baseline: Buy and Hold (B&H)
    bh_ret = df["future_ret"].astype(float).to_numpy()
    bh_cumret = total_return(bh_ret)
    bh_sharpe = sharpe_like(bh_ret)
    
    # Baseline: 簡單多數決
    majority_class = int(pd.Series(actual).mode().iloc[0])
    majority_pred = np.full(len(actual), majority_class, dtype=int)
    
    # Baseline: 隨機亂猜
    bh_auc = 0.5 
    
    # --- 1. 計算你的模型指標 ---
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

    # --- 2. 計算 Baseline 分類指標 ---
    bh_macro_f1 = f1_score(actual, majority_pred, average="macro")
    bh_precision = precision_score(actual, majority_pred, pos_label=1, zero_division=0)
    bh_recall = recall_score(actual, majority_pred, pos_label=1, zero_division=0)
    bh_accuracy = accuracy_score(actual, majority_pred)
    total_days = len(actual) 
    
    # --- 3. 組合並計算 Delta (d_) ---
    return {
        "target": target,
        "strategy": STRATEGY_NAME,
        "macro_f1": macro_f1,
        "d_macro_f1": macro_f1 - bh_macro_f1,
        "precision": precision,
        "d_precision": precision - bh_precision,
        "recall": recall,
        "d_recall": recall - bh_recall,
        "accuracy": accuracy,
        "d_accuracy": accuracy - bh_accuracy,
        "auc": auc,
        "d_auc": auc - bh_auc,
        "sharpe": sharpe,
        "d_sharpe": sharpe - bh_sharpe,
        "cumret": cumret,
        "d_cumret": cumret - bh_cumret,
        "trades": trades,
        "d_trades": trades - total_days
    }

def main():
    results = []
    
    for target in TARGETS:
        print(f"\n{'-'*50}")
        print(f"🚀 開始訓練: {target}")
        print(f"{'-'*50}")
        
        target_out_dir = f"data/output/run_all_compare/{target}"
        os.makedirs(target_out_dir, exist_ok=True)
        
        cmd = BASE_CMD + ["--target", target, "--output-dir", target_out_dir]
        
        try:
            # 呼叫你的 event_combo.py 進行訓練
            subprocess.run(cmd, check=True)
            
            pred_path = os.path.join(target_out_dir, "test_predictions.csv")
            if os.path.exists(pred_path):
                target_metrics = calculate_metrics(target, pred_path)
                results.append(target_metrics)
                print(f"✅ {target} 完成! Accuracy: {target_metrics['accuracy']:.4f}, CumRet: {target_metrics['cumret']:.4f}")
            else:
                print(f"❌ 找不到預測檔: {pred_path}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ 訓練 {target} 失敗: {e}")
            continue

    if results:
        # 強制與同學報表的欄位順序一模一樣
        columns_order = [
            'target', 'strategy', 'macro_f1', 'd_macro_f1', 'precision', 'd_precision', 
            'recall', 'd_recall', 'accuracy', 'd_accuracy', 'auc', 'd_auc', 
            'sharpe', 'd_sharpe', 'cumret', 'd_cumret', 'trades', 'd_trades'
        ]
        
        df_results = pd.DataFrame(results)[columns_order]
        df_results = df_results.sort_values(by='target')
        
        df_results.to_csv(OUTPUT_CSV_NAME, index=False)
        print(f"\n🎉 評估完成！請查看報表: {OUTPUT_CSV_NAME}")
        
        # 用 to_string() 避免 tabulate 報錯
        print("\n📊 你的模型表現 (依 CumRet 排序):")
        print(df_results.sort_values(by='cumret', ascending=False)[['target', 'accuracy', 'sharpe', 'cumret', 'trades']].head(5).to_string(index=False))
    else:
        print("\n⚠️ 執行失敗，未產生結果。")

if __name__ == "__main__":
    main()