import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# 從我們建立的模組中引入功能
from src.data_loader import prepare_text_dataframe, load_market_features, CustomDataset, expanding_window_walk_forward
from src.metrics import compute_f1_scores, test_incremental_power
from src.trainer import train_and_eval_ablation

if __name__ == "__main__":
    VOLATILITY_WINDOW = 20
    Z_SCORE = 1.25
    
    base_dir = Path.cwd() 
    text_path = base_dir / "data/text/trump_posts_features_2017_2026.csv"
    if not text_path.exists(): raise FileNotFoundError(f"找不到文本檔案: {text_path}")
    text_df = prepare_text_dataframe(str(text_path))

    target_list = ["0050.TW", "00632R.TW", "00679B.TW", "2303.TW", "2308.TW", "2317.TW", "2330.TW", "2376.TW", "2377.TW", "2382.TW", "2454.TW", "3711.TW"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []

    for target_ticker in target_list:
        try:
            market_df = load_market_features(
                str(base_dir / "data/taiwan_market_data/global_prices.csv"),
                str(base_dir / "data/taiwan_market_data/global_volumes.csv"),
                str(base_dir / "data/taiwan_market_data/institutional_investors.csv"),
                str(base_dir / "data/taiwan_market_data/margin_trading.csv"),
                str(base_dir / "data/taiwan_market_data/tx_futures_night.csv"),
                target_ticker,
            )

            dataset = CustomDataset(
                market_df=market_df, text_df=text_df, window_size=20,
                close_price_col=f"close_{target_ticker}", open_price_col=f"close_{target_ticker}", 
            )

            splits = expanding_window_walk_forward(dataset.sample_index, 800, 100, 100, 100)
            print(f"\n=== Target {target_ticker} ===")
            
            if not splits:
                print(f"   [警告] {target_ticker} 有效資料天數({len(dataset.sample_index)})不足切分標準(最少1000天)。略過此標的。")
                continue

            all_val_targets, all_val_preds = [], []
            for split_idx, split in enumerate(splits, start=1):
                train_subset = torch.utils.data.Subset(dataset, split.train_idx)
                val_subset = torch.utils.data.Subset(dataset, split.val_idx)
                
                # 標準化處理 (Standardization)
                market_train = dataset.market_features[dataset.valid_indices[split.train_idx]]
                dataset.market_features = (dataset.market_features - market_train.mean(axis=0)) / np.where(market_train.std(axis=0) < 1e-8, 1.0, market_train.std(axis=0))
                text_train = dataset.text_features[dataset.valid_indices[split.train_idx]]
                dataset.text_features = (dataset.text_features - text_train.mean(axis=0)) / np.where(text_train.std(axis=0) < 1e-8, 1.0, text_train.std(axis=0))

                class_props = np.bincount(dataset.labels[dataset.valid_indices[split.train_idx]].astype(int), minlength=3) / max(1, len(split.train_idx))
                print(f"-- Split {split_idx} --")
                
                _, market_only_preds = train_and_eval_ablation(train_subset, val_subset, class_props, device, dataset[0][0], dataset[0][1], zero_text=True)
                y_true, full_model_preds = train_and_eval_ablation(train_subset, val_subset, class_props, device, dataset[0][0], dataset[0][1], zero_text=False)
                
                print(f"   [F1 Scores] Market Only: {compute_f1_scores(y_true, market_only_preds):.4f} | Full Model: {compute_f1_scores(y_true, full_model_preds):.4f}")
                test_incremental_power(market_only_preds, full_model_preds, y_true)

                all_val_targets.append(y_true)
                all_val_preds.append(full_model_preds)

            if all_val_targets:
                macro_avg_f1 = compute_f1_scores(np.concatenate(all_val_targets), np.concatenate(all_val_preds))
                metrics_rows.append({"target": target_ticker, "macro_avg_f1": macro_avg_f1})
                print(f"=== Target {target_ticker} Final Macro F1 = {macro_avg_f1:.4f} ===\n")
                
        except Exception as e:
            print(f"處理標的 {target_ticker} 時發生錯誤: {e}")

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(output_dir / "training_metrics_avg.csv", index=False)
        print(f"✅ 已儲存訓練指標至 {output_dir / 'training_metrics_avg.csv'}")