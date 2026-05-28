from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score

# 從我們建立的模組中引入功能
from src.data_loader import prepare_text_dataframe, load_market_features, CustomDataset, expanding_window_walk_forward
from src.metrics import compute_f1_scores, test_incremental_power
from src.trainer import train_and_eval_ablation
from evaluation.backtest import backtest
from evaluation.training_outputs import evaluate_training_predictions


def build_prediction_frame(target, split_idx, model_type, dates, y_true, y_pred, y_proba):
    if y_proba.shape[1] == 2:
        proba_down = y_proba[:, 0]
        proba_flat = np.zeros_like(proba_down)
        proba_up = y_proba[:, 1]
    else:
        proba_down = y_proba[:, 0]
        proba_flat = y_proba[:, 1]
        proba_up = y_proba[:, 2]
    return pd.DataFrame(
        {
            "target": target,
            "split": split_idx,
            "date": pd.DatetimeIndex(dates).strftime("%Y-%m-%d"),
            "y_true": y_true.astype(int),
            "pred_label": y_pred.astype(int),
            "proba_down": proba_down,
            "proba_flat": proba_flat,
            "proba_up": proba_up,
            "model_type": model_type,
        }
    )


def evaluate_event_binary_predictions(predictions, prices_csv, out_summary):
    prices = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index).normalize()
    rows = []

    for (target, model_type), group in predictions.groupby(["target", "model_type"]):
        group = group.sort_values("date").copy()
        close = prices[target].dropna()
        dates = pd.DatetimeIndex(pd.to_datetime(group["date"]).dt.normalize())
        if len(dates) == 0:
            continue

        y_true = group["y_true"].to_numpy(dtype=int)
        y_pred = group["pred_label"].to_numpy(dtype=int)
        precision = precision_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)

        start_pos = close.index.get_indexer([dates.min()])[0]
        end_pos = close.index.get_indexer([dates.max()])[0]
        if start_pos < 0 or end_pos < 0 or end_pos + 1 >= len(close):
            continue

        close_window = close.iloc[start_pos : end_pos + 2]
        signals = pd.Series(0, index=close_window.index[:-1], dtype=np.int8)
        # Binary event strategy: predict up -> long for next period; predict down -> cash.
        event_signals = pd.Series((y_pred == 1).astype(np.int8), index=dates)
        signals.loc[event_signals.index] = event_signals
        bt = backtest(close_window, signals)

        rows.append({
            "target": target,
            "model_type": model_type,
            "n_event_predictions": len(group),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_down": precision[0],
            "precision_up": precision[1],
            "hit_rate": float((y_true == y_pred).mean()),
            "cumulative_return": bt["cumulative_return"],
            "sharpe": bt["sharpe"],
            "max_drawdown": bt["max_drawdown"],
            "n_trades": bt["n_trades"],
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_summary, index=False)
    print(f"✅ 已儲存二元高訊號評估表至 {out_summary}")
    return summary

if __name__ == "__main__":
    TASK_MODE = "event_binary"  # "event_binary" or "ternary"
    VOLATILITY_WINDOW = 20
    Z_SCORE = 1.25
    EPOCHS = 20
    NUM_CLASSES = 2 if TASK_MODE == "event_binary" else 3
    SPLIT_CONFIG = (300, 80, 80, 80) if TASK_MODE == "event_binary" else (800, 100, 100, 100)
    
    base_dir = Path.cwd() 
    text_path = base_dir / "data/text/trump_posts_features_2017_2026.csv"
    if not text_path.exists(): raise FileNotFoundError(f"找不到文本檔案: {text_path}")
    text_df = prepare_text_dataframe(str(text_path))

    target_list = ["0050.TW", "00632R.TW", "00679B.TW", "2303.TW", "2308.TW", "2317.TW", "2330.TW", "2376.TW", "2377.TW", "2382.TW", "2454.TW", "3711.TW"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    
    output_dir = base_dir / "output"
    split_output_dir = output_dir / "split_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    metrics_by_model_rows = []
    prediction_frames = []

    for target_ticker in target_list:
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
            volatility_window=VOLATILITY_WINDOW, z_score=Z_SCORE,
            aggregation="trumpcode_daily",
            label_mode=TASK_MODE,
            high_signal_only=(TASK_MODE == "event_binary"),
            signal_intensity_threshold=2.0,
        )

        splits = expanding_window_walk_forward(dataset.sample_index, *SPLIT_CONFIG)
        print(f"\n=== Target {target_ticker} ===")
        
        if not splits:
            print(f"   [警告] {target_ticker} 有效資料天數({len(dataset.sample_index)})不足切分標準。略過此標的。")
            continue

        raw_market_features = dataset.market_features.copy()
        raw_text_features = dataset.text_features.copy()
        all_market_targets, all_market_preds = [], []
        all_full_targets, all_full_preds = [], []
        for split_idx, split in enumerate(splits, start=1):
            train_subset = torch.utils.data.Subset(dataset, split.train_idx)
            val_subset = torch.utils.data.Subset(dataset, split.val_idx)
            test_subset = torch.utils.data.Subset(dataset, split.test_idx)
            
            # 標準化處理 (Standardization)
            market_train = raw_market_features[dataset.valid_indices[split.train_idx]]
            dataset.market_features = (raw_market_features - market_train.mean(axis=0)) / np.where(market_train.std(axis=0) < 1e-8, 1.0, market_train.std(axis=0))
            text_train = raw_text_features[dataset.valid_indices[split.train_idx]]
            dataset.text_features = (raw_text_features - text_train.mean(axis=0)) / np.where(text_train.std(axis=0) < 1e-8, 1.0, text_train.std(axis=0))

            class_props = np.bincount(dataset.labels[dataset.valid_indices[split.train_idx]].astype(int), minlength=NUM_CLASSES) / max(1, len(split.train_idx))
            print(f"-- Split {split_idx} --")
            
            market_y_true, market_only_preds, market_only_proba, market_dates = train_and_eval_ablation(
                train_subset, val_subset, class_props, device, dataset[0][0], dataset[0][1],
                zero_text=True, eval_subset=test_subset, epochs=EPOCHS,
            )
            y_true, full_model_preds, full_model_proba, full_dates = train_and_eval_ablation(
                train_subset, val_subset, class_props, device, dataset[0][0], dataset[0][1],
                zero_text=False, eval_subset=test_subset, epochs=EPOCHS,
            )

            if not np.array_equal(market_y_true, y_true):
                raise ValueError(f"{target_ticker} split {split_idx}: pure_market and full_model y_true mismatch.")
            if not pd.DatetimeIndex(market_dates).equals(pd.DatetimeIndex(full_dates)):
                raise ValueError(f"{target_ticker} split {split_idx}: pure_market and full_model date mismatch.")
            
            print(f"   [F1 Scores] Market Only: {compute_f1_scores(y_true, market_only_preds):.4f} | Full Model: {compute_f1_scores(y_true, full_model_preds):.4f}")
            test_incremental_power(market_only_preds, full_model_preds, y_true)

            market_frame = build_prediction_frame(
                target_ticker, split_idx, "pure_market", market_dates,
                market_y_true, market_only_preds, market_only_proba,
            )
            full_frame = build_prediction_frame(
                target_ticker, split_idx, "full_model", full_dates,
                y_true, full_model_preds, full_model_proba,
            )
            market_frame.to_csv(split_output_dir / f"preds_{target_ticker}_split{split_idx}_pure_market.csv", index=False)
            full_frame.to_csv(split_output_dir / f"preds_{target_ticker}_split{split_idx}_full_model.csv", index=False)
            prediction_frames.extend([market_frame, full_frame])

            all_market_targets.append(market_y_true)
            all_market_preds.append(market_only_preds)
            all_full_targets.append(y_true)
            all_full_preds.append(full_model_preds)

        if all_full_targets:
            market_macro_f1 = compute_f1_scores(np.concatenate(all_market_targets), np.concatenate(all_market_preds))
            full_macro_f1 = compute_f1_scores(np.concatenate(all_full_targets), np.concatenate(all_full_preds))
            metrics_rows.append({"target": target_ticker, "macro_avg_f1": full_macro_f1})
            metrics_by_model_rows.append({"target": target_ticker, "model_type": "pure_market", "macro_avg_f1": market_macro_f1})
            metrics_by_model_rows.append({"target": target_ticker, "model_type": "full_model", "macro_avg_f1": full_macro_f1})
            print(f"=== Target {target_ticker} Final Macro F1 | Market Only = {market_macro_f1:.4f} | Full Model = {full_macro_f1:.4f} ===\n")
            
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(output_dir / "training_metrics_avg.csv", index=False)
        print(f"✅ 已儲存訓練指標至 {output_dir / 'training_metrics_avg.csv'}")
    if metrics_by_model_rows:
        pd.DataFrame(metrics_by_model_rows).to_csv(output_dir / "training_metrics_by_model.csv", index=False)
        print(f"✅ 已儲存分模型訓練指標至 {output_dir / 'training_metrics_by_model.csv'}")
    if prediction_frames:
        predictions_path = output_dir / "training_predictions.csv"
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions.to_csv(predictions_path, index=False)
        print(f"✅ 已儲存模型預測至 {predictions_path}")
        if TASK_MODE == "event_binary":
            evaluate_event_binary_predictions(
                predictions=predictions,
                prices_csv=base_dir / "data/taiwan_market_data/global_prices.csv",
                out_summary=output_dir / "event_binary_summary.csv",
            )
        else:
            evaluate_training_predictions(
                predictions_path=predictions_path,
                prices_csv=base_dir / "data/taiwan_market_data/global_prices.csv",
                out_summary=output_dir / "evaluation_summary.csv",
                out_cm_dir=output_dir / "confusion_matrices",
            )
