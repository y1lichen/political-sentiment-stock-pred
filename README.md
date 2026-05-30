# Deep Trump Code

以 Trump 貼文事件與台灣市場資料預測台股短期方向的低容量深度學習管線。設計依據 `PROJECT_PLAN_DEEP_TRUMP_CODE.md`：小模型優先、事件觸發、可拒絕交易，避免在低信噪比金融時序上使用過高容量模型。

## Ubuntu Server 快速開始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

建立資料集：

```bash
bash scripts/build_dataset.sh 2330.TW
```

訓練模型：

```bash
bash scripts/train.sh 2330.TW event_gated_mlp regime_aware
```

也可訓練其他模型：

```bash
bash scripts/train.sh 2330.TW logistic regime_aware
bash scripts/train.sh 2330.TW elasticnet regime_matched
bash scripts/train.sh 2454.TW random_forest all_history
bash scripts/train.sh 0050.TW small_mlp regime_aware
```

推理最新一列：

```bash
bash scripts/predict.sh 2330.TW outputs/models/event_gated_mlp_2330_TW_regime_aware_Global_plus_Trump_with_gate.pt latest
```

訓練 Market baseline + Trump overlay（單一標的）：

```bash
bash scripts/train_overlay.sh 2330.TW lightgbm elasticnet regime_aware
# args: TARGET MARKET_MODEL OVERLAY_MODEL SPLIT
```

若 server 尚未安裝 LightGBM，可先用 sklearn 版本跑通：

```bash
bash scripts/train_overlay.sh 2330.TW logistic elasticnet regime_aware
```

批次訓練全部標的（含交易成本與滑點設定）：

```bash
bash scripts/train_overlay_all_tickers.sh lightgbm elasticnet regime_aware 0.001 0.0005
# args: MARKET_MODEL OVERLAY_MODEL SPLIT TRANSACTION_COST SLIPPAGE
```

Overlay 管線會先用 `TW_plus_global_market` 訓練 market baseline，再用 Trump text/regime features 訓練兩個 overlay：

- `profit_model`：在 Trump event days 判斷 market signal 是否該被 veto。
- `direction_model`：在 Trump event days 且 market baseline 沒出手時，判斷是否 override 成 LONG / SHORT。

主要比較輸出是：

```text
market_only
market_plus_trump_overlay
```

如果 overlay 的 test `return_after_costs`、`compound_nav`、`avg_signal_return` 或 `cumulative_return` 高於 market baseline，才代表 Trump overlay 在目前設定下真的改善收益。

輸出位置：

- 資料集：`outputs/datasets/`
- 模型：`outputs/models/`
- 預測：`outputs/predictions/`
- 指標：`outputs/reports/`
- Overlay 報告：`outputs/reports/report_overlay_*.md`

## 重要設計

- 每列資料代表一個台股交易日開盤前可用資訊。
- 市場、法人、融資券、夜盤特徵採保守滯後，降低未來資訊洩漏。
- `event_gate_default` 會讓模型在沒有 Trump 政策事件時傾向 `NO_TRADE`。
- `regime_aware` 與 `all_history` 使用相同的日期切分（train: 2017–2022、val: 2023–2024、test: 2025+），差異在於 `regime_aware` 會對 COVID 政策期間（2020-02 至 2021-12）的樣本降權至 ×0.75。
- `2025-01-20` 之後保留為最終測試期。

## CLI 參數

```bash
python -m src.training.train --help
python -m src.inference.predict --help
```

## Integration Compare（`train_integration_compare.py` 最新版）

本章節說明 `src/training/train_integration_compare.py` 的最新整合實驗流程（目前為 v7 邏輯）。這支程式的目的，是在同一資料切分、同一評估口徑下，直接比較：

- `market_only`：純市場 baseline
- `trump_full`：市場 + Trump 特徵（含 gate）
- `step3_integrated`：從多個 Step3 候選組合中自動挑選的最佳配置
- `step3_moe_blend`：將 `market_only` 與 `step3_integrated` 依事件日做機率融合（Mixture-of-Experts 風格）

程式會輸出一份「可直接比較」的總表（CSV），避免手動彙整不同模型結果。

### 核心概念

1. 先固定跑兩條主線：
- `market_only`（`TW_plus_global_market`）
- `trump_full`（`Global_plus_Trump_with_gate`）

2. 再跑 Step3 候選集合（模型 + 特徵集 + feature budget + threshold）。

3. 用 validation 指標選最佳 `step3_integrated`：
- 一般標的：`val_f1 + λ * val_auc`
- `2454.TW`：額外做防過擬合懲罰（限制候選模型族 + train/val gap 懲罰）

4. 建立 `step3_moe_blend`：
- 事件日（`event_gate_default == 1`）用 `alpha_event` 融合
- 非事件日用 `beta_non_event` 融合
- 在 validation 上搜尋最佳 `alpha_event` / `beta_non_event`

### 目標標的模式

`--targets` 支援三種模式：

- `default`：讀 `src/config.py` 的 `TARGETS`
- `all` 或 `all_legacy`：11 檔固定清單
- 自訂逗號清單，例如：`2330.TW,2454.TW,0050.TW`

目前 `all_legacy` 清單為：

```text
0050.TW, 00632R.TW, 2303.TW, 2308.TW, 2317.TW,
2330.TW, 2376.TW, 2377.TW, 2382.TW, 2454.TW, 3711.TW
```

### 主要參數

```bash
python -m src.training.train_integration_compare --help
```

常用參數：

- `--targets`：`default` / `all` / 自訂標的清單
- `--split`：`regime_aware` / `regime_matched` / `all_history`
- `--market-feature-set`：預設 `TW_plus_global_market`
- `--full-feature-set`：預設 `Global_plus_Trump_with_gate`
- `--feature-budget`：基礎特徵數上限
- `--signal-threshold`：策略信號門檻（影響 signal metrics）
- `--val-auc-weight`：選模時 AUC 權重（預設 `0.15`）
- `--metrics-output`：總表輸出檔名（在 `output/`）
- `--predictions-output`：預測輸出檔名（在 `output/`）

### Step3 候選池（最新版）

程式內建會測試下列 `step3_candidates`（可能依標的動態限制）：

- `event_gated_mlp` + `Global_plus_Trump_no_gate`（budget 80, threshold 0.55）
- `event_gated_mlp` + `Global_plus_Trump_no_gate`（budget 48, threshold 0.60）
- `small_mlp` + `Global_plus_Trump_no_gate`（budget 48, threshold 0.60）
- `logistic` + `Global_plus_Trump_no_gate`（budget 48/80, threshold 0.60/0.55）
- `elasticnet` + `Global_plus_Trump_no_gate`（budget 48/80, threshold 0.60/0.55）

`2454.TW` 會走保守候選族（偏 `elasticnet/logistic/small_mlp`）並加強泛化約束。

### 融合策略（`step3_moe_blend`）

融合用機率層做，不是硬投票：

- 事件日機率：
  - `p = alpha_event * p_step3 + (1 - alpha_event) * p_market`
- 非事件日機率：
  - `p = beta_non_event * p_step3 + (1 - beta_non_event) * p_market`

搜尋網格：

- `alpha_event`: `[0.0, 0.25, 0.5, 0.75, 1.0]`
- `beta_non_event`: `[0.0, 0.1, 0.2, 0.3]`

搜尋目標同樣是 `val_f1 + λ * val_auc`。

### 指令範例

1. 跑 3 檔（`src/config.TARGETS`）

```bash
./.venv/bin/python -m src.training.train_integration_compare \
  --targets default \
  --split regime_aware
```

2. 跑 11 檔全量

```bash
./.venv/bin/python -m src.training.train_integration_compare \
  --targets all \
  --split regime_aware
```

3. 自訂輸出檔名（建議每次實驗都命名）

```bash
./.venv/bin/python -m src.training.train_integration_compare \
  --targets all \
  --split regime_aware \
  --metrics-output integration_compare_metrics_step3_v7_all.csv \
  --predictions-output integration_compare_predictions_step3_v7_all.csv
```

### 輸出檔案說明

整合 runner 會在 `output/` 產生兩份主檔：

- `integration_compare_metrics_*.csv`
- `integration_compare_predictions_*.csv`

同時底層仍會更新 `outputs/` 目錄的模型與中間報告：

- `outputs/models/*.(pt|pkl)`
- `outputs/predictions/predictions_*.csv`
- `outputs/reports/metrics_*.json`

### `integration_compare_metrics_*.csv` 欄位解讀

- `target`：標的代號
- `config`：`market_only` / `trump_full` / `step3_integrated` / `step3_moe_blend`
- `model`：該列使用模型
- `feature_set`：特徵集名稱
- `feature_budget`：特徵上限（若是融合列可能為空）
- `signal_threshold`：信號閾值
- `selected_from`：選模/融合參數記錄（例如 alpha/beta）
- `val_macro_f1`：validation macro f1（僅模型列有）
- `macro_f1`：test macro f1（主要分類比較指標）
- `precision` / `recall` / `accuracy` / `auc`：test 分類指標
- `signal_coverage`：有訊號的比例
- `signal_hit_rate`：訊號命中率
- `signal_cumulative_return`：累積報酬（策略口徑）
- `signal_sharpe`：Sharpe ratio
- `signal_mdd`：最大回撤（越接近 0 越好）
- `signal_trades`：交易次數（信號筆數）

### 解讀建議（實務）

- 分類優先：先看 `macro_f1` 是否超過 `market_only`
- 交易次優先：再看 `signal_sharpe`、`signal_cumulative_return`、`signal_mdd`
- 若分類提升但 Sharpe 下降：代表交易頻率或門檻可能過激，需要再調 `signal_threshold`、`beta_non_event`
- 若單一標的退步：優先看 `selected_from`，確認是否是候選過度偏向某模型

### 與其他訓練腳本的關係

- `src/training/train.py`：單次訓練單模型（基礎訓練入口）
- `src/training/train_overlay.py`：Market + Overlay 策略研究
- `src/training/train_integration_compare.py`：把 baseline / trump_full / step3 / moe 放在同口徑比較

建議流程：

1. 用 `train_integration_compare.py` 做整體選模與比較。
2. 對落後標的再回到 `train.py` 單獨做深度調參。
3. 若要策略型收益強化，再進一步用 `train_overlay.py` 驗證。
