# Step3 策略深度說明（`step3_integrated` 與 `step3_moe_blend`）

> 文件目的：
> 這份文件是給報告/簡報使用，重點用「淺白、可解釋」方式說明我們在 `Integration Compare`（`src/training/train_integration_compare.py` 最新版）新增的策略設計、選模邏輯與輸出解讀。

---

## 1. 為什麼要做這兩個策略？

我們原本有兩條主線：

- `market_only`：只用市場特徵（穩，但可能忽略 Trump 訊號）
- `trump_full`：市場 + Trump 特徵（資訊更完整，但不一定穩定）

實驗中發現：

- 有些標的，Trump 訊號確實有幫助
- 但有些標的，直接全量吃 Trump 特徵會造成雜訊

所以我們新增兩個中間策略：

1. `step3_integrated`：先「挑出最適合這檔股票」的 Trump 版本模型
2. `step3_moe_blend`：再把 `market_only` 與 `step3_integrated` 做機率融合，避免單一模型失誤

一句話：

`step3_integrated` 解決「選對模型」，`step3_moe_blend` 解決「穩定落地」。

---

## 2. 整體流程（簡報版）

每個標的（ticker）都走同一流程：

1. 先訓練固定兩條基準線
- `market_only`（`TW_plus_global_market`）
- `trump_full`（`Global_plus_Trump_with_gate`）

2. 再跑 Step3 候選池
- 多組模型 + 特徵預算 + 門檻

3. 在 Validation 上選出最佳 `step3_integrated`
- 使用目標函數（見第 5 節）

4. 用 `market_only` + `step3_integrated` 做融合
- 事件日用 `alpha_event`
- 非事件日用 `beta_non_event`
- 在 Validation 搜尋最好的 alpha/beta

5. 產出四組可比較結果
- `market_only`
- `trump_full`
- `step3_integrated`
- `step3_moe_blend`

---

## 3. 我們怎麼選特徵？

### 3.1 特徵集角色

在 `train_integration_compare.py` 中，主要用三種特徵集語意：

- `TW_plus_global_market`：市場基準（market baseline）
- `Global_plus_Trump_with_gate`：完整版 Trump 管線
- `Global_plus_Trump_no_gate`：Step3 候選主要使用（強調可控比較）

### 3.2 Step3 的特徵設定

`step3_integrated` 候選大多使用：

- `Global_plus_Trump_no_gate`
- `feature_budget` = 48 或 80（控制維度、抑制雜訊）

目的：

- 避免特徵過多導致過擬合
- 保留 Trump 訊號的同時維持模型可泛化

### 3.3 為什麼要有 `feature_budget`

在金融資料上，高維特徵常造成：

- 訓練表現很好，實測崩掉
- 模型對市場 regime 轉換不穩

`feature_budget` 是最實際的降噪手段之一：

- `80`：資訊較完整
- `48`：更保守、較抗噪

---

## 4. 我們怎麼選模型？

`step3_candidates` 目前包含：

- `event_gated_mlp`
- `small_mlp`
- `logistic`
- `elasticnet`

都搭配 `Global_plus_Trump_no_gate` 與不同 budget/threshold 組合。

### 4.1 為什麼不是只用深度模型？

實務上，某些標的（尤其樣本噪訊高）反而線性/稀疏模型更穩：

- `logistic`：簡單、可解釋
- `elasticnet`：可做特徵稀疏化，抗共線性

### 4.2 2454 的特別處理

`2454.TW` 在程式中有 target-specific 保守策略：

- 候選限制成 `elasticnet/logistic/small_mlp`
- 選模分數加入過擬合懲罰（train-val gap penalty）

目的：

- 避免 validation 一時好看但 test 不穩

---

## 5. 核心選模邏輯（最重要）

### 5.1 `step3_integrated` 選模分數

一般標的：

`score = val_f1 + λ * val_auc`

- `val_f1`：分類平衡能力（主目標）
- `val_auc`：機率排序能力（輔助目標）
- `λ`：`--val-auc-weight`（目前預設 0.15）

2454：

`score = val_f1 + (λ + 0.10) * val_auc - 0.50 * overfit_gap`

- `overfit_gap = max(0, train_f1 - val_f1)`

意義：

- 如果 train 明顯比 val 好，代表過擬合，會被扣分

### 5.2 `step3_moe_blend` 融合分數

融合前先定義機率：

- 事件日：
  `p = alpha_event * p_step3 + (1 - alpha_event) * p_market`
- 非事件日：
  `p = beta_non_event * p_step3 + (1 - beta_non_event) * p_market`

網格搜尋：

- `alpha_event ∈ [0.0, 0.25, 0.5, 0.75, 1.0]`
- `beta_non_event ∈ [0.0, 0.1, 0.2, 0.3]`

同樣在 validation 用 `val_f1 + λ*val_auc` 挑最佳權重。

---

## 6. 這兩個策略的差異（簡報可直接貼）

### `step3_integrated`

定位：

- 「模型挑選器」

做法：

- 在候選池中挑出單一最佳模型/特徵設定

優點：

- 清楚、可解釋
- 容易知道哪種模型在該標的有效

風險：

- 單一模型仍可能對 regime 切換敏感

### `step3_moe_blend`

定位：

- 「穩定器」

做法：

- 用 market baseline 當底座
- 讓 step3 在事件日/非事件日用不同權重參與

優點：

- 通常比直接換模型更平滑
- 能減少單模型方向偏誤

風險：

- 參數更多（alpha/beta），需要驗證資料挑選

---

## 7. 輸出檔案與欄位怎麼看

主輸出（在 `output/`）：

- `integration_compare_metrics_step3_v7*.csv`
- `integration_compare_predictions_step3_v7*.csv`

中間產物（在 `outputs/`）：

- 模型：`outputs/models/*.(pt|pkl)`
- 預測：`outputs/predictions/predictions_*.csv`
- 指標：`outputs/reports/metrics_*.json`

### 指標表重點欄位

- `config`：是哪一個策略
- `macro_f1`：分類主指標（越高越好）
- `auc`：機率排序能力
- `signal_sharpe`：風險調整後報酬
- `signal_cumulative_return`：累積報酬
- `signal_mdd`：最大回撤（越接近 0 越好）
- `selected_from`：選模/融合參數紀錄（很關鍵）

---

## 8. 如何報告成果（建議口徑）

建議順序：

1. 先報 `macro_f1`：
- 幾檔超過 `market_only` baseline

2. 再報交易指標：
- Sharpe / Cumulative Return / MDD 是否同步提升

3. 強調 trade-off：
- 分類提升不一定代表財金績效等比例提升

4. 針對落後標的提出下一步：
- 做標的專屬候選池、門檻或權重限制

---

## 9. 簡報可用的一句話結論

> `step3_integrated` 讓我們能「為每檔股票挑到更適配的 Trump 模型」，
> `step3_moe_blend` 讓我們能「在事件日與非事件日之間做風險控制式融合」，
> 最終在多數標的上把分類指標拉高，同時保留策略層面可調整空間。

---

## 10. 重跑指令（最新版）

```bash
# 3 檔
./.venv/bin/python -m src.training.train_integration_compare \
  --targets default \
  --split regime_aware

# 11 檔
./.venv/bin/python -m src.training.train_integration_compare \
  --targets all \
  --split regime_aware

# 指定輸出檔名
./.venv/bin/python -m src.training.train_integration_compare \
  --targets all \
  --split regime_aware \
  --metrics-output integration_compare_metrics_step3_v7_all.csv \
  --predictions-output integration_compare_predictions_step3_v7_all.csv
```

