# Evaluation 說明

本目錄負責把訓練輸出的分類預測轉成統一評估表，包含：

- ML metrics：分類預測品質。
- Financial metrics：把模型預測轉成交易訊號後的回測績效。
- Baselines：B1 Buy & Hold、B2 Pure Market、B3 SMA 5/20。

主要入口：

```bash
python -m evaluation.training_outputs \
  --predictions output/training_predictions.csv \
  --prices-csv data/taiwan_market_data/global_prices.csv \
  --out-summary output/evaluation_summary.csv \
  --out-cm-dir output/confusion_matrices
```

若直接執行：

```bash
python train.py
```

訓練結束後也會自動呼叫上述 evaluation 流程。

## Label 定義

模型是三分類任務：

| Label | 意義 |
| --- | --- |
| `0` | 大跌 |
| `1` | 盤整 |
| `2` | 大漲 |

交易策略目前採 long-only 設定：

```text
pred_label = 2 -> position = 1, 持有股票
pred_label = 0 或 1 -> position = 0, 空手
```

也就是只有模型預測「大漲」時才進場。

## ML Metrics

### Confusion Matrix

Confusion matrix 用來統計真實類別與預測類別的對應關係。

公式：

```text
C[i, j] = count(y_true = i and y_pred = j)
```

其中：

- row 代表真實類別 `y_true`。
- column 代表預測類別 `y_pred`。
- 對角線 `C[k, k]` 是預測正確的數量。
- 非對角線是分類錯誤。

本專案額外標記 fatal error：

```text
true = 大跌(0), pred = 大漲(2)
```

這代表模型在實際大跌時發出買進訊號，是 long-only 策略中最危險的錯誤。

輸出位置：

```text
output/confusion_matrices/cm_{target}_{model_type}.csv
output/confusion_matrices/cm_{target}_{model_type}.png
```

### Precision

Precision 衡量「模型預測為某一類時，有多少是真的」。

對類別 `k`：

```text
Precision_k = TP_k / (TP_k + FP_k)
```

用 confusion matrix 表示：

```text
TP_k = C[k, k]
FP_k = sum_i C[i, k] - C[k, k]
Precision_k = C[k, k] / sum_i C[i, k]
```

本專案輸出三個 precision：

| 欄位 | 意義 |
| --- | --- |
| `precision_down` | 預測大跌時，有多少真的大跌 |
| `precision_flat` | 預測盤整時，有多少真的盤整 |
| `precision_up` | 預測大漲時，有多少真的大漲 |

其中 `precision_up` 對交易最重要，因為它直接對應買進訊號的可靠度。

### Recall

目前 `evaluation_summary.csv` 不輸出 recall，但 confusion matrix 可計算。

Recall 衡量「真實為某一類時，模型抓到多少」。

對類別 `k`：

```text
Recall_k = TP_k / (TP_k + FN_k)
```

用 confusion matrix 表示：

```text
FN_k = sum_j C[k, j] - C[k, k]
Recall_k = C[k, k] / sum_j C[k, j]
```

Recall 可用來看模型是否漏掉大漲或大跌事件。

### F1 Score

F1 同時考慮 precision 和 recall。

對類別 `k`：

```text
F1_k = 2 * Precision_k * Recall_k / (Precision_k + Recall_k)
```

當 precision 或 recall 任一方很低時，F1 也會低。

### Macro F1

Macro F1 是三個類別 F1 的平均。

公式：

```text
Macro F1 = (F1_down + F1_flat + F1_up) / 3
```

意義：

- 適合類別不平衡的任務。
- 不會讓樣本數最多的「盤整」類別完全主導結果。
- 若模型只會預測盤整，Macro F1 通常不會太高。

輸出欄位：

```text
macro_f1
```

## Financial Metrics

Financial metrics 會把模型分類預測轉成交易訊號，再用收盤價做回測。

### Daily Return

股票第 `t` 天到第 `t+1` 天的報酬率：

```text
r_stock,t = close[t+1] / close[t] - 1
```

策略毛報酬：

```text
r_strategy,t = position_t * r_stock,t
```

交易成本：

| 動作 | 成本 |
| --- | --- |
| 買進 `0 -> 1` | `fee_buy = 0.001425` |
| 賣出 `1 -> 0` | `fee_sell + tax_sell = 0.001425 + 0.003` |

策略淨報酬：

```text
r_net,t = r_strategy,t - transaction_cost_t
```

### Cumulative Return

累積報酬率衡量整段回測期間總獲利。

資產曲線：

```text
Equity_t = Equity_{t-1} * (1 + r_net,t)
```

累積報酬率：

```text
Cumulative Return = Equity_T / Equity_0 - 1
```

輸出欄位：

```text
cumulative_return
```

解讀：

- `0.20` 代表總報酬 `+20%`。
- `-0.10` 代表總虧損 `-10%`。

### Sharpe Ratio

Sharpe Ratio 衡量每承擔一單位波動風險，能得到多少報酬。

目前假設 risk-free rate 為 0，並用 252 個交易日年化：

```text
Sharpe = mean(r_net) / std(r_net) * sqrt(252)
```

輸出欄位：

```text
sharpe
```

解讀：

- 越高代表風險調整後報酬越好。
- 若策略幾乎都空手，且 daily returns 全為 0，Sharpe 會回傳 `0.0`。

### Maximum Drawdown

Maximum Drawdown 衡量資產曲線從歷史高點到後續低點的最大跌幅。

每一天的 running peak：

```text
Peak_t = max(Equity_0, Equity_1, ..., Equity_t)
```

每一天的 drawdown：

```text
Drawdown_t = (Peak_t - Equity_t) / Peak_t
```

最大回撤：

```text
MDD = max_t Drawdown_t
```

輸出欄位：

```text
max_drawdown
```

解讀：

- `0.30` 代表最嚴重時從高點下跌 `30%`。
- 越低越好。

### Number of Trades

交易次數統計 position 改變的次數。

```text
n_trades = count(position_t != position_{t-1})
```

其中：

- `0 -> 1` 算一次買進。
- `1 -> 0` 算一次賣出。

輸出欄位：

```text
n_trades
```

## Baselines

### Full Model

完整雙分支模型：

```text
market features + Trump text/sentiment features
```

這是主要模型。

### B2 Pure Market

只使用市場資料，不使用川普文本特徵。

在訓練流程中對應：

```text
zero_text = True
```

也就是 text branch 輸入全設為 0，只保留 market branch。

用途：

- 檢查 Trump 文本/情緒特徵是否真的有增益。
- 若 Full Model 顯著優於 B2，代表文本特徵有幫助。

### B1 Buy & Hold

買入並持有策略：

```text
第一天買進，持有到回測期最後一天
```

目前採 mark-to-market：

- 第一天扣買進手續費。
- 最後一天不強制賣出，因此不扣賣出手續費與交易稅。

B1 沒有分類預測，所以 ML metrics 為 `N/A`。

### B3 SMA 5/20

傳統均線策略：

```text
SMA_5 > SMA_20 -> position = 1
SMA_5 <= SMA_20 -> position = 0
```

其中：

```text
SMA_n,t = mean(close[t-n+1], ..., close[t])
```

前 `slow - 1` 天因為 `SMA_20` 尚未形成，預設空手。

B3 也沒有分類預測，所以 ML metrics 為 `N/A`。

## Output 檔案

### `output/training_predictions.csv`

訓練後的逐日預測結果。

欄位：

| 欄位 | 意義 |
| --- | --- |
| `target` | 股票代號 |
| `split` | walk-forward split 編號 |
| `date` | 預測日期 |
| `y_true` | 真實 label |
| `pred_label` | 模型預測 label |
| `proba_down` | 預測大跌機率 |
| `proba_flat` | 預測盤整機率 |
| `proba_up` | 預測大漲機率 |
| `model_type` | `full_model` 或 `pure_market` |

### `output/evaluation_summary.csv`

最重要的總表。

欄位：

```text
target, model, macro_f1, precision_down, precision_flat, precision_up,
cumulative_return, sharpe, max_drawdown, n_trades
```

每個 target 會有四列：

```text
Full Model
B2 Pure Market
B1 Buy & Hold
B3 SMA 5/20
```

注意：

- `Full Model` 和 `B2 Pure Market` 有 ML metrics 與 financial metrics。
- `B1 Buy & Hold` 和 `B3 SMA 5/20` 只有 financial metrics，ML metrics 會是 `N/A`。

### `output/confusion_matrices/`

每個 target/model 的混淆矩陣。

```text
cm_{target}_{model_type}.csv
cm_{target}_{model_type}.png
```

## 常用指令

完整重跑訓練與評估：

```bash
conda activate dla
python train.py
```

只重跑 evaluation，不重新訓練：

```bash
conda activate dla
python -m evaluation.training_outputs \
  --predictions output/training_predictions.csv \
  --prices-csv data/taiwan_market_data/global_prices.csv \
  --out-summary output/evaluation_summary.csv \
  --out-cm-dir output/confusion_matrices
```

檢查 B1/B3 是否已輸出：

```bash
rg -n "B1 Buy|B3 SMA" output/evaluation_summary.csv
```

檢查中文 confusion matrix 繪圖：

```bash
python -m evaluation.ml_metrics
```
