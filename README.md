# Dual-Branch Feature Fusion Network（台股）

此專案實作一個 Dual-Branch Feature Fusion Network，用於預測下一個交易日的價格類別：
- 0 = 大跌
- 1 = 盤整/小幅變動
- 2 = 大漲

模型融合兩支分支：
1. 文本分支（社群貼文 / NLP 特徵，依交易日聚合）
2. 市場分支（過去 N 天的價格/量與衍生指標）

本 README 以中文說明整體資料處理、訓練、移動回測（walk-forward）與結果輸出位置。

---

**主要目錄（已整理）**
- `model/`：訓練腳本與實驗程式碼
	- `model/dual_branch_training.py`
- `data/`：原始與處理後的資料（金融資料、文本特徵）
- `output/`：訓練與評估輸出（混淆矩陣、預測檔、metrics）
	- `output/split_outputs/`：每個 split 的 `preds_*.csv`, `cm_*.csv`
	- `output/training_metrics_avg.csv`：每個標的的整體 Macro F1

---

## 快速開始

1. 建立虛擬環境並安裝套件：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r environment_setup/requirements_full.txt
```

2. 準備市場資料（使用 notebooks/stock_data.ipynb）

執行 `data/stock_data.ipynb` 來匯出：
- `data/taiwan_market_data/global_prices.csv`
- `data/taiwan_market_data/global_volumes.csv`
- `data/taiwan_market_data/institutional_investors.csv`
- `data/taiwan_market_data/margin_trading.csv`
- `data/taiwan_market_data/tx_futures_night.csv`

3. 準備文本特徵

將 NLP 特徵放於：
- `data/trump_nlp/trump_posts_features_2017_2026.csv`（或 `data/trump_post_data/...` 作為備援）

4. 執行訓練（整體 walk-forward）：

```bash
python model/dual_branch_training.py
```

訓練將自動：
- 執行 walk-forward 分割（expanding window）
- 每個 split 訓練模型並在 validation 上記錄 epoch-level 的 Macro F1
- 在每個 split 結束後輸出：
	- `output/split_outputs/preds_{ticker}_split{i}.csv`（該 split 的 y_true / y_pred）
	- `output/split_outputs/cm_{ticker}_split{i}.csv`（該 split 的混淆矩陣）
- 最後彙整每個標的的整體 Macro F1 至 `output/training_metrics_avg.csv`

---

## 評估指標與解讀

- 主要指標：**Macro F1**（處理類別不平衡）
	- Macro F1 對三類別分別計算 F1 後平均，能公平反映對少數類別（大漲/大跌）的辨識效果。
- 訓練輸出：
	- 每個 split 的混淆矩陣（`output/split_outputs/cm_*.csv`）方便計算 Precision/Recall/False-Alarm Rate（假警報率）。
	- `output/training_metrics_avg.csv`：每個標的的整體 Macro F1（跨所有 splits 的預測合併後計算），作為報告中的核心數字。

如何判斷模型是否有實務價值：
- 假警報率（對做空/做多訊號的 FP 比例）需配合交易成本、停利停損策略與夏普比率評估。
- 若 Macro F1 明顯高於基準（例如使用多數類 baseline 的 Macro F1），且混淆矩陣顯示 recall 對極端類別有合理改善，即具備實務價值。

---

## 檔案說明（重要輸出）
- `output/split_outputs/`:
	- `preds_{ticker}_split{i}.csv`：欄位 `y_true`, `y_pred`。
	- `cm_{ticker}_split{i}.csv`：混淆矩陣，index 為 `true_0/true_1/true_2`，columns 為 `pred_0/pred_1/pred_2`。
- `output/training_metrics_avg.csv`：欄位 `target`, `macro_avg_f1`。

進一步解讀與可複製的計算方法：

- `output/split_outputs/preds_{ticker}_split{i}.csv`：
	- 欄位：`y_true`（實際類別）、`y_pred`（模型預測）。每一列對應一個驗證或測試樣本（視腳本存檔時的資料集而定）。
	- 如何使用：可用此檔計算該 split 的各類別 Precision/Recall/F1、支援度（support）。範例（Pandas + scikit-learn）：

		```python
		import pandas as pd
		from sklearn.metrics import classification_report

		df = pd.read_csv('output/split_outputs/preds_2330.TW_split1.csv')
		print(classification_report(df['y_true'], df['y_pred'], digits=4))
		```

	- 注意：若某類別在該 split 中完全沒有樣本，`classification_report` 會以 `zero_division` 規則填 0 或顯示警告；Macro F1 在計算時通常會忽略該類別的影響（或以 0 處理），因此跨 split 彙整時要注意樣本數差異。

- `output/split_outputs/cm_{ticker}_split{i}.csv`（混淆矩陣）：
	- 格式：列為真實類別（`true_0, true_1, true_2`），欄為預測類別（`pred_0, pred_1, pred_2`）。
	- 範例矩陣：

		|        | pred_0 | pred_1 | pred_2 |
		|--------|--------|--------|--------|
		| true_0 |   50   |   10   |    5   |
		| true_1 |   20   |  300   |   30   |
		| true_2 |    8   |   25   |   60   |

		- 解讀：
			- Precision(class=2) = TP2 / (TP2 + FP2) = 60 / (5 + 30 + 60) = 60/95
			- Recall(class=2) = TP2 / (TP2 + FN2) = 60 / (8 + 25 + 60) = 60/93
			- Macro F1 = 平均三類別的 F1

	- 合併多個 split：若要得到整個實驗期間（所有 splits）的整體混淆矩陣，請把同一 `target` 的所有 `cm_{target}_split*.csv` 做 element-wise 相加，然後再依此矩陣計算 Precision/Recall/F1。這樣能反映不同時間段樣本量差異的總體表現。

- `output/training_metrics_avg.csv`：
	- 欄位說明：`target`（標的代碼）、`macro_avg_f1`（該標的跨所有 splits 合併後的 Macro F1）。
	- 計算方式：訓練腳本會將每個 split 的預測合併（concatenate）成一組長序列 y_true / y_pred，然後用 `sklearn.metrics.f1_score(..., average='macro')` 計算最終值。因此 `macro_avg_f1` 反映的是「把所有 split 的預測視為同一個集合」後的 Macro F1，而非對每個 split 的 Macro F1 做單純平均。

常見疑問與建議：

- 為什麼要同時提供 `preds_*.csv` 與 `cm_*.csv`？
	- `preds_*.csv` 可以讓你做更細緻的後續分析（例如：把預測對應到實際報酬率、計算經濟回測績效），而 `cm_*.csv` 是快速檢視分類錯誤型態的摘要（容易理解模型偏誤）。

- 如何評估「假警報率（False Alarm Rate）」？
	- 若你把 class=2 視為做多訊號，假警報率 = FP_for_class2 / (FP_for_class2 + TP_for_class2 + FN? )，通常更常見的是用 Precision(class=2) 來量化假警報佔預測做多之比率。對於做空（class=0）同理。

- 若要輸出每類別的 Precision/Recall/F1（跨所有 splits）：
	1. 將 `preds_{target}_split*.csv` 全部讀進並合併成一個 DataFrame。  
	2. 用 `classification_report` 或手動計算 TP/FP/FN（從合併後的混淆矩陣）得到每類別指標。  

範例：合併混淆矩陣並計算每類別指標（NumPy + Pandas）：

```python
import glob
import numpy as np
import pandas as pd

files = glob.glob('output/split_outputs/cm_2330.TW_split*.csv')
cms = [pd.read_csv(f, index_col=0).to_numpy() for f in files]
total_cm = np.sum(cms, axis=0)
tp = np.diag(total_cm)
fp = total_cm.sum(axis=0) - tp
fn = total_cm.sum(axis=1) - tp
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print(pd.DataFrame({'precision':precision,'recall':recall,'f1':f1}))
```

---

若需要，我可以把上述的彙整與指標計算腳本加入 `tools/` 或 `train/` 中，並自動產生 `output/confusion_summary_{target}.csv` 與 `output/precision_recall_f1_{target}.csv`。請回覆是否要我自動產出並放到 `output/` 下。

---

## 未來改進方向
- 將 Macro F1 拆成 per-class F1（顯示 class0/class2 的 trade-off）。
- 針對混淆矩陣計算 Precision/Recall/False Alarm（假警報率）並輸出。
- 將模型預測與真實報酬率做回測（加上交易成本），評估實際盈虧與風險指標（Sharpe, Max Drawdown）。

---

README 更新版本：2026-05-26
