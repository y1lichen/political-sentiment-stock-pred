# Step 2：川普貼文 NLP 情感特徵萃取 (2017–2026)

本資料夾為本專案的**第二步驟**，負責處理 2017–2026 年間川普於各社群媒體平台的原始發文資料，透過自然語言處理 (NLP) 技術將每篇貼文轉換為可供機器學習模型訓練的量化情緒特徵向量。

**上游輸入**：Step 1 產生的台股量化數據（`data/taiwan_market_data/`）  
**本步驟輸出**：`trump_posts_features_2017_2026.csv`（70,730 筆帶有 NLP 特徵的發文資料）

---

## 📁 腳本說明

| 腳本名稱 | 說明 |
| :--- | :--- |
| `scrape_truth_social.py` | 爬取 Truth Social 平台上的川普發文（2021 年後），輸出 `trump_truth_social_posts.csv` |
| `merge_datasets.py` | 將 Twitter 歷史資料（`tweets.csv`）與 Truth Social 爬蟲結果合併，統一格式，輸出 `merged_trump_posts.csv` |
| `extract_nlp_features.py` | 對合併後的發文資料逐筆進行 NLP 特徵萃取，輸出最終的 `trump_posts_features_2017_2026.csv` |

---

## 🚀 如何執行

### 環境需求
```bash
pip install transformers torch pandas vaderSentiment nltk tqdm
```
> 執行前請先完成 NLTK VADER 字典的初次下載：
> ```python
> import nltk; nltk.download('vader_lexicon')
> ```

### 執行順序

**Step 2-1：爬取 Truth Social 資料**
```bash
python scrape_truth_social.py
```
- 輸出：`trump_truth_social_posts.csv`

**Step 2-2：合併資料集**
```bash
python merge_datasets.py
```
- 輸入：`tweets.csv`（Twitter 歷史，請自行取得）、`trump_truth_social_posts.csv`
- 輸出：`merged_trump_posts.csv`（共 70,730 筆）

**Step 2-3：執行 NLP 特徵萃取**
```bash
python extract_nlp_features.py
```
- 輸入：`merged_trump_posts.csv`
- 輸出：`trump_posts_features_2017_2026.csv`
- ⏱️ 預計執行時間：約 25–30 分鐘（使用 Apple Silicon MPS GPU 加速）或 40–60 分鐘（CPU）
- 🔁 支援**斷點續傳**：若中途中斷，重新執行會從上次存檔位置繼續

---

## 📊 輸出欄位說明（trump_posts_features_2017_2026.csv）

> ⚠️ 注意：此 CSV 因體積較大已加入 `.gitignore`，請向本步驟負責人取得，或自行重新執行腳本產生。

| 欄位 | 型態 | 說明 |
| :--- | :--- | :--- |
| `Timestamp` | DateTime (UTC) | 發文時間（UTC 時區，含時區資訊） |
| `Content` | String | 貼文原始文字 |
| `Likes` | Integer | 按讚數 |
| `Retweets` | Integer | 轉發數 |
| `Platform` | String | `Twitter_Legacy` 或 `Truth_Social` |
| `kw_china` | 0/1 | 是否提及 China / Chinese |
| `kw_taiwan` | 0/1 | 是否提及 Taiwan / Taiwanese |
| `kw_tariffs` | 0/1 | 是否提及 Tariff / Tax |
| `kw_sanctions` | 0/1 | 是否提及 Sanction |
| `kw_chips` | 0/1 | 是否提及 Chips / Semiconductor |
| `kw_tech` | 0/1 | 是否提及 Tech / Technology |
| `kw_ai` | 0/1 | 是否提及 AI / Artificial Intelligence |
| `vader_compound` | Float [-1, 1] | VADER 情感綜合分數 |
| `emotion_label` | String | RoBERTa 模型情緒分類（7 類） |
| `emotion_score` | Float [0, 1] | 情緒分類的置信度機率 |
| `weighted_vader` | Float | 按讚數加權後的 VADER 分數：`vader_compound × ln(1 + Likes)` |

**7 種情緒標籤 (emotion_label)**：`joy`, `anger`, `sadness`, `fear`, `surprise`, `disgust`, `neutral`

---

## 🧠 技術細節

### 核心情緒模型
- **模型**：[`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- 自動偵測並使用 GPU 加速（支援 CUDA / Apple Silicon MPS / CPU fallback）

### 互動加權情感分數
$$\text{weighted\_vader} = \text{vader\_compound} \times \ln(1 + \text{Likes})$$
透過對數縮放排除極端點讚值的影響，合理突顯高曝光貼文的情緒影響力。

---

## 🔗 與 Step 3 對接

後續步驟（特徵合併與預測建模）在讀取本步驟的輸出時，請注意：

1. **時區轉換**：`Timestamp` 為 UTC，需轉為台灣時間 `Asia/Taipei`（UTC+8）
2. **交易日遞延**：下午 13:30 後的發文及假日發文，需**往後遞延**至下一個台灣交易日
3. **讀取範例**：
```python
import pandas as pd

df = pd.read_csv('trump_posts_features_2017_2026.csv')
# 必須使用 format='mixed' 以正確解析混合毫秒格式的時間戳
df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, format='mixed')
df['Timestamp_CST'] = df['Timestamp'].dt.tz_convert('Asia/Taipei')
```
