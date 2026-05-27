# 川普貼文 NLP 特徵分析資料集 (2017-2026) — README

本資料集收錄了唐納·川普 (Donald Trump) 自 **2017 年 1 月 1 日至 2026 年 4 月 30 日** 期間在 **Twitter/X** 及 **Truth Social** 上發表的所有貼文（共 **70,730 筆**），並整合了高精度的自然語言處理 (NLP)、情感分析、與特定經濟/科技主題的關鍵字標記。

本資料集特別適合用於量化交易、金融市場時間序列預測（如預測台股、美股波動）以及政治輿情分析。

---

## 📂 檔案基本資訊
- **檔名**：`trump_posts_features_2017_2026.csv`
- **資料期間**：2017-01-01 00:00:00 UTC ~ 2026-04-30 23:59:59 UTC
- **總筆數**：70,730 筆
- **編碼格式**：UTF-8 (以雙引號 `"` 包含包含換行符號的文字)

---

## 📊 欄位詳細說明 (Schema)

本 CSV 檔案包含 **16 個欄位**，具體欄位定義如下：

| 欄位名稱 | 資料類型 | 說明 | 範例與數值範圍 |
| :--- | :--- | :--- | :--- |
| **`Timestamp`** | DateTime | 貼文發佈時間 (預設為 **UTC** 時區) | `2017-02-04 03:07:47+00:00` |
| **`Content`** | String | 貼文的原始文字內容 | `"MAKE AMERICA GREAT AGAIN!"` |
| **`Likes`** | Integer | 該篇貼文獲得的按讚數 (Like / Heart) | `132817` |
| **`Retweets`** | Integer | 該篇貼文獲得的轉發/轉貼數 (Retweet / Re-truth) | `21891` |
| **`Platform`** | String | 貼文發布平台來源 | `Twitter_Legacy` (推特歷史資料)<br>`Truth_Social` (Truth Social 貼文) |
| **`kw_china`** | Binary (0/1) | 貼文是否提及「中國」關鍵字 (China, Chinese) | `1` 或 `0` |
| **`kw_taiwan`** | Binary (0/1) | 貼文是否提及「台灣」關鍵字 (Taiwan, Taiwanese) | `1` 或 `0` |
| **`kw_tariffs`** | Binary (0/1) | 貼文是否提及「關稅」關鍵字 (Tariff, Tariffs, Tax) | `1` 或 `0` |
| **`kw_sanctions`** | Binary (0/1) | 貼文是否提及「制裁」關鍵字 (Sanction, Sanctions) | `1` 或 `0` |
| **`kw_chips`** | Binary (0/1) | 貼文是否提及「晶片」關鍵字 (Chips, Semiconductor, Semiconductors) | `1` 或 `0` |
| **`kw_tech`** | Binary (0/1) | 貼文是否提及「科技」關鍵字 (Tech, Technology) | `1` 或 `0` |
| **`kw_ai`** | Binary (0/1) | 貼文是否提及「人工智慧」關鍵字 (AI, Artificial Intelligence) | `1` 或 `0` |
| **`vader_compound`** | Float | 採用 VADER 情感分析器計算出的**綜合情緒分數** | `-1.0` (極度悲觀) 至 `1.0` (極度樂觀)<br>`0.0` 表示中立 |
| **`emotion_label`** | String | 基於 RoBERTa 深度學習模型預測的 **7 大核心情緒分類標籤** | `neutral` (中立), `joy` (喜悅), `anger` (憤怒),<br>`sadness` (悲傷), `fear` (恐懼), `surprise` (驚訝), `disgust` (厭惡) |
| **`emotion_score`** | Float | 深度學習模型對該情緒標籤的**預測機率/置信度** | `0.0` 至 `1.0` |
| **`weighted_vader`** | Float | **互動量加權情感得分**。使用按讚數 (`Likes`) 進行對數縮放加權，用以突顯高曝光度貼文的影響力 | 公式：`vader_compound * ln(1 + Likes)` |

---

## 🧠 NLP 技術實作說明

### 1. 核心情緒分類 (Emotion Classification)
使用 Hugging Face Transformer 著名的英文情緒分析模型：
- **模型**：[`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- **硬體**：使用 Apple Silicon GPU (`mps` 裝置) 進行硬體加速推理。
- **作用**：比傳統的「正/負面」分析更精細，能有效抓出川普是否處於「憤怒 (anger)」或「恐懼 (fear)」等對市場波動影響顯著的情緒。

### 2. 互動加權分數 (Weighted Sentiment)
若僅使用傳統情感分數，一萬次點讚與一百萬次點讚的推文權重相同，這不符合市場現實。因此引入了：
$$\text{weighted\_vader} = \text{vader\_compound} \times \ln(1 + \text{Likes})$$
利用對數縮放 (Log scaling) 避免極端點讚數過度放大特徵，同時能合理突顯「熱門貼文」的情緒渲染力。

---

## 📈 推薦後續對接使用方式 (以台股對齊為例)

若您是接續做金融預測（例如台股時間序列預測）的同學，建議採用以下對齊邏輯：

1. **時區轉換**：將 `Timestamp` (UTC) 轉為台灣時間 `Asia/Taipei` (UTC+8)。
2. **交易日遞延滾動 (Time-shifting)**：
   - 台灣股市交易時間為 **09:00 ~ 13:30**。
   - 如果川普發文時間大於 **13:30**（台灣時間），該貼文的情緒影響應**遞延至下一交易日**結算。
   - 週末及國定假日的發文同樣**遞延至下一開盤日**。
3. **特徵聚合 (Aggregation)**：
   - 如果同一交易日有多篇發文，建議：
     - 情感分數 (`vader_compound`, `weighted_vader`) 取 **平均值 (Mean)**。
     - 關鍵字 Flag (`kw_china` 等) 與情緒數量 (`emotion_anger` 計數) 取 **總和 (Sum)**。
     - 每日推文總數 (`post_count`) 作為波動率特徵。

### Python 快速讀取與時區處理範例：
```python
import pandas as pd

# 1. 讀取資料 (mixed 格式確保毫秒解析無誤)
df = pd.read_csv('trump_posts_features_2017_2026.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, format='mixed')

# 2. 轉換為台灣時間
df['Timestamp_CST'] = df['Timestamp'].dt.tz_convert('Asia/Taipei')

# 3. 提取日期與時間
df['CST_Date'] = df['Timestamp_CST'].dt.date
df['CST_Time'] = df['Timestamp_CST'].dt.time

print(df[['Timestamp_CST', 'Content', 'weighted_vader']].head())
```

---
*如有任何欄位擴充需求（例如新增其他關鍵字或情緒特徵），可直接參考目錄下的 `extract_nlp_features.py` 修改關鍵字清單並重新執行推理。*
