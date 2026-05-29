# 深度學習版 Trump Code 台股預測專案開發計劃書

## 1. 專案定位

本專案目標是打造一個「深度學習版 Trump Code」，用川普社群貼文與台灣市場資料預測台股重要標的的短期價格反應。原始 `sstklen/trump-code` 以川普 Truth Social / X 貼文為事件來源，將貼文行為轉成每日事件特徵，透過大量規則組合暴力搜尋、回測、驗證與每日閉環學習，找出可能具有預測力的交易訊號。本專案保留這套研究邏輯，但將核心從「人工規則 + brute-force search」改為「深度學習序列模型 + 嚴格 walk-forward 評估」。

本專案使用的主要資料如下：

- 文字資料：`data/text/trump_posts_features_2017_2026.csv`
- 台灣市場資料：`data/taiwan_market_data/`
- 預測標的：優先鎖定 `2330.TW`、`2454.TW`、`0050.TW`，可延伸至 `00632R.TW`、半導體與電子權值股。

本專案應定位為課堂研究與模型比較，不作為投資建議。

## 2. 原始 trump-code 的核心邏輯摘要

依據 `sstklen/trump-code` README 與核心程式設計，原專案可拆成四個層次。

### 2.1 貼文事件化

原專案不直接把貼文全文丟進交易規則，而是先萃取每日二元或統計特徵，例如：

- 發文量：`posts_high`、`posts_low`、`volume_spike`
- 主題事件：`has_tariff`、`has_deal`、`has_relief`、`has_china`
- 發文時段：`pre_tariff`、`open_tariff`、`has_night_post`
- 語氣與形式：`high_emotion`、`lots_of_excl`、`sig_djt`、`sig_potus`
- 組合語意：`deal_over_tariff`、`tariff_only`、`tariff_rising`

這些特徵被聚合到日層級，再和市場資料對齊。

### 2.2 規則暴力搜尋

原專案的 `analysis_11_brute_force.py` 會列舉 2、3、4 個條件組合，對每組條件測試：

- 交易方向：`LONG` / `SHORT`
- 持有天數：1 / 2 / 3 天
- 觸發條件：所有條件同時成立才進場
- 訓練期門檻：至少 10 筆交易、勝率 >= 60%、平均報酬 > 0.1%
- 統計檢定：二項檢定 p-value
- 驗證期門檻：至少 5 筆交易、勝率 >= 60%、平均報酬 > 0.1%

這是典型的可解釋規則探勘：優點是透明，缺點是容易資料探勘偏誤、只能捕捉離散條件，且對非線性時序關係表達有限。

### 2.3 市場反應分析與回測

原專案會分析發文後的市場反應，例如同日、隔日、隔夜跳空、持有數日報酬，並區分關稅、中國、股市、發文量、情緒強度等事件類型。其重點不是單篇貼文分類，而是「貼文事件序列如何對應市場報酬」。

### 2.4 閉環學習與風控

README 描述的每日管線是：

`Fetch -> Analyze -> Run rules -> Predict -> Verify -> Circuit Breaker -> Learn -> Evolve -> Briefing`

也就是每天產生預測、驗證前一日預測、根據表現保留或淘汰規則，並以 circuit breaker 防止系統在近期退化時繼續輸出高信心訊號。

## 3. 深度學習版的對應設計

本專案不複製原專案的 brute-force 規則搜尋，而是保留其事件研究框架，改用深度學習模型學習「文字事件 + 市場狀態 -> 未來報酬」的映射。

| trump-code 做法 | 本專案深度學習對應 |
| --- | --- |
| 將貼文轉成每日二元特徵 | 保留每日聚合特徵，並加入連續 NLP 特徵與可選文字 embedding |
| 暴力搜尋 2-4 條件規則 | 用低容量深度模型學習非線性、多變數交互作用，並保留事件觸發門檻 |
| LONG / SHORT + 持有 1-3 天 | 多任務輸出：方向分類、報酬回歸、波動分類 |
| train/test 規則驗證 | walk-forward validation 與時間序列切分 |
| 勝率、平均報酬、p-value | 加入 MAE/RMSE、方向準確率、AUC、Sharpe、max drawdown |
| 每日 predict -> verify -> learn | 建立 daily inference log、訊號拒絕機制與週期性再訓練 |
| circuit breaker | 用近期 out-of-sample 表現、校準誤差與回撤門檻降低或停用訊號 |
| 規則排行榜 | 模型排行榜、特徵重要性、attention/SHAP 解釋 |

## 4. 資料盤點

### 4.1 文字資料

`data/text/trump_posts_features_2017_2026.csv` 目前包含約 70,730 筆貼文，時間範圍約為 2017-01-01 至 2026-03-25 UTC。重要欄位包含：

- 時間與平台：`Timestamp`、`Platform`
- 原文與互動：`Content`、`Likes`、`Retweets`
- 主題關鍵字：`kw_china`、`kw_taiwan`、`kw_tariffs`、`kw_chips`、`kw_tech`、`kw_ai`
- Trump Code 風格事件特徵：`tc_tariff`、`tc_deal`、`tc_relief`、`tc_action`、`tc_attack`、`tc_positive`、`tc_market_brag`
- 文體與強度：`caps_ratio`、`uppercase_char_ratio`、`exclamation_count`、`emotional_intensity`
- 時段特徵：`post_hour_tw`、`post_hour_et`、`is_pre_market`、`is_market_hours`、`is_after_market`、`is_night_post`
- 情緒特徵：`vader_compound`、`weighted_vader`、`emotion_label`、`emotion_score`

這份資料已經高度接近原 `trump-code` 的手工事件化結果，因此第一階段可不重新做 NLP，只做交易日對齊與日層級聚合。

### 4.2 台灣市場資料

`data/taiwan_market_data/` 包含：

- `global_prices.csv`：台股標的、美股指數、ADR、匯率、VIX、利率的每日價格
- `global_volumes.csv`：對應成交量
- `institutional_investors.csv`：三大法人買賣超
- `margin_trading.csv`：融資融券餘額
- `tx_futures_night.csv`：台指期夜盤價量與 `spread_per`

台灣市場資料可分為三類：

- 預測目標：`2330.TW`、`2454.TW`、`0050.TW` 的隔日開盤、收盤、報酬方向
- 控制變數：`TSM`、`^SOX`、`^NDX`、`^GSPC`、`^VIX`、`TWD=X`、`^TNX`
- 本地市場狀態：三大法人、融資融券、台指期夜盤

## 5. 資料對齊策略

台股交易時段與美國政治貼文存在時區差異，因此資料對齊是專案成敗關鍵。

### 5.1 時區轉換

- 將 `Timestamp` 從 UTC 轉為 `Asia/Taipei`
- 保留 `post_hour_tw` 與 `post_hour_et`
- 將貼文分成台股盤前、盤中、盤後、美股交易時段與夜間事件

### 5.2 事件歸屬交易日

建議建立 `effective_trade_date`：

- 台灣時間 09:00 前貼文：歸入當日台股可反應資訊
- 台灣時間 09:00-13:30 貼文：可建立盤中事件特徵，但若只預測開盤，應歸入下一交易日
- 台灣時間 13:30 後貼文：歸入下一個台股交易日
- 週末與台股休市日：順延至下一個交易日

若預測目標是「隔日收盤報酬」，可使用貼文 `effective_trade_date` 對齊到該交易日開盤前可觀測資訊，避免未來資訊洩漏。

### 5.3 市場特徵滯後

所有市場控制變數必須只使用預測當下已知資訊：

- 台股昨日收盤價、成交量與技術指標
- 前一美股交易日的 `TSM`、`^SOX`、`^NDX`、`^GSPC`、`^VIX`
- 台指期夜盤 `spread_per`，需確認其時間是否在預測目標前已可得
- 三大法人與融資融券通常盤後公布，預測隔日可用，預測當日開盤不可用

## 6. 預測任務定義

第一階段建議定義三個任務，形成多任務學習。

### 6.1 方向分類

預測標的 `t` 在未來 horizon 的報酬方向：

- `y_direction = 1`：未來報酬 > 0
- `y_direction = 0`：未來報酬 <= 0

可分別建立：

- `close_to_close_1d`
- `open_to_close_1d`
- `open_gap`
- `close_to_close_3d`

### 6.2 報酬回歸

預測連續報酬率：

- `r_1d = (close_t - close_t-1) / close_t-1`
- `gap = (open_t - close_t-1) / close_t-1`
- `intraday = (close_t - open_t) / open_t`

### 6.3 大波動分類

模仿原專案的 `big moves` 思路，預測是否出現顯著波動：

- `abs(return) > rolling_volatility_60d`
- 或以前 20% / 後 20% 分位數定義大漲與大跌

## 7. 特徵工程設計

### 7.1 Trump Code 風格日聚合特徵

對每個 `effective_trade_date` 聚合：

- `post_count`
- `platform_count_truth_social`、`platform_count_x`
- `sum_tc_tariff`、`sum_tc_deal`、`sum_tc_relief`、`sum_tc_action`
- `has_tariff`、`has_china`、`has_taiwan`、`has_chips`
- `avg_vader_compound`、`sum_weighted_vader`
- `emotion_anger_count`、`emotion_fear_count`、`emotion_joy_count`
- `max_emotional_intensity`、`avg_emotional_intensity`
- `pre_market_tariff_count`
- `after_market_tariff_count`
- `night_post_count`
- `deal_over_tariff_daily`
- `tariff_only_daily`
- `relief_positive_daily`

這一層直接對應原 `trump-code` 的每日事件矩陣。

### 7.2 序列市場特徵

對每個標的與控制變數建立滯後特徵：

- 過去 1、3、5、10、20 日報酬
- 過去 5、20 日波動率
- 成交量變化率
- `^VIX` 變化
- `TWD=X` 變化
- `TSM` 與 `2330.TW` 的 ADR/local lead-lag spread
- `^SOX` 與台灣半導體股的 lead-lag 關係

### 7.3 法人與信用交易特徵

- 三大法人各自 `net_buy = buy - sell`
- 外資買賣超的 1、3、5 日累積
- 融資餘額變化率
- 融券餘額變化率
- 融資融券比

### 7.4 政權狀態與市場 regime 特徵

川普貼文對市場的影響力不只取決於文字內容，也取決於他是否具有政策執行權，以及當時市場是否處於特殊危機環境。因此模型需加入 regime 特徵，避免把不同時期混為同一種資料生成機制。

建議加入下列政權狀態特徵：

- `is_president`：是否為川普總統任內。
- `first_term`：2017-01-20 至 2021-01-20。
- `post_presidency`：2021-01-21 至 2025-01-19。
- `second_term`：2025-01-20 之後。
- `campaign_period`：選舉或高強度競選期間。
- `policy_power_score`：政策實權分數，例如總統任內為 1，非總統期為 0，競選高峰可設為 0.3-0.5。
- `tariff_regime_intensity`：關稅政策環境強度，可由 `tc_tariff` 滾動次數、重大政策日期或人工標註事件建立。

建議加入下列 COVID / 危機市場特徵：

- `covid_crash_period`：2020-02-01 至 2020-04-30，代表疫情初期全球市場急跌與流動性衝擊。
- `covid_recovery_liquidity_period`：2020-05-01 至 2021-12-31，代表寬鬆貨幣政策與科技股估值擴張期間。
- `covid_policy_period`：2020-02-01 至 2021-12-31，作為較寬鬆的疫情總體 regime。
- `high_vix_regime`：例如 `^VIX` 高於過去 3 年 80 分位數。
- `market_stress_score`：由 `^VIX`、`^SOX` 跌幅、`^GSPC` 跌幅、成交量異常共同組成。

這些特徵的目的不是讓模型「預測疫情」，而是控制疫情期間市場由總體危機主導的特殊狀態。若不納入，模型可能錯把 2020 年的極端波動歸因於川普貼文。

## 8. 模型架構

本專案不應把大型 Transformer 視為預設答案。2017-2026 台股交易日約 2,200 天，扣除非總統任期、COVID regime 與無明確事件日後，真正可用於學習「川普政策衝擊」的樣本可能只有數百天。這是典型的低信噪比、小樣本金融時序問題。模型設計必須遵守「低容量、強正則、事件觸發、可拒絕交易」原則。

### 8.1 Baseline 模型

先建立非深度學習 baseline，作為評估基準：

- Zero-return / previous direction baseline
- Logistic Regression
- L1 / Elastic Net Logistic Regression
- Random Forest 或 XGBoost / LightGBM
- 原 Trump Code 風格規則 baseline：用 `tc_*` 條件建立簡單規則組合

沒有 baseline 時，深度學習模型即使準確率高也難以證明有價值。

### 8.2 主模型：低容量事件條件式深度模型

第一版深度學習主模型建議採用低容量 MLP 或 event-gated MLP，而不是大型 LSTM / Transformer。

設計原則：

- 輸入以日聚合事件特徵與少量市場狀態特徵為主，不餵入過長歷史序列。
- 隱藏層控制在 1-2 層，每層 16-64 hidden units。
- 使用 dropout、weight decay、early stopping。
- 使用 L1 或 group sparsity 讓模型自動壓低無效特徵。
- 輸出除了方向與報酬，也輸出 `trade_probability` 或 `confidence`。

建議架構：

```text
Trump event features ----\
Market state features ----> small MLP encoder -> multi-task heads
Regime features ---------/

Heads:
  direction probability
  expected return
  big move probability
  abstention / trade gate
```

這種模型仍可學到非線性關係，但參數量遠低於 Transformer，比較符合幾百個有效事件樣本的資料規模。

### 8.3 事件觸發與拒絕交易機制

深度模型不應每天都強迫交易。原 `trump-code` 的一個優點是「沒有條件成立就不進場」，因此深度學習版也要保留低 coverage 的特性。

建議建立兩層 gate：

1. `event_gate`：只有當 Trump event intensity 超過門檻才允許輸出交易訊號。
2. `confidence_gate`：只有當模型信心、校準後機率或預期報酬超過門檻才進場。

事件門檻可包含：

- `post_count > 0`
- `tc_event_intensity > threshold`
- `sum_tc_tariff + sum_tc_action + sum_tc_attack > 0`
- `abs(sum_weighted_vader)` 或 `max_emotional_intensity` 超過分位數門檻
- 2025-2026 關稅事件日或台指期夜盤大幅反應日

評估時必須同時報告：

- all-day prediction performance
- event-day-only performance
- high-confidence signal performance
- coverage
- no-trade days ratio

若模型每天都有輸出但交易績效只來自少數事件日，研究結論應以 event-day 與 high-confidence 訊號為主。

### 8.4 受限序列模型：只作對照，不作預設主力

LSTM / Transformer 可以作為消融實驗，但需嚴格限制容量：

- 輸入長度：過去 20 或 60 個交易日
- 每日輸入：Trump 聚合特徵 + 市場滯後特徵 + 法人與融資融券特徵
- LSTM：1 層，hidden size 16-32
- Transformer：1-2 層，1-2 attention heads，embedding dimension 16-32
- 參數量需列入報告，並和訓練樣本數比較
- 若 test 表現沒有穩定勝過 MLP / logistic baseline，則不得把序列模型作為最終主張
- 輸出：
  - direction probability
  - return prediction
  - volatility / big move probability

損失函數：

`loss = BCE(direction) + alpha * Huber(return) + beta * BCE(big_move)`

### 8.5 文字增強模型

第二階段可加入文字 embedding：

- 使用 FinBERT、DeBERTa 或 sentence-transformers 將每日貼文壓成 embedding
- 對同一天多篇貼文做 attention pooling
- 與 tabular daily features 合併

注意：第一版不應 fine-tune 大型語言模型。若使用 embedding，建議離線產生固定 embedding，再用 PCA 或 autoencoder 壓到 8-32 維，避免文字向量維度遠大於有效樣本數。

### 8.6 多標的學習

可採用 shared encoder + per-target heads：

- shared encoder 學習川普事件與全球市場共同狀態
- `2330.TW`、`2454.TW`、`0050.TW` 各自一個輸出 head
- 優點是資料量較小時能共享訊號，並比較不同標的對川普事件的敏感度

多標的學習也要控制容量。若 shared encoder 參數量過大，應改用 pooled logistic / small MLP，並加入 target embedding 或 target one-hot。

## 9. 訓練與驗證方法

### 9.1 時間切分

不得隨機切分。原本可用 `2017-2024` 訓練與驗證、`2025-2026` 最終測試，但這個切分過於粗略，因為 2021-2024 川普不是總統，2020 疫情期間市場也被全球流動性與恐慌主導。因此建議改用「政權狀態 + 疫情 regime」敏感的實驗設計。

主要分段如下：

- 第一任總統期：2017-01-20 至 2021-01-20。
- COVID 急跌期：2020-02-01 至 2020-04-30。
- COVID 寬鬆復甦期：2020-05-01 至 2021-12-31。
- 非總統期：2021-01-21 至 2025-01-19。
- 第二任總統期：2025-01-20 至 2026 最新資料。

建議至少做三組主要實驗：

| 實驗 | 訓練資料 | 驗證資料 | 最終測試 | 研究目的 |
| --- | --- | --- | --- | --- |
| Regime-matched | 2017-01-20 至 2019-12-31，並可納入 2020-05 至 2021-01 但標記 COVID | 2020-01 至 2021-01 分段驗證 | 2025-01-20 至 2026 最新資料 | 測試「總統任內政策訊號」是否能跨任期重現 |
| All-history | 2017-01-20 至 2024-12-31 | 時間序列 rolling validation | 2025-01-20 至 2026 最新資料 | 測試更多資料是否勝過 regime matched |
| Weighted / regime-aware | 2017-01-20 至 2024-12-31，但總統任內樣本與關稅事件加權，非總統期降權，加入 regime features | 2023-2024 與 2020 COVID 分段驗證 | 2025-01-20 至 2026 最新資料 | 折衷保留資料量並降低非總統期與疫情噪音 |

另做 walk-forward validation：

1. 用 2017-2019 訓練，測 2020
2. 用 2017-2020 訓練，測 2021
3. 用 2017-2021 訓練，測 2022
4. 持續滾動至 2026

COVID 期間需做 sensitivity analysis：

- Include COVID：完整保留 2020-02 至 2021-12，並加入 COVID regime 特徵。
- Exclude crash：排除 2020-02 至 2020-04 急跌期，檢查模型是否依賴極端行情。
- Downweight COVID：保留資料但降低 2020-02 至 2021-12 的樣本權重。
- Separate COVID report：單獨報告 COVID 期間表現，不和一般市場期間混合解讀。

這對應原 `trump-code` 的 train/test survivor 規則，但更符合時間序列模型評估。

### 9.2 防止資料洩漏

必須檢查：

- 當日盤後才公布的資料不可用於預測當日開盤
- `effective_trade_date` 是否正確順延
- rolling feature 是否只用過去資料
- normalization scaler 只能 fit 在 train split
- test 區間不可參與特徵選擇與超參數調整
- COVID、總統任期、關稅 regime 等標籤只能使用該日期已知或研究者事前定義的資訊，不可用未來市場結果回填標籤

### 9.3 評估指標

模型指標：

- Direction accuracy
- Balanced accuracy
- AUC
- F1
- MAE / RMSE
- Calibration error

交易模擬指標：

- Hit rate
- Average return per signal
- Cumulative return
- Sharpe ratio
- Max drawdown
- Turnover
- Coverage：模型實際出訊號比例

研究報告應同時報告「所有交易日預測」與「高信心訊號」表現，避免只挑選少數好看的案例。

### 9.4 小樣本與過擬合控制

本專案必須把過擬合作為主要研究風險處理，而不是訓練完成後才檢查。建議採用下列限制：

- 模型參數量上限：第一版主模型參數量應控制在訓練樣本數的同一量級或更低，並在報告列出參數量。
- Feature budget：第一版只使用最有理論基礎的 30-80 個特徵，不直接把所有欄位丟入模型。
- Nested validation：特徵選擇、門檻選擇、模型超參數只能在 validation / rolling validation 中決定。
- Purged walk-forward：若 target 使用未來 3 日報酬，訓練與驗證切分間需留出 purge gap，避免 horizon 重疊造成洩漏。
- Event bootstrap：對事件日做 bootstrap 或 block bootstrap，檢查績效是否被少數極端事件主導。
- Label permutation test：打亂 target 後重訓，確認模型在隨機標籤下不應仍有漂亮績效。
- Ablation test：逐步移除 Trump event features、market features、regime features，確認模型不是只靠大盤動能或 VIX。
- Stability test：同一模型用不同 random seed、不同事件門檻、不同訓練期，表現不應大幅翻轉。

若深度模型只在 train / validation 漂亮，但在 2025-2026 final test 或 bootstrap 中不穩定，研究結論應改為「深度模型未能穩定勝過規則與簡單 baseline」，不能強行主張有效。

## 10. 專案模組規劃

建議建立以下結構：

```text
src/
  config.py
  data/
    build_trump_daily_features.py
    build_market_features.py
    align_dataset.py
    split.py
  models/
    event_gated_mlp.py
    tiny_transformer.py
    tiny_lstm.py
    mlp.py
    baselines.py
  training/
    dataset.py
    trainer.py
    losses.py
    metrics.py
  evaluation/
    backtest.py
    walk_forward.py
    explainability.py
  inference/
    predict_daily.py
    signal_logger.py
  utils/
    calendar.py
    io.py
```

輸出資料建議放在：

```text
outputs/
  features/
  datasets/
  models/
  predictions/
  reports/
```

## 11. 開發里程碑

### Milestone 1：資料對齊與 EDA

目標：

- 完成 Trump 貼文有效交易日對齊
- 完成台股、市場控制變數、法人、融資融券合併
- 產出第一版 modeling table

交付：

- `outputs/datasets/modeling_table.parquet`
- EDA notebook 或 markdown 報告
- 特徵與 target 的資料字典

### Milestone 2：Trump Code baseline 重現

目標：

- 用現有 `tc_*` 欄位建立簡化版規則 baseline
- 測試 tariff / deal / relief / china / post volume 等事件與台股報酬的關係

交付：

- 規則 baseline 結果
- 與原 `trump-code` 做法的對照表
- 顯著性與交易模擬結果

### Milestone 3：深度學習模型 v1

目標：

- 建立 PyTorch Dataset
- 訓練 Logistic / LightGBM / small MLP / event-gated MLP
- 完成方向分類與報酬回歸
- 建立 abstention / no-trade gate，避免每天強迫出訊號

交付：

- 訓練腳本
- model checkpoint
- validation/test metrics
- prediction CSV
- coverage 與 no-trade ratio 報告

### Milestone 4：walk-forward 與回測

目標：

- 實作 rolling train/test
- 用高信心訊號做簡單策略回測
- 比較 baseline 與深度模型

交付：

- walk-forward report
- backtest report
- model leaderboard

### Milestone 5：解釋性與報告

目標：

- 分析哪些事件與市場特徵最影響模型
- 對照原 `trump-code` 的 key discoveries
- 整理課堂簡報與最終報告

交付：

- attention / SHAP / permutation importance 圖表
- 最終專題報告
- Demo inference 腳本

## 12. 風險與限制

- 台股受美股、匯率、半導體產業、法人籌碼影響很大，川普貼文只是其中一個外生事件來源。
- 原 `trump-code` 的高命中率來自大量規則搜尋，可能存在資料探勘偏誤；本專案必須用 out-of-sample 與 walk-forward 驗證控制這個問題。
- 2025-2026 的川普發文型態、政策環境與 2017-2020 可能不同，模型需要處理 regime shift。
- 2021-2024 川普並非總統，貼文的政策執行力與市場定價方式可能不同於 2017-2021 與 2025-2026；不可把非總統期樣本視為完全同質。
- 2020-2021 疫情期間市場受全球恐慌、貨幣政策、供應鏈中斷與科技股估值重定價影響，必須用 COVID regime 特徵、排除實驗或降權實驗控制。
- 2017-2026 約只有 2,200 個台股交易日，有效 Trump shock 事件日更少；高容量 Transformer / LSTM 很容易記住歷史噪音，不能作為第一版主模型。
- 深度模型若每天都輸出交易方向，會失去原 `trump-code` 低 coverage、事件觸發的優點；必須加入 event gate、confidence gate 與 no-trade 狀態。
- 資料中存在不同市場時區與公布時間差，若對齊錯誤會造成嚴重未來資訊洩漏。
- 深度學習模型可學到非線性關係，但也更容易過擬合；需要 baseline、regularization、early stopping 與簡潔模型設計。

## 13. 建議第一版最小可行成果

第一版不必直接 fine-tune 文字模型。最務實的 MVP 是：

1. 將貼文轉為台股交易日聚合特徵。
2. 合併 `global_prices.csv`、`global_volumes.csv`、`tx_futures_night.csv`。
3. 先預測 `2330.TW` 的隔日方向與報酬。
4. 實作 Logistic Regression、LightGBM、small MLP、event-gated MLP 四種模型。
5. 加入 `is_president`、`first_term`、`post_presidency`、`second_term`、`covid_crash_period`、`covid_recovery_liquidity_period`、`high_vix_regime` 等 regime 特徵。
6. 比較三種切分：`Regime-matched`、`All-history`、`Weighted / regime-aware`。
7. 將 2025-01-20 至 2026 最新資料作為最終測試，並單獨報告 2025-2026 關稅事件日表現。
8. 加入 `event_gate` 與 `confidence_gate`，明確允許模型輸出 no-trade。
9. 用 hit rate、AUC、平均訊號報酬、coverage、no-trade ratio 與簡單回測比較模型。

這樣可清楚展示「從原 Trump Code 規則搜尋，升級為深度學習序列預測」的核心貢獻，也能在課程專案時間內完成可驗證成果。

## 14. 參考來源

- `sstklen/trump-code` GitHub repository: https://github.com/sstklen/trump-code
- 原專案 README：描述事件分析、暴力搜尋、模型排行榜、每日閉環與 circuit breaker 架構。
- 原專案 `analysis_11_brute_force.py`：提供 2-4 條件組合、LONG/SHORT、持有 1-3 天、train/test 篩選與 p-value 檢定的核心規則搜尋邏輯。
- 本地文字資料說明：`data/text/README.md`
- 本地台灣市場資料說明：`data/taiwan_market_data/README.md`
