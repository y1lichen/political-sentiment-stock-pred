
# 📊 台灣科技權值股與全球總經市場數據集

本資料集（`taiwan_market_data/`）專為分析外部事件（如美國政治領袖之社群言論）對台灣科技權值股（2330、2454、0050）之隔日開盤/收盤報酬預測能力所建置。資料涵蓋個股價量、美股領先指標、三大法人籌碼、融資券信用交易以及台指期貨夜盤。

## 📁 檔案清單與資料結構 (Data Dictionary)

所有資料皆以 CSV 格式儲存，並以時間序列為核心維度。

### 1. `global_prices.csv` & `global_volumes.csv`

這兩個檔案分別紀錄了全球股票、指數與匯率的**每日收盤價 (Close)** 與 **成交量 (Volume)**。
資料主鍵為 `Date`（交易日期）。欄位名稱即為股票代號（Ticker）：

* **台股預測標的**
* `2330.TW`：台灣積體電路製造（台積電）
* `2454.TW`：聯發科技
* `0050.TW`：元大台灣卓越50基金


* **跨國控制變數 (美國市場與總經)**
* `TSM`：台積電 ADR（直接對應美股交易時段的台積電表現）
* `^SOX`：費城半導體指數（與台灣科技供應鏈高度連動）
* `^NDX`：那斯達克 100 指數（美國大型科技股風向球）
* `^GSPC`：S&P 500 指數（美股大盤）
* `^VIX`：CBOE 波動率指數（恐慌指數，衡量市場對政治/貿易風險的避險情緒）
* `TWD=X`：美元兌新台幣匯率（捕捉國際資金移動）
* `^TNX`：美國 10 年期公債殖利率（代表無風險利率環境）

---
註：global_volumes.csv中
- `^GSPC` 的成交量 = 標普 500 指數中 500 家成分股（如蘋果、微軟、輝達等）當天成交股數的總和。
- `^NDX` 的成交量 = 那斯達克 100 指數中 100 家成分股當天成交股數的總和。

### 2. `institutional_investors.csv`

紀錄台灣市場「三大法人」針對指定標的的每日買賣超狀況。

* `date`：交易日期
* `stock_id`：股票代號（2330, 2454, 0050）
* `name`：法人名稱（包含：外資及陸資 `Foreign_Investor`、投信 `Investment_Trust`、自營商 `Dealer`）
* `buy`：該法人當日買進股數
* `sell`：該法人當日賣出股數

---
註：
- 可自行衍生計算 `net_buy = buy - sell` 作為淨買賣超特徵)*
- Foreign_Investor（外資及陸資）
代表台灣以外的國際投資機構與外國資金。外資的資金部位龐大，對台股大型權值股（如台積電、聯發科）的走勢具有決定性的影響力，通常被視為市場長期趨勢的重要風向球。

- Investment_Trust（投信）
代表「證券投資信託公司」，即國內發行共同基金或 ETF 的機構法人。投信的買賣超反映了國內基金經理人的操作動向，這股資金通常有季底結算與作帳的需求，對中小型股或具備短中期題材的股票影響力較顯著。

- Dealer_self（自營商－自行買賣）
代表國內證券商使用公司的「自有資金」進入股市進行投資或投機操作。這部分的資金操作通常較為靈活且偏向短線，主要以追求絕對報酬與波段價差為目的。

- Dealer_Hedging（自營商－避險）
代表證券商為了「風險對沖」而進行的被動買賣。最常見的情況是當一般投資人大量買進權證（認購/認售權證）或特定 ETF 時，發行這些金融商品的券商必須在現貨市場買進或賣出相對應比例的股票，以鎖定風險。這部分的買賣不代表券商主觀看多或看空該檔個股。
### 3. `margin_trading.csv`

紀錄散戶與本土資金參與的「融資」與「融券」信用交易餘額，用以評估市場過熱或悲觀情緒。

* `date`：交易日期
* `stock_id`：股票代號
* `MarginPurchaseBuy` / `MarginPurchaseSell`：融資買進 / 融資賣出（張數）
* `MarginPurchaseTodayBalance`：**融資今日餘額**（市場做多槓桿水位）
* `ShortSaleBuy` / `ShortSaleSell`：融券買進 / 融券賣出（張數）
* `ShortSaleTodayBalance`：**融券今日餘額**（市場做空水位）

### 4. `tx_futures_night.csv`

過濾出「台指期貨（TX）」的盤後交易（夜盤）數據，這是捕捉美國白晝時間發生重大政治言論時，台股市場最即時的避險反應。

* `date`：夜盤開盤所屬之交易日期（注意：夜盤的結算日歸屬通常為次一交易日）
* `futures_id`：期貨代號（固定為 TX）
* `trade_session`：交易時段（固定為 `after_market`）
* `open` / `high` / `low` / `close`：夜盤的開高低收點數
* `volume`：夜盤總成交口數

---

## 🛠️ 如何使用這項數據進行預測建模 (Usage Guide)

在將此數據集與「川普社群文本情緒分數」合併進行機器學習（如 XGBoost, LSTM）或時間序列迴歸時，請務必遵循以下處理步驟，以避免「前視偏誤」（Data Leakage / Look-ahead Bias）：

### 步驟一：處理跨時區的時間平移 (Time Shifting)

台股的交易時間為 T 日 09:00 - 13:30，而美股交易時間對應台灣為 T-1 日的晚上至 T 日凌晨。
如果您要預測 **T 日** 的台股報酬，您只能看見 **T-1 日** 的美股收盤價。因此，在合併資料前，必須將美股特徵往下平移一天：

```python
import pandas as pd

# 讀取價格資料
prices = pd.read_csv('taiwan_market_data/global_prices.csv', index_col='Date', parse_dates=True)

# 分離台股與美股
taiwan_stocks = ['2330.TW', '2454.TW', '0050.TW']
us_features = ['TSM', '^SOX', '^NDX', '^GSPC', '^VIX', 'TWD=X', '^TNX']

# 美股與總經指標必須 shift(1)，確保模型預測 T 日台股時，使用的是 T-1 日的美股收盤狀態
df_us_shifted = prices[us_features].shift(1)

```

### 步驟二：建構預測目標 (Target Variable $Y$)

依據您的研究設計，計算台積電等標的的隔日報酬率。建議將報酬率平穩化：

```python
# 目標 Y：計算 2330.TW 今日收盤相較於昨日收盤的報酬率
df_target = prices[taiwan_stocks].pct_change()
df_target = df_target.rename(columns={col: f"{col}_Return" for col in taiwan_stocks})

```

### 步驟三：資料表關聯與合併 (Merging)

籌碼資料（三大法人與融資券）是長表格（Long Format），需要先篩選出特定個股，再與價格時間序列合併。

```python
# 讀取籌碼資料
inst_df = pd.read_csv('taiwan_market_data/institutional_investors.csv', parse_dates=['date'])

# 篩選 2330 的外資買賣超
fii_2330 = inst_df[(inst_df['stock_id'] == 2330) & (inst_df['name'] == 'Foreign_Investor')]
fii_2330['net_buy'] = fii_2330['buy'] - fii_2330['sell']
fii_2330 = fii_2330.set_index('date')[['net_buy']].rename(columns={'net_buy': 'FII_Net_Buy_2330'})

# 前一天的法人買賣超也是解釋 T 日報酬的特徵，因此也要 shift(1)
fii_2330_shifted = fii_2330.shift(1)

```

### 步驟四：整合社群文本資料 (最終模型矩陣)

將處理好的美股平移資料、法人平移資料、台指期夜盤資料，與您爬蟲獲得並量化後的「川普社群發文情緒分數（Sentiment Score）」透過日期（Date）進行 `pd.concat()` 或 `.join()`。

**模型解釋性重點**：
在回歸模型中，如果 `川普情緒分數` 的係數在加入了 `TSM (ADR)` 與 `tx_futures_night (台指夜盤)` 的控制後**依然達到統計顯著（p-value < 0.05）**，這就是您的研究最具價值的結論——證明了政治人物的社交媒體言論具備超越傳統金融市場指標的**額外預測能力 (Incremental Predictive Power)**。