# 實驗更新與結果分析報告

基準 commit：`8773346 commit so I can run on server!`
實驗範圍：`2330.TW / event_gated_mlp / regime_aware`

### Ablation 實驗設定

`full baseline` 指的是原本完整特徵的模型。模型輸入包含 Trump text/event features、market features、regime features、institutional features、margin features、TX night futures features，交易評估時仍使用 `event_gate_default` 作為 event gate。因此它代表原始的 Trump-event-aware baseline。

`market-only` 指的是把 Trump 相關特徵拿掉後重新訓練的模型。移除的欄位包含所有 `trump_*` 欄位，以及 `event_gate_default`、`is_president`、`first_term`、`post_presidency`、`second_term`、`campaign_period`、`policy_power_score`、`tariff_regime_intensity`。保留的主要是 market-side features，例如 `mkt_*`、`vol_*`、`inst_*`、`margin_*`、`tx_night_*`，以及部分非 Trump regime features，例如 `covid_policy_period`。

目前做的 ablation / diagnostics 設定如下：

| Scenario                      | 模型吃 Trump features? | 評估時用 event gate? | 只看 event days? | 目的                                                            |
| ----------------------------- | ---------------------: | -------------------: | ---------------: | --------------------------------------------------------------- |
| `full`                      |                     是 |                   是 |               否 | 原始 baseline                                                   |
| `full_event_days`           |                     是 |                   是 |               是 | 看 full model 在 event days 內的表現                            |
| `market_only_on_event_days` |                     否 |                   是 |               是 | 檢查不吃 Trump features、只靠市場特徵，在 event days 是否仍有效 |
| `pure_market`               |                     否 |                   否 |               否 | 完全不用 Trump features 和 event gate，測純市場模型             |

最重要的比較：

| 比較                                             | 目的                                 |
| ------------------------------------------------ | ------------------------------------ |
| `full` vs `market_only_on_event_days`        | 檢查 Trump features 是否提供額外幫助 |
| `market_only_on_event_days` vs `pure_market` | 檢查 event gate 是否真的有幫助       |
| `threshold=0.55` vs `threshold=0.75`         | 檢查高信心交易是否比原本門檻更好     |

新增輸出：

| 類型                    | 路徑                                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| Market-only model       | `outputs/models/event_gated_mlp_2330_TW_regime_aware_market_only.pt`                   |
| Market-only predictions | `outputs/predictions/predictions_event_gated_mlp_2330_TW_regime_aware_market_only.csv` |
| Market-only metrics     | `outputs/reports/metrics_event_gated_mlp_2330_TW_regime_aware_market_only.json`        |
| Threshold sweep         | `outputs/reports/threshold_sweep_event_gated_mlp_2330_TW_regime_aware.csv`             |
| Event-day-only metrics  | `outputs/reports/event_day_only_event_gated_mlp_2330_TW_regime_aware.json`             |
| Ablation summary        | `outputs/reports/ablation_summary_event_gated_mlp_2330_TW_regime_aware.csv`            |
| Diagnostics report      | `outputs/reports/diagnostics_event_gated_mlp_2330_TW_regime_aware.md`                  |

驗收檢查：

- Full baseline 在 threshold `0.55` 重現原 metrics：
  - test `signal_count = 238`
  - test `coverage = 69.39%`
  - test `hit_rate = 57.98%`
- Event-day rows：
  - train `1019`
  - validation `466`
  - test `279`
- Market-only saved features 中沒有 `trump_*` 或 Trump/event regime 欄位。
- Pure-market scenario 確實產生 non-event signals，test `non_event_signal_count = 30`。

## 2. 新實驗結果分析

### 2.1 Full baseline vs market-only baseline

兩組都使用原本 threshold `0.55` 與 event gate。

| Test metric       | Full baseline | Market-only |
| ----------------- | ------------: | ----------: |
| AUC               |        0.7088 |      0.7350 |
| Accuracy          |        0.6618 |      0.6764 |
| Balanced accuracy |        0.6570 |      0.6793 |
| Signal count      |           238 |         246 |
| Coverage          |        69.39% |      71.72% |
| Hit rate          |        57.98% |      58.94% |
| Avg signal return |       0.7355% |     0.7534% |
| Cumulative return |       175.05% |     185.33% |
| Max drawdown      |        -7.11% |      -7.14% |

Market-only 在 test AUC、accuracy、hit rate、avg signal return 都略高於 full baseline。這是最重要的發現：目前這一組結果不能證明績效主要來自 Trump text/event features。相反地，模型可能大量依賴市場動能、波動、法人、融資券、夜盤等 market-side features。

這不代表 Trump features 沒用，而是代表目前實驗下，移除 Trump features 後績效沒有下降；因此「Trump 訊號提供獨立 alpha」還沒有被支持。

### 2.2 Threshold sweep 結果

Validation 選出的 threshold 四個 scenario 都是 `0.75`。

| Scenario                   | Test threshold | Signals | Coverage | Hit rate | Avg signal return | Cumulative return | Non-event signals |
| -------------------------- | -------------: | ------: | -------: | -------: | ----------------: | ----------------: | ----------------: |
| Full baseline threshold    |           0.55 |     238 |   69.39% |   57.98% |           0.7355% |           175.05% |                 0 |
| Full validation-selected   |           0.75 |     103 |   30.03% |   67.96% |           1.2029% |           123.90% |                 0 |
| Full event days only       |           0.75 |     103 |   36.92% |   67.96% |           1.2029% |           123.90% |                 0 |
| Market-only on event days  |           0.75 |     124 |   44.44% |   65.32% |           1.1259% |           139.61% |                 0 |
| Pure market, no event gate |           0.75 |     154 |   44.90% |   67.53% |           1.2199% |           187.86% |                30 |

Threshold 從 `0.55` 提高到 `0.75` 後，full model 的交易次數從 `238` 降到 `103`，hit rate 從 `57.98%` 提高到 `67.96%`，avg signal return 從 `0.7355%` 提高到 `1.2029%`。這表示模型的高信心區間確實比較有用。

但 cumulative return 從 `175.05%` 降到 `123.90%`，原因是交易數量大幅下降。這裡要明確區分：

- 如果目標是提高單筆交易品質，`0.75` 比 `0.55` 好。
- 如果目標是最大化未扣成本的 arithmetic cumulative return，`0.55` 不一定輸。
- 現在 threshold 選擇規則是依 validation `avg_signal_return`，不是依 cumulative return 或 risk-adjusted return。

### 2.3 Event-day-only 評估

`full` 原本就使用 `event_gate_default`，所以 `full` 和 `full_event_days` 的 signal count、hit rate、return 相同；差異只在 coverage 的分母：

- `full` test coverage：`103 / 343 = 30.03%`
- `full_event_days` test coverage：`103 / 279 = 36.92%`

這說明 event gate 的角色主要是限制交易 universe。當只看 event days，模型仍會拒絕大約六成 event-day rows，代表 threshold `0.75` 是很保守的高信心設定。

### 2.4 Trump feature ablation

最關鍵比較是：

| Scenario                  | Signals | Hit rate | Avg signal return | Cumulative return |
| ------------------------- | ------: | -------: | ----------------: | ----------------: |
| Full event days only      |     103 |   67.96% |           1.2029% |           123.90% |
| Market-only on event days |     124 |   65.32% |           1.1259% |           139.61% |

Market-only-on-event-days 不吃 Trump text/event features，但在同一批 event days 上表現接近 full model，甚至 cumulative return 更高。這代表 event-day performance 很可能不是只靠 Trump features，而是 market features 在 event-day subset 上已經有很強解釋力。

另一個重要比較是 pure-market：

| Scenario                   | Signals | Hit rate | Avg signal return | Cumulative return | Non-event signals |
| -------------------------- | ------: | -------: | ----------------: | ----------------: | ----------------: |
| Market-only on event days  |     124 |   65.32% |           1.1259% |           139.61% |                 0 |
| Pure market, no event gate |     154 |   67.53% |           1.2199% |           187.86% |                30 |

Pure-market 拿掉 event gate 後，test 期多出 `30` 筆 non-event signals，hit rate 和 avg signal return 都沒有變差。這進一步挑戰了目前 event gate 的必要性：至少在這一組 `2330.TW / event_gated_mlp / regime_aware` 實驗中，event gate 不是明顯提升績效的關鍵。

### 2.5 目前結論

1. Threshold sweep 顯示模型高信心區間較有預測力，`0.75` 明顯提高 hit rate 與單筆平均報酬。
2. Market-only ablation 沒有讓 test 績效下降，反而略優於 full baseline。
3. 在 event days 上，market-only model 接近 full model，表示目前績效可能主要來自市場狀態與動能，而不是 Trump text features 的獨立貢獻。
4. Pure-market 在 no-event days 也能產生有效 signals，表示 event gate 需要重新檢驗。
5. 目前所有 return 都未扣交易成本、滑價，`cumulative_return` 是逐日 strategy return 加總，不是複利 NAV。因此不能直接解讀成可交易績效。

## 3. 下一步可以做什麼

### 3.1 先補嚴格交易績效

目前結果仍偏研究診斷。下一步應該加入：

- 交易成本與滑價。
- long/short 分別成本。
- turnover、平均持有天數、連續交易限制。
- compound NAV、annualized return、Sharpe、Sortino、Calmar。
- drawdown duration。

這會回答：high hit rate 是否仍能在成本後存活。

### 3.2 做更多 ablation

建議新增以下 feature sets：

| Feature set         | 目的                                                 |
| ------------------- | ---------------------------------------------------- |
| `full`            | 原始模型                                             |
| `market_only`     | 檢查是否只靠市場動能                                 |
| `trump_only`      | 檢查 Trump features 單獨是否有訊號                   |
| `no_momentum`     | 移除 `mkt_*ret*`，檢查是否只靠 price momentum      |
| `no_vix_macro`    | 移除 VIX、US index、TX night 等 global risk features |
| `event_gate_only` | 不用 text magnitude，只保留 event gate/regime        |

特別是 `trump_only` 和 `no_momentum`，能更直接回答「Trump features 是否有獨立貢獻」。

### 3.3 擴到其他 target 與 baseline models

目前只看 `2330.TW`，容易過度解讀單一股票結果。建議跑：

- Targets：`2454.TW`、`0050.TW`
- Models：`logistic`、`elasticnet`、`random_forest`、`small_mlp`

如果 market-only 在多數 target/model 都接近或優於 full，就代表 Trump text features 的 alpha 更可疑。反之，如果 full 只在特定 target 或事件 regime 顯著勝出，就能定位 Trump 訊號的有效範圍。

### 3.4 改進 threshold selection

目前 threshold sweep 的 validation best 都卡在 `0.75`，代表搜尋上界可能太低或 objective 太單一。建議：

- threshold range 擴到 `0.90`。
- 設定更嚴格的 minimum signal count。
- 同時計算 validation Sharpe / max drawdown / cumulative return。
- 用 walk-forward threshold selection，避免單一 validation 區間決定全測試期。

### 3.5 做 placebo / robustness checks

為了避免把市場 regime 誤認成 Trump effect，可以做：

- 打亂 Trump event dates，再重跑 event-gated evaluation。
- 將 event gate 往前/往後 shift 1 到 5 天。
- 比較 event days vs matched non-event days，匹配 VIX、前一日報酬、波動度。
- 分年度、分月份、分 high/low volatility regime 看穩定性。
- 對 signal return 做 bootstrap confidence interval。

### 3.6 檢查 `pred_trade_gate`

目前訓練模型有輸出 `pred_trade_gate`，但 metrics 實際使用的是資料欄位 `event_gate_default`。下一步可以比較：

- `event_gate_default`
- `pred_trade_gate`
- `event_gate_default AND pred_trade_gate`
- no gate

這能確認模型學到的 trade gate 是否比手寫 event gate 更有用。

## 建議優先順序

1. 加入交易成本與 compound NAV，重算現在所有 diagnostics。
2. 新增 `trump_only` 與 `no_momentum` ablation。
3. 將同一套 diagnostics 擴到 `2454.TW` 和 `0050.TW`。
4. 做 event-date placebo 與 matched non-event-day test。
5. 再決定是否保留 event gate，或改用 `pred_trade_gate` / pure-market 作為主要策略。
