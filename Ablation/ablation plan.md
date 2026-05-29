## Trump Post 對台股預測之 Ablation Study 實作 Plan

### 1. 研究目標

本實驗目標是重新設計 ablation study，用來檢查 Trump post features 是否對台灣股價預測具有額外貢獻。

目前原本的 ablation 只有比較：

　　`Full model` vs `Market-only model`

但目前的 `Market-only model` 雖然移除了直接的 Trump text/event features，仍然保留了美股、VIX、SOX、NDX、TSM ADR、TX night futures 等變數。這些變數可能已經反映 Trump post 對全球市場與夜盤的影響，因此不能被視為完全沒有 Trump effect 的乾淨 baseline。

因此，本次實驗要把 input features 拆成六組：

　　1. `TW_self_only`
　　2. `TW_market_only`
　　3. `TW_plus_global_market`
　　4. `Trump_text_only`
　　5. `TW_plus_Trump`
　　6. `Global_plus_Trump` / `Full`

目的不是只看哪個模型最高，而是透過不同 feature set 的比較，判斷 Trump post 的資訊是否：

　　1. 單獨具有預測力
　　2. 在台灣本地資訊之外有額外貢獻
　　3. 在美股、VIX、SOX、TX night futures 等全球市場反應之外仍有額外貢獻
　　4. 主要透過 global/overnight market reaction 傳導到台股

---

### 2. 實驗總覽

請建立以下六組 feature set，並使用相同的 model architecture、train/validation/test split、label、threshold selection rule 與 evaluation metrics 進行比較。

| Experiment                     | Trump features | Taiwan local info | US/global market | TX night | Purpose                                   |
| ------------------------------ | -------------: | ----------------: | ---------------: | -------: | ----------------------------------------- |
| `TW_self_only`               |             No |         2330 only |               No |       No | 最基本 baseline                           |
| `TW_market_only`             |             No |               Yes |               No |       No | 純台灣市場模型                            |
| `TW_plus_global_market`      |             No |               Yes |              Yes |      Yes | 檢查全球市場是否已吸收 Trump effect       |
| `Trump_text_only`            |            Yes |                No |               No |       No | 檢查 Trump post 單獨訊號                  |
| `TW_plus_Trump`              |            Yes |               Yes |               No |       No | 檢查 Trump 對台灣本地資訊的額外貢獻       |
| `Global_plus_Trump`/`Full` |            Yes |               Yes |              Yes |      Yes | 檢查 Trump 在全球市場之外是否仍有額外資訊 |

---

### 3. Feature Set 定義

#### 3.1 `TW_self_only`

目的：建立最基本的 2330.TW 自身 baseline，檢查模型是否只靠個股自身 momentum 就能預測。

保留 features：

　　`mkt_2330_TW_ret_lag1`
　　`mkt_2330_TW_ret_3d_lag1`
　　`mkt_2330_TW_ret_5d_lag1`
　　`mkt_2330_TW_volatility_20d_lag1`
　　`vol_2330_TW_change_5d_lag1`

移除 features：

　　所有 `trump_*`
　　所有美股 / 全球市場 features
　　所有 `tx_night_*`
　　所有 `inst_*`
　　所有 `margin_*`
　　所有 Trump regime / event gate features

---

#### 3.2 `TW_market_only`

目的：建立純台灣本地資訊 baseline，避免讓模型透過美股、VIX、SOX、TX night futures 間接看到 Trump effect。

保留 features：

　　2330.TW market features
　　2454.TW market features
　　0050.TW market features
　　台股相關 volume features
　　`inst_*`
　　`margin_*`

建議包含：

　　`mkt_2330_TW_*`
　　`mkt_2454_TW_*`
　　`mkt_0050_TW_*`
　　`vol_2330_TW_*`
　　`vol_2454_TW_*`
　　`vol_0050_TW_*`
　　`inst_*`
　　`margin_*`

移除 features：

　　所有 `trump_*`
　　`event_gate_default`
　　`is_president`
　　`first_term`
　　`post_presidency`
　　`second_term`
　　`campaign_period`
　　`policy_power_score`
　　`tariff_regime_intensity`
　　`mkt_TSM_*`
　　`mkt_idx_GSPC_*`
　　`mkt_idx_NDX_*`
　　`mkt_idx_SOX_*`
　　`mkt_idx_VIX_*`
　　`mkt_idx_TNX_*`
　　`mkt_TWD_X_*`
　　`vol_TSM_*`
　　`vol_idx_GSPC_*`
　　`vol_idx_NDX_*`
　　`vol_idx_SOX_*`
　　`vol_idx_VIX_*`
　　`vol_idx_TNX_*`
　　`vol_TWD_X_*`
　　所有 `tx_night_*`

---

#### 3.3 `TW_plus_global_market`

目的：檢查美股、VIX、SOX、TSM ADR、TX night futures 是否已經吸收 Trump post 的市場反應。

保留 features：

　　`TW_market_only` 的所有 features
　　`mkt_TSM_*`
　　`mkt_idx_GSPC_*`
　　`mkt_idx_NDX_*`
　　`mkt_idx_SOX_*`
　　`mkt_idx_VIX_*`
　　`mkt_idx_TNX_*`
　　`mkt_TWD_X_*`
　　`vol_TSM_*`
　　`vol_idx_GSPC_*`
　　`vol_idx_NDX_*`
　　`vol_idx_SOX_*`
　　`vol_idx_VIX_*`
　　`vol_idx_TNX_*`
　　`vol_TWD_X_*`
　　`tx_night_*`
　　`high_vix_regime`
　　`market_stress_score`

移除 features：

　　所有 `trump_*`
　　`event_gate_default`
　　`is_president`
　　`first_term`
　　`post_presidency`
　　`second_term`
　　`campaign_period`
　　`policy_power_score`
　　`tariff_regime_intensity`

這組接近目前的 `market_only`，但要明確命名為 `TW_plus_global_market`，不要再稱為純粹的 `market_only`，因為它包含可能受到 Trump post 影響的 global/overnight market reaction。

---

#### 3.4 `Trump_text_only`

目的：檢查 Trump post features 本身是否具有 raw predictive signal。

保留 features：

　　所有 `trump_*`

建議先不要放入 regime features，讓這組保持最乾淨。

第一版：

　　`Trump_text_only`
　　　　只包含 `trump_*`

第二版可選：

　　`Trump_text_plus_regime`
　　　　包含 `trump_*`
　　　　`is_president`
　　　　`first_term`
　　　　`post_presidency`
　　　　`second_term`
　　　　`campaign_period`
　　　　`policy_power_score`
　　　　`tariff_regime_intensity`

移除 features：

　　所有台股 market features
　　所有美股 / 全球市場 features
　　所有 `tx_night_*`
　　所有 `inst_*`
　　所有 `margin_*`

---

#### 3.5 `TW_plus_Trump`

目的：檢查 Trump features 是否能在台灣本地資訊之外提供額外貢獻。

保留 features：

　　`TW_market_only` 的所有 features
　　所有 `trump_*`

可選擇加入：

　　`is_president`
　　`first_term`
　　`post_presidency`
　　`second_term`
　　`campaign_period`
　　`policy_power_score`
　　`tariff_regime_intensity`

移除 features：

　　所有美股 / 全球市場 features
　　所有 `tx_night_*`
　　`event_gate_default`

這組是最重要的比較之一。

主要比較：

　　`TW_market_only` vs `TW_plus_Trump`

如果 `TW_plus_Trump` 明顯優於 `TW_market_only`，代表 Trump features 在台灣本地資訊之外有額外預測貢獻。

---

#### 3.6 `Global_plus_Trump` / `Full`

目的：檢查當模型已經看到 global/overnight market reaction 時，Trump text features 是否仍有額外資訊。

保留 features：

　　`TW_plus_global_market` 的所有 features
　　所有 `trump_*`

可選擇加入：

　　Trump regime features
　　`event_gate_default`

這組對應目前 full model，但建議另外保留兩版：

　　`Global_plus_Trump_no_gate`
　　　　包含 Trump features，但不使用 `event_gate_default`

　　`Global_plus_Trump_with_gate`
　　　　包含 Trump features，且使用 `event_gate_default`

這樣可以分開檢查：

　　Trump text features 的效果
　　event gate 的效果

---

### 4. 實作要求

#### 4.1 新增 feature group builder

請在 feature selection pipeline 中新增 feature group builder，例如：

　　`build_feature_set(feature_set_name, all_columns)`

支援以下 `feature_set_name`：

　　`TW_self_only`
　　`TW_market_only`
　　`TW_plus_global_market`
　　`Trump_text_only`
　　`TW_plus_Trump`
　　`Global_plus_Trump_no_gate`
　　`Global_plus_Trump_with_gate`

每次 training 前，必須把實際使用的 feature list 存下來。

輸出路徑建議：

　　`outputs/features/selected_features_{model_name}_{ticker}_{feature_set}.json`

---

#### 4.2 保持實驗條件一致

所有 feature set 必須使用相同設定：

　　same target ticker，例如 `2330.TW`
　　same label，例如 `target_direction_1d`
　　same train/validation/test split
　　same model architecture
　　same random seed
　　same feature scaling method
　　same threshold sweep rule
　　same evaluation metrics

避免因為模型設定不同，導致 ablation comparison 不公平。

---

#### 4.3 Feature budget 處理

目前模型有 `feature_budget = 80`。這會造成一個問題：

　　不同 feature set 的候選特徵數量不同，如果都只選前 80 個，可能不是公平比較。

請做兩種版本：

版本 A：固定目前 `feature_budget = 80`

　　目的：跟現有實驗相容。

版本 B：不限制 feature budget，或每組使用該組所有合法 features

　　目的：確認結果不是因為 feature ordering 或 feature budget 造成。

如果只能做一版，優先做版本 B，因為 ablation study 的重點是檢查 feature group 的資訊量，而不是檢查前 80 個欄位的排序效果。

---

### 5. Evaluation Metrics

每一組 feature set 都要輸出以下 metrics。

Classification metrics：

　　AUC
　　Accuracy
　　Balanced accuracy
　　Precision
　　Recall
　　F1 score

Trading metrics：

　　Signal count
　　Coverage
　　Hit rate
　　Average signal return
　　Median signal return
　　Cumulative return
　　Compound NAV
　　Max drawdown
　　Sharpe ratio
　　Sortino ratio
　　Turnover
　　Average holding days
　　Return after transaction cost
　　Return after slippage

Event-related diagnostics：

　　Event-day signal count
　　Non-event-day signal count
　　Event-day hit rate
　　Non-event-day hit rate
　　Event-day average signal return
　　Non-event-day average signal return

---

### 6. 必要比較

請輸出一張 ablation summary table，至少包含以下比較。

#### 6.1 Trump raw signal

比較：

　　`Trump_text_only` vs `TW_self_only`

解釋：

　　如果 `Trump_text_only` 完全沒有預測力，代表 Trump post 單獨訊號很弱。
　　如果 `Trump_text_only` 有預測力，代表 Trump post 本身可能包含可用資訊。

---

#### 6.2 Trump 對台灣本地資訊的額外貢獻

比較：

　　`TW_market_only` vs `TW_plus_Trump`

解釋：

　　如果 `TW_plus_Trump > TW_market_only`，代表 Trump features 在台灣本地資訊之外有額外貢獻。
　　如果 `TW_plus_Trump ≈ TW_market_only`，代表 Trump features 沒有在台灣本地市場資訊之外提供明顯 incremental signal。

這是本研究最重要的比較之一。

---

#### 6.3 Global market 是否吸收 Trump effect

比較：

　　`TW_market_only` vs `TW_plus_global_market`

解釋：

　　如果 `TW_plus_global_market > TW_market_only`，代表美股、VIX、SOX、TSM ADR、TX night futures 對台股預測很重要。
　　這些變數可能是 Trump post effect 的傳導路徑，因此不能簡單說「不是 Trump effect」。

---

#### 6.4 Trump 在 global market 之外是否仍有效

比較：

　　`TW_plus_global_market` vs `Global_plus_Trump_no_gate`

解釋：

　　如果 `Global_plus_Trump_no_gate > TW_plus_global_market`，代表 Trump text features 在全球市場反應之外仍有額外資訊。
　　如果 `Global_plus_Trump_no_gate ≈ TW_plus_global_market`，代表 Trump effect 可能已經被 global/overnight market reaction 吸收。

---

#### 6.5 Event gate 是否真的有效

比較：

　　`Global_plus_Trump_no_gate` vs `Global_plus_Trump_with_gate`

解釋：

　　如果 with gate 較好，代表 `event_gate_default` 有幫助。
　　如果 no gate 較好，代表 event gate 可能過度限制交易 universe。

---

### 7. Placebo Tests

為了避免把 market regime 誤認成 Trump effect，請新增 placebo tests。

#### 7.1 Random event date placebo

把 Trump event dates 隨機打亂，建立 fake event days，重跑 event-day evaluation。

比較：

　　real Trump event days
　　random fake event days

如果 real event days 沒有明顯優於 random event days，則目前 event gate 可能只是挑到高波動市場，而不一定是 Trump effect。

---

#### 7.2 Shifted event date placebo

把 Trump event dates 往前與往後平移：

　　$t-5$
　　$t-3$
　　$t-1$
　　$t$
　　$t+1$
　　$t+3$
　　$t+5$

觀察哪個時間點的效果最強。

如果真正的 Trump effect 存在，理論上應該在合理反應窗口附近最強，例如隔夜反應、當日開盤反應或隔日整體反應。

如果 $t-5$ 或 $t+5$ 也一樣強，代表模型可能捕捉到的是 market regime，而不是 Trump post 的即時影響。

---

#### 7.3 Matched non-event day test

為每一個 Trump event day 找出市場狀態相近但沒有 Trump event 的 non-event day。

匹配條件建議包含：

　　前一日 2330 return
　　前 5 日 2330 return
　　20 日 volatility
　　0050 return
　　VIX return
　　SOX return
　　NDX return
　　TX night return
　　market stress score

比較：

　　Trump event days 的 average return / hit rate
　　matched non-event days 的 average return / hit rate

如果 Trump event days 明顯優於 matched non-event days，才比較能支持 Trump post 具有特殊影響。

---

### 8. 輸出檔案

請輸出以下檔案。

Model checkpoints：

　　`outputs/models/{model_name}_{ticker}_{feature_set}.pt`

Predictions：

　　`outputs/predictions/predictions_{model_name}_{ticker}_{feature_set}.csv`

Metrics：

　　`outputs/reports/metrics_{model_name}_{ticker}_{feature_set}.json`

Selected features：

　　`outputs/features/selected_features_{model_name}_{ticker}_{feature_set}.json`

Ablation summary：

　　`outputs/reports/ablation_summary_{model_name}_{ticker}.csv`

Diagnostics report：

　　`outputs/reports/diagnostics_ablation_{model_name}_{ticker}.md`

Placebo report：

　　`outputs/reports/placebo_tests_{model_name}_{ticker}.md`

---

### 9. 最終結果解讀規則

請依照以下邏輯解讀結果。

#### Case 1

如果：

　　`TW_plus_Trump > TW_market_only`
　　但
　　`Global_plus_Trump ≈ TW_plus_global_market`

解讀：

　　Trump post 可能對台股有影響，但這個影響大多已經透過美股、VIX、SOX、TSM ADR、TX night futures 等 global/overnight market reaction 被吸收。

這是最合理、也最有研究價值的結果之一。

---

#### Case 2

如果：

　　`TW_plus_Trump ≈ TW_market_only`
　　且
　　`Global_plus_Trump ≈ TW_plus_global_market`

解讀：

　　在目前 feature engineering、label、model 與 horizon 設定下，Trump features 沒有提供明顯 incremental predictive power。

不能直接說 Trump post 沒有影響，只能說目前模型沒有抓到直接的 Trump text alpha。

---

#### Case 3

如果：

　　`Trump_text_only` 有明顯預測力
　　且
　　`TW_plus_Trump > TW_market_only`
　　且
　　`Global_plus_Trump > TW_plus_global_market`

解讀：

　　Trump post features 不只單獨有訊號，也在台灣本地資訊與全球市場資訊之外提供額外 alpha。

這是最強的 Trump feature 支持結果。

---

#### Case 4

如果：

　　`TW_plus_global_market >> TW_market_only`
　　但
　　`Global_plus_Trump ≈ TW_plus_global_market`

解讀：

　　主要預測力來自 global/overnight market reaction，而不是 Trump text features 本身。
　　Trump post 的影響如果存在，可能已經被美股、VIX、SOX、TX night futures 反映完。

---

### 10. 實作優先順序

第一階段先完成主要 ablation：

　　1. `TW_self_only`
　　2. `TW_market_only`
　　3. `TW_plus_global_market`
　　4. `Trump_text_only`
　　5. `TW_plus_Trump`
　　6. `Global_plus_Trump_no_gate`
　　7. `Global_plus_Trump_with_gate`

第二階段補 robustness：

　　1. Random event date placebo
　　2. Shifted event date placebo
　　3. Matched non-event day test
　　4. Transaction cost and slippage
　　5. Compound NAV and risk-adjusted metrics

第三階段擴展 target：

　　1. `2330.TW`
　　2. `2454.TW`
　　3. `0050.TW`

如果三個 target 都呈現類似結論，研究結果會比較穩健。
