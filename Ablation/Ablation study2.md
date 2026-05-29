# Ablation Diagnostics

Experiment: `2330.TW / event_gated_mlp / regime_aware`

## 結論摘要

這批 ablation 不支持「Trump 文字本身有穩定直接 alpha」或「加入 Trump 特徵後有清楚增量訊號」。相對地，結果最明確支持的是 global/overnight market transmission：`TW_plus_global_market` 在 test 上明顯優於 `TW_market_only`，AUC、hit rate、平均訊號報酬與累積報酬都大幅改善。

`event_gate_default` 也沒有在這批輸出中提供額外價值。`Global_plus_Trump_with_gate` 相對 `Global_plus_Trump_no_gate` 的 AUC、hit rate、平均訊號報酬與累積報酬都下降，且 placebo 檢定顯示隨機或平移 event date 的結果與真實 event date 非常接近，matched non-event days 甚至有較高平均訊號報酬與累積報酬。

交易成本與滑價目前都是 `0.0` per position change；因此這裡的 return 不是實際可執行績效。只有在 costs/slippage 設為非零並重新評估後，才可視為更接近交易後表現。

## Validation-Selected Thresholds

門檻只用 validation 選出。候選門檻需 validation `signal_count >= 50`，再最大化 validation `avg_signal_return`。七組 feature set 都符合此門檻篩選條件。

| feature_set | selected_threshold | val_signal_count | val_hit_rate | val_avg_signal_return |
| --- | --- | --- | --- | --- |
| TW_self_only | 0.57 | 81 | 39.51% | 0.09% |
| TW_market_only | 0.66 | 60 | 55.00% | 0.30% |
| TW_plus_global_market | 0.75 | 232 | 74.57% | 1.25% |
| Trump_text_only | 0.50 | 522 | 39.27% | -0.14% |
| TW_plus_Trump | 0.51 | 367 | 43.05% | 0.01% |
| Global_plus_Trump_no_gate | 0.75 | 172 | 71.51% | 1.29% |
| Global_plus_Trump_with_gate | 0.75 | 148 | 73.65% | 1.31% |

## Feature Audit

七個 selected feature JSON 都存在，且 `feature_set` 與 `requested_feature_set` 一致。污染檢查全部 PASS：`date_cols`、`target_cols`、`other` 全為 0；非 gate 組沒有 `event_gate`；global/tx 類特徵只出現在 global feature sets；Trump text 只出現在 Trump feature sets。`Global_plus_Trump_with_gate` 的 `event_gate = 1` 是該組設計預期，不視為污染。

| feature_set | json_exists | feature_count | trump_text | tw_market | global_market | tx_night | institutional | margin | market_state | trump_regime | event_gate | date_cols | target_cols | other | contamination_check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TW_self_only | yes | 5 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| TW_market_only | yes | 37 | 0 | 15 | 0 | 0 | 10 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| TW_plus_global_market | yes | 77 | 0 | 15 | 31 | 7 | 10 | 12 | 2 | 0 | 0 | 0 | 0 | 0 | PASS |
| Trump_text_only | yes | 82 | 82 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| TW_plus_Trump | yes | 119 | 82 | 15 | 0 | 0 | 10 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| Global_plus_Trump_no_gate | yes | 166 | 82 | 15 | 31 | 7 | 10 | 12 | 2 | 7 | 0 | 0 | 0 | 0 | PASS |
| Global_plus_Trump_with_gate | yes | 167 | 82 | 15 | 31 | 7 | 10 | 12 | 2 | 7 | 1 | 0 | 0 | 0 | PASS |

## Test Summary

以下都是 validation-selected thresholds 對 test split 的結果。

| feature_set | threshold | auc | balanced_acc | signals | coverage | hit_rate | avg_signal_return | cumulative_return | compound_nav | max_drawdown | event_signals | non_event_signals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TW_self_only | 0.57 | 0.4896 | 0.5148 | 64 | 18.66% | 32.81% | -0.02% | -1.39% | 0.9700 | -18.94% | 51 | 13 |
| TW_market_only | 0.66 | 0.5330 | 0.5053 | 38 | 11.08% | 50.00% | -0.09% | -3.36% | 0.9604 | -16.30% | 31 | 7 |
| TW_plus_global_market | 0.75 | 0.7390 | 0.6722 | 145 | 42.27% | 69.66% | 1.25% | 181.41% | 5.9120 | -4.40% | 115 | 30 |
| Trump_text_only | 0.50 | 0.5042 | 0.4948 | 343 | 100.00% | 38.19% | -0.27% | -92.28% | 0.3699 | -114.62% | 279 | 64 |
| TW_plus_Trump | 0.51 | 0.5041 | 0.4992 | 265 | 77.26% | 39.62% | -0.21% | -56.96% | 0.5347 | -79.13% | 236 | 29 |
| Global_plus_Trump_no_gate | 0.75 | 0.7117 | 0.6635 | 132 | 38.48% | 60.61% | 0.96% | 126.86% | 3.4371 | -7.10% | 101 | 31 |
| Global_plus_Trump_with_gate | 0.75 | 0.6758 | 0.5999 | 146 | 42.57% | 54.11% | 0.59% | 86.83% | 2.3038 | -10.61% | 146 | 0 |

重點是 `TW_plus_global_market` 的 test 表現最好，且不是靠 event gate。`Trump_text_only` 幾乎是 random AUC，balanced accuracy 低於 0.5，平均訊號報酬為負。`TW_plus_Trump` 相對 `TW_market_only` 沒有改善，反而擴大 coverage 後帶來較差報酬。

## Threshold 0.55 vs Validation-Selected

| feature_set | baseline_hit_0.55 | baseline_avg_ret_0.55 | selected_threshold | selected_hit | selected_avg_ret |
| --- | --- | --- | --- | --- | --- |
| TW_self_only | 40.00% | 0.02% | 0.57 | 32.81% | -0.02% |
| TW_market_only | 42.15% | -0.01% | 0.66 | 50.00% | -0.09% |
| TW_plus_global_market | 58.42% | 0.87% | 0.75 | 69.66% | 1.25% |
| Trump_text_only | 39.90% | -0.27% | 0.50 | 38.19% | -0.27% |
| TW_plus_Trump | 38.32% | -0.27% | 0.51 | 39.62% | -0.21% |
| Global_plus_Trump_no_gate | 55.00% | 0.65% | 0.75 | 60.61% | 0.96% |
| Global_plus_Trump_with_gate | 51.03% | 0.38% | 0.75 | 54.11% | 0.59% |

Validation selection 對 global feature sets 有幫助，尤其 `TW_plus_global_market`、`Global_plus_Trump_no_gate`、`Global_plus_Trump_with_gate`。但對 Trump-only 或 local+Trump 組沒有產生正的 test 平均訊號報酬。

## Incremental Comparisons

表中的 delta 方向為 `left - right`。

| comparison | left | right | delta_auc | delta_hit_rate | delta_avg_signal_return | delta_cumulative_return | delta_signal_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Trump raw signal | Trump_text_only | TW_self_only | 0.0146 | 5.38% | -0.25% | -90.90% | 279 |
| Trump over Taiwan local | TW_plus_Trump | TW_market_only | -0.0289 | -10.38% | -0.13% | -53.60% | 227 |
| Global market transmission | TW_plus_global_market | TW_market_only | 0.2059 | 19.66% | 1.34% | 184.78% | 107 |
| Trump beyond global market | Global_plus_Trump_no_gate | TW_plus_global_market | -0.0273 | -9.05% | -0.29% | -54.55% | -13 |
| Event gate value | Global_plus_Trump_with_gate | Global_plus_Trump_no_gate | -0.0359 | -6.50% | -0.37% | -40.03% | 14 |

### Required Comparison Interpretation

- `Trump_text_only` vs `TW_self_only`: `Trump_text_only` 的 AUC 略高 0.0146，hit rate 高 5.38 個百分點，但平均訊號報酬低 0.25 個百分點、累積報酬低 90.90 個百分點。`Trump_text_only` test AUC 只有 0.5042，平均訊號報酬為 -0.27%，不支持 direct Trump text alpha。
- `TW_market_only` vs `TW_plus_Trump`: 以 `TW_plus_Trump - TW_market_only` 看，AUC 低 0.0289，hit rate 低 10.38 個百分點，平均訊號報酬低 0.13 個百分點，累積報酬低 53.60 個百分點。加入 Trump text 到台灣本地市場特徵後沒有改善。
- `TW_market_only` vs `TW_plus_global_market`: 以 `TW_plus_global_market - TW_market_only` 看，AUC 高 0.2059，hit rate 高 19.66 個百分點，平均訊號報酬高 1.34 個百分點，累積報酬高 184.78 個百分點。這是最強的正向證據，支持 global/overnight market transmission。
- `TW_plus_global_market` vs `Global_plus_Trump_no_gate`: 以 `Global_plus_Trump_no_gate - TW_plus_global_market` 看，AUC 低 0.0273，hit rate 低 9.05 個百分點，平均訊號報酬低 0.29 個百分點，累積報酬低 54.55 個百分點。Trump text/regime 在 global market 控制後沒有增量。
- `Global_plus_Trump_no_gate` vs `Global_plus_Trump_with_gate`: 以 `Global_plus_Trump_with_gate - Global_plus_Trump_no_gate` 看，AUC 低 0.0359，hit rate 低 6.50 個百分點，平均訊號報酬低 0.37 個百分點，累積報酬低 40.03 個百分點。event gate 沒有幫助，且 gate 後所有 test signals 都落在 event days，反而排除了 no-gate 組中仍有正報酬的 non-event signals。

## Placebo Diagnostics

Random event-date placebo 幾乎複製真實 event gate 表現。真實 event gate 的 hit rate 為 74.07%，random placebo 平均為 74.09%；真實平均訊號報酬 1.21%，random placebo 平均 1.19%。這不支持「精確 Trump event date 本身」是主要來源。

| kind | signal_count | hit_rate | avg_signal_return | cumulative_return |
| --- | --- | --- | --- | --- |
| random | 522.95 | 74.09% | 1.19% | 623.07% |
| real | 513.00 | 74.07% | 1.21% | 620.49% |

Shifted event-date placebo 也沒有明顯劣於真實 `shift_+0`。`shift_-1` 的 hit rate、平均訊號報酬與累積報酬甚至高於真實 event date。

| placebo | threshold | signal_count | coverage | hit_rate | avg_signal_return | cumulative_return | event_signal_count | non_event_signal_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shift_-5 | 0.75 | 503 | 20.80% | 73.76% | 1.19% | 596.29% | 446 | 57 |
| shift_-3 | 0.75 | 509 | 21.05% | 73.87% | 1.23% | 625.93% | 454 | 55 |
| shift_-1 | 0.75 | 521 | 21.55% | 75.24% | 1.28% | 667.44% | 463 | 58 |
| shift_+0 | 0.75 | 513 | 21.22% | 74.07% | 1.21% | 620.49% | 513 | 0 |
| shift_+1 | 0.75 | 520 | 21.51% | 74.81% | 1.25% | 650.44% | 465 | 55 |
| shift_+3 | 0.75 | 511 | 21.13% | 73.58% | 1.20% | 613.64% | 452 | 59 |
| shift_+5 | 0.75 | 506 | 20.93% | 74.70% | 1.24% | 627.74% | 449 | 57 |

Matched non-event day test 也削弱 event gate 解釋：real event days hit rate 較高，但 matched non-event days 的平均訊號報酬與累積報酬更高。

| sample | rows | signal_count | hit_rate | avg_signal_return | cumulative_return |
| --- | --- | --- | --- | --- | --- |
| real_event_days | 257 | 88 | 68.18% | 1.02% | 89.70% |
| matched_non_event_days | 257 | 110 | 64.55% | 1.15% | 126.64% |

## Overall Interpretation

- Direct Trump text alpha: 不支持。Trump-only 接近 random，且平均訊號報酬與累積報酬明顯為負。
- Global/overnight market transmission: 支持。`TW_plus_global_market` 是最強組合，且相對 `TW_market_only` 的增量很大。
- Event gate usefulness: 不支持。with-gate 低於 no-gate，且 placebo 結果不顯示真實 event date 有獨特優勢。
- Incremental Trump signal: 沒有清楚證據。Trump 特徵加入 local 或 global market controls 後都沒有提升 test 表現。
- Return interpretation: `cumulative_return` 是 arithmetic sum，`compound_nav` 是 compounded NAV；在 transaction cost 與 slippage 都為 0 的設定下，這些數字只代表無成本回測統計，不代表真實執行績效。
