# Deep Trump-Code

本分支把原本每日分類器改成「深度學習版 trump_code」：

1. 先用 Trump-code 式事件規則產生候選交易。
2. 每個候選交易由 `target × event rule × holding horizon` 定義。
3. 深度模型讀取市場序列、Trump 事件序列、規則 embedding、標的 embedding 與持有期 embedding。
4. 模型預測候選交易方向：`0 = short/down`，`1 = long/up`。
5. 每個 walk-forward split 在 validation 上選信心門檻與 survivor rules。
6. 只在 test 中對高信心、long-only、survivor candidates 出手並統計 hit rate / return。

這比每日三分類更接近 `sstklen/trump-code` 的概念，但最後的方向與信心分數由 neural scorer 學習。

## 執行

如果同學提供的是 `data/trump_nlp/trump_posts_features_2017_2026.csv`，且你只修改了 keyword / event 規則，不需要原始爬蟲檔。先用既有 `Content` 重新標記：

```bash
source .venv/bin/activate
python data/trump_nlp/extract_nlp_features.py \
  --input data/trump_nlp/trump_posts_features_2017_2026.csv \
  --output trump_posts_features_2017_2026_relabel.csv \
  --reuse-existing-nlp \
  --force
```

這會保留同學已算好的 `vader_compound`、`emotion_label`、`emotion_score`、`weighted_vader`，只重新產生新版 keywords、Trump-code events、time/signature/event intensity 等欄位。

訓練程式會優先讀：

```text
data/trump_nlp/trump_posts_features_2017_2026_relabel.csv
```

若該檔不存在，才退回讀原始 `data/trump_nlp/trump_posts_features_2017_2026.csv`。

```bash
source .venv/bin/activate
python train.py
```

主要輸出在 `output/`：

```text
deep_trump_code_predictions.csv       每個 test candidate 的模型輸出與是否 selected
deep_trump_code_summary.csv           每個 split 的候選數、出手數、hit rate、平均策略報酬
deep_trump_code_overall.csv           全部 split 的總結
deep_trump_code_target_summary.csv    各標的 selected trades 表現
deep_trump_code_rule_survivors.csv    每個 split 通過 validation 的 survivor rules
deep_trump_code_rules.csv             rule_id 對照表
```

目前 selection policy 較保守：

```text
沒有 survivor rules 的 split 不出手
只允許 pred=1 的 long candidates
每個 split 最多選 max_selected_per_split 筆最高 confidence candidates
survivor rule 需要 validation hit rate >= 0.58 且平均策略報酬 > 0
validation 整體 regime 需要 hit rate >= 0.52 且平均策略報酬 > 0
```

## 方法差異

舊版流程是：

```text
每個高訊號交易日 -> 預測下一期上/下
```

新版流程是：

```text
每個 target × event rule × horizon -> neural scorer 評分 -> validation 選 threshold/rule -> test 只出手 survivor candidates
```

這讓模型具備 trump_code 的「選擇性出手」本質，同時保留 deep learning 作為核心決策器。
