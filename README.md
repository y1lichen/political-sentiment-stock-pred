# Deep Trump-Code Event Sequence

本分支把專案主流程改成「深度學習版 trump_code」：

1. 不再把任務寫成單篇貼文情緒直接預測股價。
2. 也不把 trump_code 的規則硬寫成 test-time brute-force rule。
3. 先把 README 中的事件發現轉成每日序列特徵，例如 RELIEF、TARIFF、night tariff、silence day、China burst、pre-market density、Truth Social lead proxy。
4. 模型同時讀取 `market sequence branch` 與 `Trump event sequence branch`。
5. 兩個 branch 都使用 LSTM + attention pooling，再融合標的 embedding，輸出隔日二元方向：`0 = down`、`1 = up`。
6. Walk-forward split 只用 train fit scaler、validation 做 early stopping、test 做最終評估。

主程式是：

```text
train.py
src/trump_event_sequence.py
src/data_loader.py
```

舊的候選規則 neural scorer 仍保留在 `src/deep_trump_code.py`，但已不是 `train.py` 的主入口。

## 執行

若你修改了 NLP keywords / event 規則，可以用既有 `Content` 重新標記，不需要原始爬蟲檔：

```bash
source .venv/bin/activate
python data/trump_nlp/extract_nlp_features.py \
  --input data/trump_nlp/trump_posts_features_2017_2026.csv \
  --output trump_posts_features_2017_2026_relabel.csv \
  --reuse-existing-nlp \
  --force
```

訓練程式會優先讀：

```text
data/trump_nlp/trump_posts_features_2017_2026_relabel.csv
```

若該檔不存在，會退回：

```text
data/trump_nlp/trump_posts_features_2017_2026.csv
```

開始訓練：

```bash
source .venv/bin/activate
python train.py
```

## 輸出

主要輸出在 `output/`：

```text
event_sequence_predictions.csv        每個 test sample 的預測、機率、隔日報酬
event_sequence_summary.csv            每個 split/model 的 accuracy、Macro F1、高信心 long-only 統計
event_sequence_overall.csv            全部 split 的整體比較
event_sequence_target_summary.csv     各標的表現
event_sequence_market_features.csv    market branch 特徵列表
event_sequence_text_features.csv      Trump event branch 特徵列表
```

`event_sequence_summary.csv` 會同時比較：

```text
market_only      只使用市場序列
event_sequence   市場序列 + Trump event 序列
```

如果 `event_sequence` 長期沒有明顯優於 `market_only`，代表 Trump event features 對台股隔日方向沒有提供穩定的 out-of-sample 增量訊號。
