# Deep Trump Code

以 Trump 貼文事件與台灣市場資料預測台股短期方向的低容量深度學習管線。設計依據 `PROJECT_PLAN_DEEP_TRUMP_CODE.md`：小模型優先、事件觸發、可拒絕交易，避免在低信噪比金融時序上使用過高容量模型。

## Ubuntu Server 快速開始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

建立資料集：

```bash
bash scripts/build_dataset.sh 2330.TW
```

訓練模型：

```bash
bash scripts/train.sh 2330.TW event_gated_mlp regime_aware
```

也可訓練其他模型：

```bash
bash scripts/train.sh 2330.TW logistic regime_aware
bash scripts/train.sh 2330.TW elasticnet regime_matched
bash scripts/train.sh 2454.TW random_forest all_history
bash scripts/train.sh 0050.TW small_mlp regime_aware
```

推理最新一列：

```bash
bash scripts/predict.sh 2330.TW outputs/models/event_gated_mlp_2330_TW_regime_aware.pt latest
```

輸出位置：

- 資料集：`outputs/datasets/`
- 模型：`outputs/models/`
- 預測：`outputs/predictions/`
- 指標：`outputs/reports/`

## 重要設計

- 每列資料代表一個台股交易日開盤前可用資訊。
- 市場、法人、融資券、夜盤特徵採保守滯後，降低未來資訊洩漏。
- `event_gate_default` 會讓模型在沒有 Trump 政策事件時傾向 `NO_TRADE`。
- `regime_aware` split 會對總統任內、非總統期、COVID 期間使用不同樣本權重。
- `2025-01-20` 之後保留為最終測試期。

## CLI 參數

```bash
python -m src.training.train --help
python -m src.inference.predict --help
```

