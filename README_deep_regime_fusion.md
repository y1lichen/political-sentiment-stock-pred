# Trump Event + Deep Regime Fusion

This project extends the original `trump_code` brute-force event rules with a deep learning regime-fusion model.

The main entry point is:

```bash
python event_combo.py
```

## What the Model Does

1. Builds original Trump-event binary signals and two-way event combinations.
2. Runs brute-force filtering to keep historically useful event signals.
3. Builds market-regime features from:
   - target momentum, volatility, moving-average gaps, and drawdown
   - VIX level/change/z-score and calm/high-volatility flags
   - US market returns, TSM ADR, SOX, Nasdaq, S&P 500, TWD, TNX
   - Taiwan institutional investor flows
   - margin and short-sale balances
   - TX futures night-session spread and volume
4. Creates explicit regime labels:
   - `calm_bull`
   - `risk_off`
   - `oversold`
   - `neutral`
5. Adds event-regime interaction terms, for example tariff signals under high VIX or oversold markets.
6. Trains a PyTorch gated-fusion LSTM:
   - market sequence branch: learns current market regime from recent market history
   - event branch: encodes brute-force Trump signals and event-regime interactions
   - gate: lets market regime control how strongly event signals affect prediction
   - auxiliary regime head: forces the model to learn regime classification

## Ubuntu Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA, install the PyTorch build matching your server GPU from the official PyTorch instructions, then install the remaining packages from `requirements.txt`.

## Run

Default target is TSMC Taiwan stock `2330.TW`.

```bash
python event_combo.py
```

Recommended full run:

```bash
python event_combo.py \
  --target 2330.TW \
  --hold 1 \
  --binary-threshold 0.0 \
  --window 20 \
  --epochs 40 \
  --batch-size 64 \
  --output-dir data/output/deep_regime_fusion
```

Quick smoke test:

```bash
python event_combo.py --epochs 2 --patience 2 --output-dir data/output/deep_regime_fusion_smoke
```

Try another target:

```bash
python event_combo.py --target 2454.TW --output-dir data/output/deep_regime_fusion_2454
python event_combo.py --target 0050.TW --output-dir data/output/deep_regime_fusion_0050
```

## Outputs

The output directory contains:

- `regime_fusion_model.pt`: trained PyTorch model checkpoint
- `market_scaler.joblib`: scaler for market sequence features
- `event_scaler.joblib`: scaler for event/regime features
- `summary.json`: metrics, feature counts, confusion matrix, and strategy return summary
- `training_history.csv`: per-epoch train/validation metrics
- `test_predictions.csv`: test-set predictions, probabilities, regimes, and next-day returns
- `brute_force_all_events.csv`: all brute-force event statistics
- `brute_force_selected_events.csv`: selected event rules used by DL
- `event_regime_conditioned_stats.csv`: event performance split by market regime

## Notes

This is a binary classification model for next-period direction: `down` or `up`.
By default, next-period return `> 0` is labeled `up`; otherwise it is labeled `down`.
You can change this cutoff with `--binary-threshold`.
It uses a time split and rolling/lagged features where applicable to reduce look-ahead leakage.
Reported strategy returns are diagnostic only and do not include fees, slippage, borrow costs, or position sizing.
