# Fusion Mode Ablation Report

## Setup

- Targets: `['2330.TW', 'TSM']`
- Fusion modes: `gated_concat, raw_concat, ungated_concat, add`
- Epochs: `80`, patience: `8`
- Seed: `42`
- Feature set: `full`
- Proposed baseline fusion: `gated_concat`

## Fusion Modes

- `gated_concat`: original model, DirectionHead([market_state, gate * event_state]).
- `raw_concat`: directly concatenates scaled raw market/event features into Direction Head.
- `ungated_concat`: DirectionHead([market_state, event_state]) without gate modulation.
- `add`: DirectionHead(market_state + gate * event_state).

## 2330.TW

| fusion_mode | accuracy | d_acc vs gated | AUC | d_AUC vs gated | cumret | d_cumret vs gated | trades |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| gated_concat | 0.429 | +0.000 | 0.536 | +0.000 | -0.145 | +0.000 | 43 |
| raw_concat | 0.470 | +0.040 | 0.523 | -0.013 | -0.417 | -0.272 | 129 |
| ungated_concat | 0.449 | +0.020 | 0.537 | +0.001 | -0.467 | -0.322 | 198 |
| add | 0.439 | +0.010 | 0.457 | -0.079 | -0.478 | -0.332 | 149 |

Best accuracy: `raw_concat` (0.470); best AUC: `ungated_concat` (0.537); best cumret: `gated_concat` (-0.145).

## TSM

| fusion_mode | accuracy | d_acc vs gated | AUC | d_AUC vs gated | cumret | d_cumret vs gated | trades |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| gated_concat | 0.498 | +0.000 | 0.524 | +0.000 | -0.156 | +0.000 | 77 |
| raw_concat | 0.479 | -0.018 | 0.489 | -0.035 | 0.340 | +0.496 | 95 |
| ungated_concat | 0.498 | +0.000 | 0.518 | -0.007 | -0.059 | +0.096 | 22 |
| add | 0.539 | +0.041 | 0.537 | +0.013 | 0.062 | +0.217 | 92 |

Best accuracy: `add` (0.539); best AUC: `add` (0.537); best cumret: `raw_concat` (0.340).
