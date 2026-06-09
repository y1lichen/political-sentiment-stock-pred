# Fusion Mode Ablation Report

## Setup

- Targets: `['2330.TW', '2454.TW', '0050.TW', '2308.TW', '2382.TW', 'TSM', '^GSPC', '^NDX', '^SOX']`
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
| gated_concat | 0.455 | +0.000 | 0.524 | +0.000 | -0.327 | +0.000 | 105 |
| raw_concat | 0.439 | -0.015 | 0.538 | +0.014 | -0.057 | +0.270 | 21 |
| ungated_concat | 0.434 | -0.020 | 0.526 | +0.002 | -0.522 | -0.195 | 198 |
| add | 0.434 | -0.020 | 0.466 | -0.059 | -0.547 | -0.220 | 191 |

Best accuracy: `gated_concat` (0.455); best AUC: `raw_concat` (0.538); best cumret: `raw_concat` (-0.057).

## 2454.TW

| fusion_mode | accuracy | d_acc vs gated | AUC | d_AUC vs gated | cumret | d_cumret vs gated | trades |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| gated_concat | 0.465 | +0.000 | 0.477 | +0.000 | -0.092 | +0.000 | 21 |
| raw_concat | 0.530 | +0.064 | 0.540 | +0.063 | 0.218 | +0.310 | 73 |
| ungated_concat | 0.495 | +0.030 | 0.491 | +0.014 | -0.420 | -0.328 | 41 |
| add | 0.465 | +0.000 | 0.464 | -0.012 | -0.581 | -0.489 | 89 |

Best accuracy: `raw_concat` (0.530); best AUC: `raw_concat` (0.540); best cumret: `raw_concat` (0.218).

## 0050.TW

| fusion_mode | accuracy | d_acc vs gated | AUC | d_AUC vs gated | cumret | d_cumret vs gated | trades |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| gated_concat | 0.485 | +0.000 | 0.514 | +0.000 | -0.104 | +0.000 | 92 |
| raw_concat | 0.447 | -0.039 | 0.575 | +0.062 | -0.112 | -0.008 | 61 |
| ungated_concat | 0.437 | -0.049 | 0.543 | +0.030 | -0.222 | -0.119 | 100 |
| add | 0.422 | -0.063 | 0.498 | -0.016 | -0.138 | -0.034 | 39 |

Best accuracy: `gated_concat` (0.485); best AUC: `raw_concat` (0.575); best cumret: `gated_concat` (-0.104).

## 2308.TW

| fusion_mode | accuracy | d_acc vs gated | AUC | d_AUC vs gated | cumret | d_cumret vs gated | trades |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| gated_concat | 0.485 | +0.000 | 0.482 | +0.000 | -0.609 | +0.000 | 134 |
| raw_concat | 0.439 | -0.046 | 0.465 | -0.017 | -0.062 | +0.547 | 2 |
| ungated_concat | 0.474 | -0.010 | 0.523 | +0.041 | -0.668 | -0.058 | 171 |
| add | 0.459 | -0.026 | 0.513 | +0.031 | -0.179 | +0.431 | 16 |

Best accuracy: `gated_concat` (0.485); best AUC: `ungated_concat` (0.523); best cumret: `raw_concat` (-0.062).

## 2382.TW

| fusion_mode | accuracy | d_acc vs gated | AUC | d_AUC vs gated | cumret | d_cumret vs gated | trades |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| gated_concat | 0.510 | +0.000 | 0.510 | +0.000 | -0.029 | +0.000 | 198 |
| raw_concat | 0.515 | +0.005 | 0.513 | +0.003 | 0.011 | +0.040 | 198 |
| ungated_concat | 0.525 | +0.015 | 0.501 | -0.009 | -0.106 | -0.077 | 67 |
| add | 0.510 | +0.000 | 0.526 | +0.016 | -0.071 | -0.042 | 189 |

Best accuracy: `ungated_concat` (0.525); best AUC: `add` (0.526); best cumret: `raw_concat` (0.011).

## TSM

| fusion_mode | accuracy | d_acc vs gated | AUC | d_AUC vs gated | cumret | d_cumret vs gated | trades |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| gated_concat | 0.525 | +0.000 | 0.521 | +0.000 | -0.119 | +0.000 | 131 |
| raw_concat | 0.475 | -0.050 | 0.486 | -0.035 | -0.092 | +0.027 | 145 |
| ungated_concat | 0.507 | -0.018 | 0.524 | +0.003 | 0.020 | +0.139 | 46 |
| add | 0.539 | +0.014 | 0.528 | +0.008 | 0.231 | +0.350 | 219 |

Best accuracy: `add` (0.539); best AUC: `add` (0.528); best cumret: `add` (0.231).

## ^GSPC

| fusion_mode | accuracy | d_acc vs gated | AUC | d_AUC vs gated | cumret | d_cumret vs gated | trades |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| gated_concat | 0.498 | +0.000 | 0.482 | +0.000 | -0.035 | +0.000 | 208 |
| raw_concat | 0.502 | +0.005 | 0.458 | -0.024 | 0.009 | +0.044 | 189 |
| ungated_concat | 0.498 | +0.000 | 0.455 | -0.027 | 0.030 | +0.065 | 175 |
| add | 0.507 | +0.009 | 0.446 | -0.035 | -0.003 | +0.032 | 221 |

Best accuracy: `add` (0.507); best AUC: `gated_concat` (0.482); best cumret: `ungated_concat` (0.030).

## ^NDX

| fusion_mode | accuracy | d_acc vs gated | AUC | d_AUC vs gated | cumret | d_cumret vs gated | trades |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| gated_concat | 0.489 | +0.000 | 0.454 | +0.000 | -0.041 | +0.000 | 13 |
| raw_concat | 0.520 | +0.032 | 0.454 | +0.000 | -0.022 | +0.019 | 66 |
| ungated_concat | 0.502 | +0.014 | 0.469 | +0.015 | 0.076 | +0.117 | 46 |
| add | 0.493 | +0.005 | 0.467 | +0.013 | -0.021 | +0.020 | 15 |

Best accuracy: `raw_concat` (0.520); best AUC: `ungated_concat` (0.469); best cumret: `ungated_concat` (0.076).

## ^SOX

| fusion_mode | accuracy | d_acc vs gated | AUC | d_AUC vs gated | cumret | d_cumret vs gated | trades |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| gated_concat | 0.570 | +0.000 | 0.479 | +0.000 | 0.552 | +0.000 | 221 |
| raw_concat | 0.462 | -0.109 | 0.483 | +0.005 | 0.112 | -0.440 | 80 |
| ungated_concat | 0.511 | -0.059 | 0.479 | +0.000 | 0.101 | -0.451 | 188 |
| add | 0.434 | -0.136 | 0.459 | -0.020 | -0.379 | -0.931 | 86 |

Best accuracy: `gated_concat` (0.570); best AUC: `raw_concat` (0.483); best cumret: `gated_concat` (0.552).
