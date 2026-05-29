from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        out["auc"] = float("nan")
    return out


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
    }


def signal_metrics(
    frame: pd.DataFrame,
    proba_col: str = "pred_direction_proba",
    return_col: str = "target_return_1d",
    gate_col: str = "event_gate_default",
    threshold: float = 0.55,
) -> dict[str, float]:
    df = frame.copy()
    long_signal = (df[gate_col].fillna(0).astype(int).eq(1)) & (df[proba_col] >= threshold)
    short_signal = (df[gate_col].fillna(0).astype(int).eq(1)) & (df[proba_col] <= 1 - threshold)
    direction = np.where(long_signal, 1, np.where(short_signal, -1, 0))
    strategy_ret = direction * df[return_col].fillna(0).to_numpy()
    signal_mask = direction != 0
    hit = np.where(direction == 1, df[return_col] > 0, df[return_col] < 0)
    hit = hit[signal_mask]

    equity = pd.Series(strategy_ret).cumsum()
    drawdown = equity - equity.cummax()
    return {
        "threshold": float(threshold),
        "coverage": float(signal_mask.mean()) if len(signal_mask) else 0.0,
        "no_trade_ratio": float((~signal_mask).mean()) if len(signal_mask) else 0.0,
        "signal_count": int(signal_mask.sum()),
        "hit_rate": float(hit.mean()) if len(hit) else float("nan"),
        "avg_signal_return": float(strategy_ret[signal_mask].mean()) if signal_mask.any() else float("nan"),
        "cumulative_return": float(strategy_ret.sum()),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
    }

