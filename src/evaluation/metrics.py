from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
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
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
) -> dict[str, float]:
    df = frame.copy()
    long_signal = (df[gate_col].fillna(0).astype(int).eq(1)) & (df[proba_col] >= threshold)
    short_signal = (df[gate_col].fillna(0).astype(int).eq(1)) & (df[proba_col] <= 1 - threshold)
    direction = np.where(long_signal, 1, np.where(short_signal, -1, 0))
    strategy_ret = direction * df[return_col].fillna(0).to_numpy()
    signal_mask = direction != 0
    hit = np.where(direction == 1, df[return_col] > 0, df[return_col] < 0)
    hit = hit[signal_mask]
    position_change = np.abs(np.diff(np.r_[0, direction]))
    trading_cost = position_change * float(transaction_cost)
    trading_slippage = position_change * float(slippage)
    net_ret = strategy_ret - trading_cost
    slippage_ret = net_ret - trading_slippage

    equity = pd.Series(strategy_ret).cumsum()
    drawdown = equity - equity.cummax()
    nav = pd.Series(1.0 + strategy_ret).cumprod()
    nav_drawdown = nav / nav.cummax() - 1.0
    nonzero_ret = strategy_ret[signal_mask]
    downside = nonzero_ret[nonzero_ret < 0]
    std = nonzero_ret.std(ddof=1) if len(nonzero_ret) > 1 else 0.0
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    holding_lengths = []
    run_length = 0
    previous_pos = 0
    for pos in direction:
        if pos == 0:
            if run_length:
                holding_lengths.append(run_length)
                run_length = 0
            previous_pos = 0
        elif pos == previous_pos:
            run_length += 1
        else:
            if run_length:
                holding_lengths.append(run_length)
            run_length = 1
            previous_pos = pos
    if run_length:
        holding_lengths.append(run_length)
    return {
        "threshold": float(threshold),
        "coverage": float(signal_mask.mean()) if len(signal_mask) else 0.0,
        "no_trade_ratio": float((~signal_mask).mean()) if len(signal_mask) else 0.0,
        "signal_count": int(signal_mask.sum()),
        "hit_rate": float(hit.mean()) if len(hit) else float("nan"),
        "avg_signal_return": float(strategy_ret[signal_mask].mean()) if signal_mask.any() else float("nan"),
        "median_signal_return": float(np.median(strategy_ret[signal_mask])) if signal_mask.any() else float("nan"),
        "cumulative_return": float(strategy_ret.sum()),
        "compound_nav": float(nav.iloc[-1]) if len(nav) else 1.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "compound_max_drawdown": float(nav_drawdown.min()) if len(nav_drawdown) else 0.0,
        "sharpe": float(nonzero_ret.mean() / std * np.sqrt(252)) if std > 0 else float("nan"),
        "sortino": float(nonzero_ret.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else float("nan"),
        "turnover": float(position_change.sum()),
        "avg_holding_days": float(np.mean(holding_lengths)) if holding_lengths else 0.0,
        "return_after_transaction_cost": float(net_ret.sum()),
        "return_after_slippage": float(slippage_ret.sum()),
    }


def event_signal_metrics(
    frame: pd.DataFrame,
    proba_col: str = "pred_direction_proba",
    return_col: str = "target_return_1d",
    gate_col: str = "event_gate_default",
    event_col: str = "event_gate_default",
    threshold: float = 0.55,
) -> dict[str, float]:
    df = frame.copy()
    event = df[event_col].fillna(0).astype(int).eq(1)
    trade_gate = df[gate_col].fillna(0).astype(int).eq(1)
    long_signal = trade_gate & (df[proba_col] >= threshold)
    short_signal = trade_gate & (df[proba_col] <= 1 - threshold)
    direction = np.where(long_signal, 1, np.where(short_signal, -1, 0))
    returns = df[return_col].fillna(0).to_numpy()
    strategy_ret = direction * returns
    signal = direction != 0
    hit = np.where(direction == 1, returns > 0, returns < 0)

    event_signal = signal & event.to_numpy()
    non_event_signal = signal & (~event.to_numpy())
    return {
        "event_signal_count": int(event_signal.sum()),
        "non_event_signal_count": int(non_event_signal.sum()),
        "event_hit_rate": float(hit[event_signal].mean()) if event_signal.any() else float("nan"),
        "non_event_hit_rate": float(hit[non_event_signal].mean()) if non_event_signal.any() else float("nan"),
        "event_avg_signal_return": float(strategy_ret[event_signal].mean()) if event_signal.any() else float("nan"),
        "non_event_avg_signal_return": float(strategy_ret[non_event_signal].mean()) if non_event_signal.any() else float("nan"),
    }
