"""
Evaluate training prediction CSVs produced by train.py.

This module bridges the model-training outputs in ``src/`` with the reusable
metric and backtest utilities under ``evaluation/``.  It treats ``full_model``
as the text+market model and ``pure_market`` as B2.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import backtest, signals_from_predictions
from .baselines import run_buy_and_hold, run_sma
from .ml_metrics import confusion_matrix_report, full_report


REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "target",
    "split",
    "date",
    "y_true",
    "pred_label",
    "proba_down",
    "proba_flat",
    "proba_up",
    "model_type",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "target",
    "model",
    "macro_f1",
    "precision_down",
    "precision_flat",
    "precision_up",
    "cumulative_return",
    "sharpe",
    "max_drawdown",
    "n_trades",
)

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "full_model": "Full Model",
    "pure_market": "B2 Pure Market",
}

MODEL_ORDER: tuple[str, ...] = ("full_model", "pure_market")
NA = "N/A"


def _safe_name(value: str) -> str:
    return value.replace("/", "-").replace(" ", "_")


def _load_predictions(predictions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)
    missing = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"prediction file is missing required columns: {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )

    df = df.loc[:, REQUIRED_PREDICTION_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["y_true"] = df["y_true"].astype(int)
    df["pred_label"] = df["pred_label"].astype(int)

    invalid_true = sorted(set(df["y_true"]) - {0, 1, 2})
    invalid_pred = sorted(set(df["pred_label"]) - {0, 1, 2})
    if invalid_true or invalid_pred:
        raise ValueError(
            f"labels must be in {{0, 1, 2}}. "
            f"invalid y_true={invalid_true}, invalid pred_label={invalid_pred}"
        )

    return df


def _load_close(prices_csv: Path, target: str) -> pd.Series:
    prices = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index).normalize()
    prices.index.name = "Date"
    if target not in prices.columns:
        raise KeyError(f"{target} not found in {prices_csv}.")
    close = prices[target].dropna().copy()
    close.name = target
    return close


def _sorted_unique_dates(group: pd.DataFrame) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(group["date"]).sort_values()
    if dates.has_duplicates:
        duplicated = dates[dates.duplicated()].unique()
        raise ValueError(f"prediction dates contain duplicates: {duplicated.tolist()}")
    return dates


def _close_window_for_dates(close: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    if len(dates) == 0:
        raise ValueError("prediction dates must not be empty.")

    positions = close.index.get_indexer(dates)
    missing = dates[positions < 0]
    if len(missing) > 0:
        raise KeyError(f"prediction dates missing from close series: {missing.tolist()}")

    expected_positions = np.arange(positions[0], positions[-1] + 1)
    if not np.array_equal(positions, expected_positions):
        raise ValueError(
            "prediction dates must be contiguous trading dates for one backtest window."
        )

    end_position = positions[-1] + 1
    if end_position >= len(close):
        raise ValueError(
            f"close series has no next-day price after final prediction date {dates[-1].date()}."
        )

    close_window = close.iloc[positions[0] : end_position + 1].copy()
    if not close_window.index[:-1].equals(dates):
        raise ValueError("close-window alignment failed: close.index[:-1] != prediction dates.")
    return close_window


def _write_confusion_outputs(
    target: str,
    model_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_cm_dir: Path,
) -> None:
    out_cm_dir.mkdir(parents=True, exist_ok=True)
    file_stem = f"cm_{_safe_name(target)}_{_safe_name(model_type)}"
    png_path = out_cm_dir / f"{file_stem}.png"
    csv_path = out_cm_dir / f"{file_stem}.csv"

    cm_info = confusion_matrix_report(y_true, y_pred, plot_path=png_path)
    cm_df = pd.DataFrame(
        cm_info["matrix"],
        index=cm_info["row_labels"],
        columns=cm_info["col_labels"],
    )
    cm_df.to_csv(csv_path)


def _model_summary_row(
    target: str,
    model_type: str,
    group: pd.DataFrame,
    close: pd.Series,
    out_cm_dir: Path,
) -> tuple[dict[str, Any], pd.DatetimeIndex]:
    group = group.sort_values("date")
    dates = _sorted_unique_dates(group)
    close_window = _close_window_for_dates(close, dates)

    y_true = group["y_true"].to_numpy(dtype=np.int64)
    y_pred = group["pred_label"].to_numpy(dtype=np.int64)
    y_proba = group[["proba_down", "proba_flat", "proba_up"]].to_numpy(dtype=float)
    report = full_report(y_true, y_pred, y_proba=y_proba)
    _write_confusion_outputs(target, model_type, y_true, y_pred, out_cm_dir)

    signals = pd.Series(signals_from_predictions(y_pred), index=dates, dtype=np.int8)
    bt_result = backtest(close_window, signals)

    row = {
        "target": target,
        "model": MODEL_DISPLAY_NAMES.get(model_type, model_type),
        "macro_f1": report["macro_f1"],
        "precision_down": report["precision_大跌"],
        "precision_flat": report["precision_盤整"],
        "precision_up": report["precision_大漲"],
        "cumulative_return": bt_result["cumulative_return"],
        "sharpe": bt_result["sharpe"],
        "max_drawdown": bt_result["max_drawdown"],
        "n_trades": bt_result["n_trades"],
    }
    return row, dates


def _financial_row(target: str, model: str, bt_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": target,
        "model": model,
        "macro_f1": NA,
        "precision_down": NA,
        "precision_flat": NA,
        "precision_up": NA,
        "cumulative_return": bt_result["cumulative_return"],
        "sharpe": bt_result["sharpe"],
        "max_drawdown": bt_result["max_drawdown"],
        "n_trades": bt_result["n_trades"],
    }


def _ordered_model_types(model_types: pd.Series) -> list[str]:
    present = set(model_types)
    ordered = [model_type for model_type in MODEL_ORDER if model_type in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def evaluate_training_predictions(
    predictions_path: str | Path,
    prices_csv: str | Path,
    out_summary: str | Path,
    out_cm_dir: str | Path,
) -> pd.DataFrame:
    predictions_path = Path(predictions_path)
    prices_csv = Path(prices_csv)
    out_summary = Path(out_summary)
    out_cm_dir = Path(out_cm_dir)

    predictions = _load_predictions(predictions_path)
    rows: list[dict[str, Any]] = []

    for target in sorted(predictions["target"].unique()):
        target_df = predictions[predictions["target"] == target].copy()
        close = _load_close(prices_csv, target)
        baseline_dates: pd.DatetimeIndex | None = None

        for model_type in _ordered_model_types(target_df["model_type"]):
            model_df = target_df[target_df["model_type"] == model_type].copy()
            row, dates = _model_summary_row(target, model_type, model_df, close, out_cm_dir)
            rows.append(row)

            if baseline_dates is None:
                baseline_dates = dates
            elif not baseline_dates.equals(dates):
                raise ValueError(
                    f"{target}: model_type={model_type} dates do not match baseline dates."
                )

        close_window = _close_window_for_dates(close, baseline_dates)
        rows.append(_financial_row(target, "B1 Buy & Hold", run_buy_and_hold(close_window)))
        rows.append(_financial_row(target, "B3 SMA 5/20", run_sma(close_window, fast=5, slow=20)))

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    print(f"✅ 已儲存完整評估表至 {out_summary}")
    print(f"✅ 已儲存混淆矩陣至 {out_cm_dir}")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.training_outputs",
        description="Evaluate train.py prediction outputs with ML and financial metrics.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("output/training_predictions.csv"),
        help="Prediction CSV produced by train.py.",
    )
    parser.add_argument(
        "--prices-csv",
        type=Path,
        default=Path("data/taiwan_market_data/global_prices.csv"),
        help="Price CSV containing target close-price columns.",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("output/evaluation_summary.csv"),
        help="Path for the summary CSV.",
    )
    parser.add_argument(
        "--out-cm-dir",
        type=Path,
        default=Path("output/confusion_matrices"),
        help="Directory for confusion-matrix CSV/PNG files.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    evaluate_training_predictions(
        predictions_path=args.predictions,
        prices_csv=args.prices_csv,
        out_summary=args.out_summary,
        out_cm_dir=args.out_cm_dir,
    )
