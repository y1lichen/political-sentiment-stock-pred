"""
evaluation/run_eval.py
======================
CLI orchestrator for the Trump-Sentiment × TSMC validation prototype.

Runs all models and baselines on the **test split only** (post val_end=2024-12-31),
then prints and saves a single comparison table covering ML metrics and
financial metrics.

Usage
-----
::

    # Baselines + dummies only
    python -m evaluation.run_eval

    # With teammate's DAF-Net predictions CSV
    python -m evaluation.run_eval --predictions path/to/predictions.csv

    # Full options
    python -m evaluation.run_eval \\
        --predictions path/to/predictions.csv \\
        --prices-csv path/to/global_prices.csv \\
        --out-table results.csv \\
        --out-cm-dir /tmp/confusion_matrices

Output table columns
--------------------
Model | Macro F1 | Precision(大漲) | Precision(大跌) | CumReturn | Sharpe | MDD | n_trades

Rows with no class-label output (B1 Buy & Hold, B3 SMA) have ``"N/A"`` in
all ML-metric columns.  B2 Pure-Market is a placeholder row when the
teammate CSV at ``data/pure_market_predictions.csv`` has not yet been delivered.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve repository root so relative data paths work whether this module is
# run from the repo root or from a subdirectory.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent

# Fallback for worktree environments where data/ may live in the canonical repo.
def _find_repo_root() -> Path:
    """Return the repo root that contains ``data/taiwan_market_data/global_prices.csv``.

    Tries the natural parent of this file first, then falls back to the
    hard-coded canonical project path used by other modules in this package.
    """
    candidate = _THIS_FILE.parent.parent
    if (candidate / "data" / "taiwan_market_data" / "global_prices.csv").exists():
        return candidate
    # Worktree structure: .claude/worktrees/<id>/evaluation/ → go up 4 levels
    for ancestor in candidate.parents:
        if (ancestor / "data" / "taiwan_market_data" / "global_prices.csv").exists():
            return ancestor
    # Hard-coded fallback (same pattern as baselines.py)
    fallback = Path(
        "/mnt/sda/home/r147250250916/2026spring/DLA/political-sentiment-stock-pred"
    )
    if fallback.exists():
        return fallback
    return candidate  # best effort


_REPO_ROOT = _find_repo_root()
_DEFAULT_PRICES_CSV = _REPO_ROOT / "data" / "taiwan_market_data" / "global_prices.csv"
_DEFAULT_OUT_TABLE = _REPO_ROOT / "evaluation_results.csv"
_B2_CSV = _REPO_ROOT / "data" / "pure_market_predictions.csv"

# ---------------------------------------------------------------------------
# Upstream module imports
# ---------------------------------------------------------------------------
from .data_module import load_tsmc_prices, make_labels, temporal_split, LABEL_NAMES
from .model_interface import (
    DummyRandomModel,
    DummyMajorityModel,
    CsvPredictionsModel,
    ClassifierModel,
)
from .ml_metrics import full_report, confusion_matrix_report
from .backtest import backtest, signals_from_predictions
from .baselines import run_buy_and_hold, run_sma

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NA = "N/A"
_TABLE_COLS = [
    "Model",
    "Macro F1",
    "Precision(大漲)",
    "Precision(大跌)",
    "CumReturn",
    "Sharpe",
    "MDD",
    "n_trades",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_float(val: float, decimals: int = 4) -> str:
    """Format a float to a fixed number of decimal places, or return N/A."""
    if val is _NA or val == _NA:
        return _NA
    return f"{val:.{decimals}f}"


def _evaluate_ml_model(
    model: ClassifierModel,
    test: pd.DataFrame,
    close_filtered: pd.Series,
    model_name: str,
    out_cm_dir: Path | None = None,
) -> dict[str, Any]:
    """Run a ClassifierModel through ML metrics + backtest and return a result row.

    Parameters
    ----------
    model:
        Any ``ClassifierModel`` instance (DummyRandom, DummyMajority, CsvPredictionsModel, …).
    test:
        Test-split DataFrame from ``temporal_split``, indexed by ``pd.DatetimeIndex``,
        with a ``"label"`` column of ground-truth integers in {0, 1, 2}.
    close_filtered:
        Close price series filtered to ``close.index >= test.index[0]``.
        Has one extra trailing date beyond ``test.index[-1]``.
    model_name:
        Human-readable name for the comparison table row.
    out_cm_dir:
        Optional directory to save a confusion-matrix PNG.

    Returns
    -------
    dict
        Keys match ``_TABLE_COLS`` (Model, Macro F1, Precision(大漲), …).
    """
    test_dates: pd.DatetimeIndex = test.index  # type: ignore[assignment]
    y_true: np.ndarray = test["label"].values

    # Step 1 — ML predictions
    y_pred: np.ndarray = model.predict(test_dates)

    # Step 2 — ML metrics
    report = full_report(y_true, y_pred)

    # Step 3 — Confusion matrix PNG (optional)
    if out_cm_dir is not None:
        out_cm_dir.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.replace(" ", "_").replace("/", "-")
        cm_path = out_cm_dir / f"cm_{safe_name}.png"
        confusion_matrix_report(y_true, y_pred, plot_path=cm_path)

    # Step 4 — Convert predictions → position signals
    signals_arr: np.ndarray = signals_from_predictions(y_pred)

    # Step 5 — Align signals to close_filtered.index[:-1] (lookahead guard)
    signals_series = pd.Series(
        signals_arr, index=close_filtered.index[:-1], dtype=np.int8
    )
    assert signals_series.index.equals(close_filtered.index[:-1]), (
        f"[{model_name}] signals index mismatch — check close_filtered alignment."
    )

    # Step 6 — Run backtest
    bt_result = backtest(close_filtered, signals_series)

    return {
        "Model": model_name,
        "Macro F1": report["macro_f1"],
        "Precision(大漲)": report["precision_大漲"],
        "Precision(大跌)": report["precision_大跌"],
        "CumReturn": bt_result["cumulative_return"],
        "Sharpe": bt_result["sharpe"],
        "MDD": bt_result["max_drawdown"],
        "n_trades": bt_result["n_trades"],
    }


def _financial_row(name: str, bt_result: dict) -> dict[str, Any]:
    """Build a result-row dict for a purely financial baseline (no ML metrics).

    Parameters
    ----------
    name:
        Row label for the Model column.
    bt_result:
        Return value of :func:`evaluation.backtest.backtest` (or compatible dict).

    Returns
    -------
    dict with ML columns set to ``"N/A"`` and financial columns filled.
    """
    return {
        "Model": name,
        "Macro F1": _NA,
        "Precision(大漲)": _NA,
        "Precision(大跌)": _NA,
        "CumReturn": bt_result["cumulative_return"],
        "Sharpe": bt_result["sharpe"],
        "MDD": bt_result["max_drawdown"],
        "n_trades": bt_result["n_trades"],
    }


def _placeholder_row(name: str, message: str) -> dict[str, Any]:
    """Build a placeholder row for missing upstream output (e.g., B2).

    Parameters
    ----------
    name:
        Row label for the Model column.
    message:
        Message to show in every data cell.

    Returns
    -------
    dict with all non-Model columns set to *message*.
    """
    return {col: (name if col == "Model" else message) for col in _TABLE_COLS}


# ---------------------------------------------------------------------------
# Table rendering helpers
# ---------------------------------------------------------------------------

def _render_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render a list of row dicts as a fixed-width markdown table string.

    Parameters
    ----------
    rows:
        List of dicts whose keys are exactly ``_TABLE_COLS``.

    Returns
    -------
    str
        Multi-line string suitable for ``print()``.
    """
    # Format values
    formatted: list[dict[str, str]] = []
    for row in rows:
        frow: dict[str, str] = {}
        for col in _TABLE_COLS:
            val = row[col]
            if val is _NA or val == _NA:
                frow[col] = _NA
            elif isinstance(val, float):
                frow[col] = f"{val:.4f}"
            else:
                frow[col] = str(val)
        formatted.append(frow)

    # Compute column widths
    col_widths: dict[str, int] = {col: len(col) for col in _TABLE_COLS}
    for frow in formatted:
        for col in _TABLE_COLS:
            col_widths[col] = max(col_widths[col], len(frow[col]))

    def _row_str(data: dict[str, str]) -> str:
        parts = []
        for col in _TABLE_COLS:
            w = col_widths[col]
            cell = data[col]
            # Right-align numeric columns; left-align Model column
            if col == "Model":
                parts.append(cell.ljust(w))
            else:
                parts.append(cell.rjust(w))
        return "| " + " | ".join(parts) + " |"

    def _sep_str() -> str:
        parts = []
        for col in _TABLE_COLS:
            w = col_widths[col]
            if col == "Model":
                parts.append("-" * w)
            else:
                parts.append("-" * w)
        return "|-" + "-|-".join(parts) + "-|"

    lines = [_row_str({col: col for col in _TABLE_COLS}), _sep_str()]
    for frow in formatted:
        lines.append(_row_str(frow))
    return "\n".join(lines)


def _rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of row dicts to a DataFrame with columns = ``_TABLE_COLS``."""
    return pd.DataFrame(rows, columns=_TABLE_COLS)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_evaluation(
    prices_csv: Path,
    predictions_path: Path | None,
    out_table: Path,
    out_cm_dir: Path | None,
) -> pd.DataFrame:
    """Execute the full evaluation pipeline and return the comparison table.

    Parameters
    ----------
    prices_csv:
        Path to ``global_prices.csv`` (must contain a ``2330.TW`` column).
    predictions_path:
        Optional path to a teammate's predictions CSV
        (``date, proba_down, proba_flat, proba_up, pred_label``).
        When ``None``, only dummies and baselines are evaluated.
    out_table:
        Path to write the CSV version of the comparison table.
    out_cm_dir:
        Optional directory for confusion-matrix PNG files.
        When ``None``, no PNGs are saved.

    Returns
    -------
    pd.DataFrame
        The comparison table with one row per model/baseline.
    """
    # ------------------------------------------------------------------
    # 1. Load prices + labels + split
    # ------------------------------------------------------------------
    print(f"[run_eval] Loading prices from: {prices_csv}")
    close_full: pd.Series = load_tsmc_prices(prices_csv)
    print(
        f"[run_eval] Full close series: {len(close_full)} rows  "
        f"[{close_full.index.min().date()} → {close_full.index.max().date()}]"
    )

    labels: pd.DataFrame = make_labels(close_full)
    _, _, test = temporal_split(labels)

    print(
        f"[run_eval] Test split: {len(test)} rows  "
        f"[{test.index.min().date()} → {test.index.max().date()}]"
    )

    # ------------------------------------------------------------------
    # 2. Build close_filtered: close series from test_first_date onward.
    #
    #    Relationship:
    #      labels.index   = close_full.index[:-1]  (last day dropped)
    #      test.index     = labels.index filtered to > val_end
    #      close_filtered = close_full[close_full.index >= test.index[0]]
    #
    #    This gives len(test) + 1 rows (the extra row is the "next day" after
    #    the last test label date, needed to compute the final return).
    #
    #    signals must be indexed at close_filtered.index[:-1] = test.index.
    # ------------------------------------------------------------------
    test_first_date: pd.Timestamp = test.index[0]
    close_filtered: pd.Series = close_full[close_full.index >= test_first_date].copy()

    # Verify alignment: test.index must equal close_filtered.index[:-1]
    assert test.index.equals(close_filtered.index[:-1]), (
        "Alignment error: test.index does not match close_filtered.index[:-1].\n"
        f"  test.index    : {test.index[[0, -1]].tolist()} (len={len(test)})\n"
        f"  close_filt[:-1]: {close_filtered.index[[-2, -1]].tolist()} "
        f"(len={len(close_filtered)-1})"
    )
    print(
        f"[run_eval] close_filtered: {len(close_filtered)} rows  "
        f"(includes 1 extra trailing date for return computation)"
    )

    # ------------------------------------------------------------------
    # 3. Collect results rows
    # ------------------------------------------------------------------
    rows: list[dict[str, Any]] = []

    # -- (Optional) teammate DAF-Net predictions -----------------------
    if predictions_path is not None:
        print(f"[run_eval] Loading teammate predictions from: {predictions_path}")
        daf_model = CsvPredictionsModel(predictions_path)
        row = _evaluate_ml_model(
            daf_model, test, close_filtered,
            model_name="DAF-Net (teammate)",
            out_cm_dir=out_cm_dir,
        )
        rows.append(row)
        print(
            f"[run_eval]   DAF-Net  Macro F1={row['Macro F1']:.4f}  "
            f"CumReturn={row['CumReturn']:.4f}  n_trades={row['n_trades']}"
        )

    # -- DummyRandom ---------------------------------------------------
    print("[run_eval] Evaluating DummyRandomModel(seed=42)...")
    dummy_rand = DummyRandomModel(seed=42)
    row = _evaluate_ml_model(
        dummy_rand, test, close_filtered,
        model_name="DummyRandom",
        out_cm_dir=out_cm_dir,
    )
    rows.append(row)
    print(
        f"[run_eval]   DummyRandom  Macro F1={row['Macro F1']:.4f}  "
        f"CumReturn={row['CumReturn']:.4f}  n_trades={row['n_trades']}"
    )

    # -- DummyMajority -------------------------------------------------
    print("[run_eval] Evaluating DummyMajorityModel...")
    dummy_maj = DummyMajorityModel()
    row = _evaluate_ml_model(
        dummy_maj, test, close_filtered,
        model_name="DummyMajority",
        out_cm_dir=out_cm_dir,
    )
    rows.append(row)
    print(
        f"[run_eval]   DummyMajority  Macro F1={row['Macro F1']:.4f}  "
        f"CumReturn={row['CumReturn']:.4f}  n_trades={row['n_trades']}"
    )

    # -- B1 Buy & Hold -------------------------------------------------
    print("[run_eval] Evaluating B1 Buy & Hold (mark-to-market)...")
    bh_result = run_buy_and_hold(close_filtered)
    rows.append(_financial_row("B1 Buy & Hold", bh_result))
    print(
        f"[run_eval]   B1 B&H  CumReturn={bh_result['cumulative_return']:.4f}  "
        f"Sharpe={bh_result['sharpe']:.4f}  n_trades={bh_result['n_trades']}"
    )

    # -- B2 Pure-Market (placeholder or real) --------------------------
    if _B2_CSV.exists():
        print(f"[run_eval] Loading B2 Pure-Market predictions from: {_B2_CSV}")
        b2_model = CsvPredictionsModel(_B2_CSV)
        row = _evaluate_ml_model(
            b2_model, test, close_filtered,
            model_name="B2 Pure-Market",
            out_cm_dir=out_cm_dir,
        )
        rows.append(row)
        print(
            f"[run_eval]   B2 Pure-Market  Macro F1={row['Macro F1']:.4f}  "
            f"CumReturn={row['CumReturn']:.4f}  n_trades={row['n_trades']}"
        )
    else:
        print(f"[run_eval] B2 CSV not found at {_B2_CSV} — inserting placeholder row.")
        rows.append(_placeholder_row("B2 Pure-Market", "[B2: pending teammate output]"))

    # -- B3 SMA 5/20 ---------------------------------------------------
    print("[run_eval] Evaluating B3 SMA 5/20...")
    sma_result = run_sma(close_filtered, fast=5, slow=20)
    rows.append(_financial_row("B3 SMA 5/20", sma_result))
    print(
        f"[run_eval]   B3 SMA  CumReturn={sma_result['cumulative_return']:.4f}  "
        f"Sharpe={sma_result['sharpe']:.4f}  n_trades={sma_result['n_trades']}"
    )

    # ------------------------------------------------------------------
    # 4. Render + print table
    # ------------------------------------------------------------------
    table_md = _render_markdown_table(rows)
    print()
    print("=" * 80)
    print("EVALUATION COMPARISON TABLE")
    print("=" * 80)
    print(table_md)
    print("=" * 80)

    # ------------------------------------------------------------------
    # 5. Save CSV
    # ------------------------------------------------------------------
    df = _rows_to_dataframe(rows)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_table, index=False)
    print(f"\n[run_eval] Table saved to: {out_table}")

    if out_cm_dir is not None:
        print(f"[run_eval] Confusion matrices saved to: {out_cm_dir}")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for run_eval.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run_eval",
        description=(
            "Evaluate all models and baselines on the TSMC test split and "
            "produce a single comparison table (ML metrics + financial metrics)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Baselines + dummies only\n"
            "  python -m evaluation.run_eval\n\n"
            "  # With teammate DAF-Net predictions\n"
            "  python -m evaluation.run_eval --predictions path/to/predictions.csv\n\n"
            "  # Save table + confusion-matrix PNGs\n"
            "  python -m evaluation.run_eval "
            "--out-table results.csv --out-cm-dir /tmp/cms\n"
        ),
    )
    parser.add_argument(
        "--predictions",
        metavar="PATH",
        type=Path,
        default=None,
        help=(
            "Path to a teammate's predictions CSV with columns: "
            "date, proba_down, proba_flat, proba_up, pred_label. "
            "When omitted, only baselines + dummies are evaluated."
        ),
    )
    parser.add_argument(
        "--prices-csv",
        metavar="PATH",
        type=Path,
        default=_DEFAULT_PRICES_CSV,
        help=(
            f"Path to global_prices.csv (default: {_DEFAULT_PRICES_CSV}). "
            "Must contain a '2330.TW' column."
        ),
    )
    parser.add_argument(
        "--out-table",
        metavar="PATH",
        type=Path,
        default=_DEFAULT_OUT_TABLE,
        help=(
            f"Path to write the CSV comparison table "
            f"(default: {_DEFAULT_OUT_TABLE}). "
            "Results are always printed to stdout as well."
        ),
    )
    parser.add_argument(
        "--out-cm-dir",
        metavar="DIR",
        type=Path,
        default=None,
        help=(
            "Directory to save confusion-matrix PNG files for each ML-capable "
            "model.  When omitted, no PNGs are saved."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the evaluation pipeline.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]`` when ``None``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate --predictions path if given
    if args.predictions is not None and not args.predictions.exists():
        parser.error(f"--predictions file not found: {args.predictions}")

    run_evaluation(
        prices_csv=args.prices_csv,
        predictions_path=args.predictions,
        out_table=args.out_table,
        out_cm_dir=args.out_cm_dir,
    )


if __name__ == "__main__":
    main()
