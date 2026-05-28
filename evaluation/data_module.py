"""
evaluation/data_module.py
=========================
Single source of truth for TSMC price loading, label construction, and
temporal splitting.  All public functions are pure (no global state).

Label convention
----------------
  0 = 大跌  (next-day return < -threshold)
  1 = 盤整  (|next-day return| <= threshold)
  2 = 大漲  (next-day return > +threshold)

Temporal split defaults (overridable via parameters)
------------------------------------------------------
  train : 2017-01-03 – 2023-12-31
  val   : 2024-01-01 – 2024-12-31
  test  : 2025-01-01 – 2026-05-15  (or whatever the data end-date is)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level constant: consistent label display order used everywhere
# ---------------------------------------------------------------------------
LABEL_NAMES: list[str] = ["大跌", "盤整", "大漲"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_tsmc_prices(path: str | Path) -> pd.Series:
    """Load daily close prices for 2330.TW from the global_prices CSV.

    Parameters
    ----------
    path:
        Path to ``data/taiwan_market_data/global_prices.csv``.
        The CSV must have a ``Date`` column (or index) and a ``2330.TW`` column.

    Returns
    -------
    pd.Series
        Indexed by ``pd.DatetimeIndex`` named ``"Date"``, values are float
        close prices.  Series name is ``"2330.TW"``.  The single NaN row at
        2017-01-02 is dropped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"global_prices.csv not found at: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"

    if "2330.TW" not in df.columns:
        raise KeyError(
            f"Column '2330.TW' not found in {path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    series: pd.Series = df["2330.TW"].copy()
    series.name = "2330.TW"

    # Drop NaN rows (only 2017-01-02 expected; guard if more appear)
    n_nan = series.isna().sum()
    series = series.dropna()

    assert isinstance(series.index, pd.DatetimeIndex), (
        "Index must be DatetimeIndex after parsing."
    )
    assert series.isna().sum() == 0, "NaN values remain after dropna — unexpected."
    assert len(series) > 0, "Series is empty after dropping NaN rows."

    if n_nan > 5:  # soft warning; dataset has exactly 1 expected NaN
        import warnings
        warnings.warn(
            f"Dropped {n_nan} NaN rows from 2330.TW (expected ~1).",
            UserWarning,
            stacklevel=2,
        )

    return series


def make_labels(
    close: pd.Series,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """Build a label DataFrame from a close-price series.

    The label is based on the **next-day** return relative to a fixed
    ±threshold:

    * ``next_return > +threshold``  → label 2 (大漲)
    * ``next_return < -threshold``  → label 0 (大跌)
    * otherwise                     → label 1 (盤整)

    The last row is dropped because its next-day return is undefined.

    Parameters
    ----------
    close:
        Daily close prices indexed by ``pd.DatetimeIndex``.  Must contain no
        NaN values (call :func:`load_tsmc_prices` first).
    threshold:
        Fractional return threshold (default ``0.01`` = 1 %).

    Returns
    -------
    pd.DataFrame
        Indexed by the same ``DatetimeIndex`` as ``close`` (minus last row),
        with columns:

        - ``close``       : float — the close price on that date
        - ``next_return`` : float — ``close_{t+1} / close_t - 1``
        - ``label``       : int8  — 0, 1, or 2
    """
    if close.isna().any():
        raise ValueError(
            "close series contains NaN values; call load_tsmc_prices() first."
        )
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close.index must be a pd.DatetimeIndex.")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}.")
    if len(close) < 2:
        raise ValueError("close series must have at least 2 rows to compute labels.")

    # INDICATOR: compute next-day return (NaN for last row)
    next_return: pd.Series = close.shift(-1) / close - 1

    # Drop last row (NaN next_return) before labeling
    valid_idx = next_return.dropna().index
    close_out = close.loc[valid_idx]
    ret_out = next_return.loc[valid_idx]

    # Vectorised label assignment
    raw_label = np.where(
        ret_out > threshold,
        2,
        np.where(ret_out < -threshold, 0, 1),
    )
    label_series = pd.Series(raw_label, index=valid_idx, dtype=np.int8, name="label")

    df = pd.DataFrame(
        {
            "close": close_out,
            "next_return": ret_out,
            "label": label_series,
        },
        index=valid_idx,
    )

    # Boundary checks — must hold before returning
    assert df["close"].isna().sum() == 0, "NaN in 'close' column."
    assert df["next_return"].isna().sum() == 0, "NaN in 'next_return' column."
    assert df["label"].isna().sum() == 0, "NaN in 'label' column."
    assert set(df["label"].unique()).issubset({0, 1, 2}), (
        f"Unexpected label values: {df['label'].unique()}"
    )
    assert len(df) == len(close) - 1, (
        f"Expected {len(close)-1} rows after dropping last row, got {len(df)}."
    )

    return df


def temporal_split(
    df: pd.DataFrame,
    train_end: str = "2023-12-31",
    val_end: str = "2024-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a date-indexed DataFrame into non-overlapping train/val/test folds.

    The split is purely by calendar date (inclusive on both ends).  No
    shuffling is performed to preserve temporal ordering (lookahead-free).

    Parameters
    ----------
    df:
        DataFrame indexed by ``pd.DatetimeIndex``.
    train_end:
        Last date (inclusive) of the training fold.  Default ``"2023-12-31"``.
    val_end:
        Last date (inclusive) of the validation fold.  Default ``"2024-12-31"``.
        Everything after ``val_end`` becomes the test fold.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train, val, test)`` — each a slice of the original DataFrame with
        no overlapping indices.

    Raises
    ------
    ValueError
        If any split is empty, if splits overlap, or if their combined length
        does not equal the input length.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a pd.DatetimeIndex.")

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    if train_end_ts >= val_end_ts:
        raise ValueError(
            f"train_end ({train_end}) must be strictly before val_end ({val_end})."
        )

    train = df[df.index <= train_end_ts]
    val = df[(df.index > train_end_ts) & (df.index <= val_end_ts)]
    test = df[df.index > val_end_ts]

    # Emptiness checks
    if len(train) == 0:
        raise ValueError(f"train split is empty (train_end={train_end}).")
    if len(val) == 0:
        raise ValueError(f"val split is empty (train_end={train_end}, val_end={val_end}).")
    if len(test) == 0:
        raise ValueError(f"test split is empty (val_end={val_end}).")

    # Overlap check — guaranteed by construction but assert defensively
    assert train.index.max() < val.index.min(), (
        "Overlap detected between train and val splits."
    )
    assert val.index.max() < test.index.min(), (
        "Overlap detected between val and test splits."
    )

    # Length conservation
    assert len(train) + len(val) + len(test) == len(df), (
        f"Split lengths {len(train)}+{len(val)}+{len(test)} != total {len(df)}."
    )

    return train, val, test


# ---------------------------------------------------------------------------
# Smoke test (run with: python -m evaluation.data_module)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    # Locate the data file relative to repo root (two levels up from this file)
    _this_dir = Path(__file__).resolve().parent
    _repo_root = _this_dir.parent
    _prices_path = _repo_root / "data" / "taiwan_market_data" / "global_prices.csv"

    print(f"Loading prices from: {_prices_path}")
    close = load_tsmc_prices(_prices_path)
    print(f"  Series length : {len(close)}")
    print(f"  Date range    : {close.index.min().date()} → {close.index.max().date()}")
    print(f"  NaN count     : {close.isna().sum()}")
    print()

    print("Building labels (threshold=±1%)...")
    labeled = make_labels(close, threshold=0.01)
    total_rows = len(labeled)
    counts = labeled["label"].value_counts().sort_index()
    print(f"  Total rows    : {total_rows}")
    print("  Label distribution:")
    for lbl_int, lbl_name in enumerate(LABEL_NAMES):
        n = counts.get(lbl_int, 0)
        pct = 100 * n / total_rows
        print(f"    [{lbl_int}] {lbl_name:>3s} : {n:5d}  ({pct:.1f}%)")
    print()

    print("Splitting (train≤2023-12-31, val≤2024-12-31, test=rest)...")
    train, val, test = temporal_split(labeled)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(
            f"  {name:5s}: {len(split):5d} rows  "
            f"[{split.index.min().date()} → {split.index.max().date()}]"
        )
    print()
    print("All checks passed.")
