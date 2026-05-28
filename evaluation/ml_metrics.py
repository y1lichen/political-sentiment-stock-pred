"""
evaluation/ml_metrics.py
========================
ML classification metrics for the 3-class TSMC return-prediction task.

Label convention (must match evaluation/data_module.py)
--------------------------------------------------------
  0 = 大跌  (next-day return < -1%)
  1 = 盤整  (next-day return in [-1%, +1%])
  2 = 大漲  (next-day return > +1%)

Public API
----------
  confusion_matrix_report(y_true, y_pred, *, plot_path=None) -> dict
  precision_per_class(y_true, y_pred) -> dict[str, float]
  macro_f1(y_true, y_pred) -> float
  full_report(y_true, y_pred, y_proba=None) -> dict

All functions validate their inputs before computing any metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
)

# Canonical label order used across the entire pipeline
_LABEL_INTS: list[int] = [0, 1, 2]
_LABEL_NAMES: list[str] = ["大跌", "盤整", "大漲"]
_VALID_VALUES: frozenset[int] = frozenset({0, 1, 2})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to 1-D int arrays and run boundary checks.

    Parameters
    ----------
    y_true:
        Ground-truth labels, array-like with values in {0, 1, 2}.
    y_pred:
        Predicted labels, array-like with values in {0, 1, 2}.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(y_true_arr, y_pred_arr)`` — 1-D numpy int arrays.

    Raises
    ------
    ValueError
        If inputs are empty, shapes mismatch, are not 1-D, or contain
        values outside {0, 1, 2}.
    """
    y_true_arr = np.asarray(y_true, dtype=np.intp)
    y_pred_arr = np.asarray(y_pred, dtype=np.intp)

    if y_true_arr.ndim != 1:
        raise ValueError(
            f"y_true must be 1-D, got shape {y_true_arr.shape}."
        )
    if y_pred_arr.ndim != 1:
        raise ValueError(
            f"y_pred must be 1-D, got shape {y_pred_arr.shape}."
        )
    if len(y_true_arr) == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, "
            f"got {y_true_arr.shape} vs {y_pred_arr.shape}."
        )

    # Domain check: values must be in {0, 1, 2}
    bad_true = np.unique(y_true_arr[~np.isin(y_true_arr, list(_VALID_VALUES))])
    bad_pred = np.unique(y_pred_arr[~np.isin(y_pred_arr, list(_VALID_VALUES))])
    if len(bad_true) > 0 or len(bad_pred) > 0:
        offenders: list[str] = []
        if len(bad_true) > 0:
            offenders.append(f"y_true contains invalid values: {bad_true.tolist()}")
        if len(bad_pred) > 0:
            offenders.append(f"y_pred contains invalid values: {bad_pred.tolist()}")
        raise ValueError(
            "Label values must be in {0, 1, 2}. "
            + "; ".join(offenders) + "."
        )

    return y_true_arr, y_pred_arr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def confusion_matrix_report(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
    *,
    plot_path: str | Path | None = None,
) -> dict:
    """Compute the 3×3 confusion matrix and flag the fatal-error cell.

    The fatal-error cell is [0, 2]: true=大跌 but predicted=大漲. In a
    long-only strategy this means the model signals "buy" on a day where the
    stock actually falls sharply — the worst possible financial outcome.

    Parameters
    ----------
    y_true:
        Ground-truth labels, 1-D array-like with values in {0, 1, 2}.
    y_pred:
        Predicted labels, 1-D array-like with values in {0, 1, 2}.
    plot_path:
        Optional path (str or Path) to save a matplotlib heatmap PNG.
        The fatal-error cell [0, 2] is highlighted with a red rectangle.
        ``plt.show()`` is **never** called.

    Returns
    -------
    dict with keys:
        ``"matrix"``            : np.ndarray of shape (3, 3).
        ``"row_labels"``        : list[str] — true-class labels.
        ``"col_labels"``        : list[str] — predicted-class labels.
        ``"fatal_error_count"`` : int — count of true=大跌, pred=大漲.
        ``"fatal_error_rate"``  : float — rate within true-大跌 rows.
    """
    y_true_arr, y_pred_arr = _validate_inputs(y_true, y_pred)

    cm: np.ndarray = confusion_matrix(
        y_true_arr, y_pred_arr, labels=_LABEL_INTS
    )

    # Fatal error: row 0 (true=大跌), col 2 (pred=大漲)
    fatal_count = int(cm[0, 2])
    true_down_total = int(cm[0].sum())
    fatal_rate = fatal_count / max(true_down_total, 1)

    result: dict = {
        "matrix": cm,
        "row_labels": [f"true_{name}" for name in _LABEL_NAMES],
        "col_labels": [f"pred_{name}" for name in _LABEL_NAMES],
        "fatal_error_count": fatal_count,
        "fatal_error_rate": fatal_rate,
    }

    if plot_path is not None:
        _save_confusion_matrix_plot(cm, plot_path)

    return result


def _save_confusion_matrix_plot(
    cm: np.ndarray,
    plot_path: str | Path,
) -> None:
    """Save a heatmap of the confusion matrix to *plot_path*.

    The fatal-error cell (row 0, col 2) is highlighted with a red rectangle
    and its text is rendered in red to draw the reviewer's attention.

    Parameters
    ----------
    cm:
        3×3 confusion-matrix array.
    plot_path:
        Filesystem path for the output PNG (parent directory must exist or
        will be created).
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; never calls plt.show()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw heatmap manually so we can control individual cell text colours
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate every cell
    cm_max = cm.max() if cm.max() > 0 else 1
    for row in range(3):
        for col in range(3):
            val = cm[row, col]
            text_color = "red" if (row == 0 and col == 2) else (
                "white" if val > cm_max * 0.5 else "black"
            )
            ax.text(
                col, row, str(val),
                ha="center", va="center",
                color=text_color,
                fontsize=13,
                fontweight="bold" if (row == 0 and col == 2) else "normal",
            )

    # Red rectangle border around the fatal-error cell
    rect = mpatches.Rectangle(
        (2 - 0.5, 0 - 0.5),  # (x=col-0.5, y=row-0.5)
        width=1.0,
        height=1.0,
        linewidth=3,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)

    # Axes labels and ticks
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(
        [f"pred_{name}" for name in _LABEL_NAMES],
        fontsize=11,
    )
    ax.set_yticklabels(
        [f"true_{name}" for name in _LABEL_NAMES],
        fontsize=11,
    )
    ax.set_xlabel("Predicted label", fontsize=12, labelpad=8)
    ax.set_ylabel("True label", fontsize=12, labelpad=8)
    ax.set_title(
        "Confusion Matrix\n"
        "(red cell = fatal error: true=大跌 predicted as 大漲)",
        fontsize=12,
        pad=12,
    )

    plt.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


def precision_per_class(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
) -> dict[str, float]:
    """Compute per-class precision for the 3-label problem.

    Uses ``sklearn.metrics.precision_score`` with ``zero_division=0.0`` so
    that classes absent from ``y_pred`` receive precision 0 rather than
    raising a warning or returning NaN.

    Parameters
    ----------
    y_true:
        Ground-truth labels, 1-D array-like with values in {0, 1, 2}.
    y_pred:
        Predicted labels, 1-D array-like with values in {0, 1, 2}.

    Returns
    -------
    dict[str, float]
        Keys: ``"precision_大跌"``, ``"precision_盤整"``, ``"precision_大漲"``.
        Values in [0.0, 1.0].
    """
    y_true_arr, y_pred_arr = _validate_inputs(y_true, y_pred)

    scores: np.ndarray = precision_score(
        y_true_arr,
        y_pred_arr,
        labels=_LABEL_INTS,
        average=None,
        zero_division=0.0,
    )

    return {
        f"precision_{name}": float(scores[i])
        for i, name in enumerate(_LABEL_NAMES)
    }


def macro_f1(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
) -> float:
    """Compute macro-averaged F1 score over the three classes.

    Uses ``sklearn.metrics.f1_score`` with ``zero_division=0.0`` so classes
    absent from both ``y_true`` and ``y_pred`` contribute 0.0 to the macro
    average rather than producing NaN or a warning.

    Parameters
    ----------
    y_true:
        Ground-truth labels, 1-D array-like with values in {0, 1, 2}.
    y_pred:
        Predicted labels, 1-D array-like with values in {0, 1, 2}.

    Returns
    -------
    float
        Macro F1 in [0.0, 1.0].
    """
    y_true_arr, y_pred_arr = _validate_inputs(y_true, y_pred)

    return float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            labels=_LABEL_INTS,
            average="macro",
            zero_division=0.0,
        )
    )


def full_report(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
    y_proba: np.ndarray | None = None,
) -> dict:
    """Aggregate all classification metrics into a single flat dict.

    This is the primary output consumed by ``run_eval.py`` when building the
    comparison table.  ``y_proba`` is accepted for forward-compatibility but
    is not used in the current implementation (reserved for AUC-ROC once
    multi-class ROC is added).

    Parameters
    ----------
    y_true:
        Ground-truth labels, 1-D array-like with values in {0, 1, 2}.
    y_pred:
        Predicted labels, 1-D array-like with values in {0, 1, 2}.
    y_proba:
        Optional array of shape ``(N, 3)`` with per-class probabilities.
        Accepted but currently unused; reserved for future AUC metrics.

    Returns
    -------
    dict with keys:
        ``"macro_f1"``          : float
        ``"precision_大跌"``    : float
        ``"precision_盤整"``    : float
        ``"precision_大漲"``    : float
        ``"support_大跌"``      : int  — count of class 0 in y_true
        ``"support_盤整"``      : int  — count of class 1 in y_true
        ``"support_大漲"``      : int  — count of class 2 in y_true
        ``"confusion"``         : dict — output of confusion_matrix_report
                                         (without plot_path)
        ``"n_samples"``         : int
    """
    y_true_arr, y_pred_arr = _validate_inputs(y_true, y_pred)

    mf1 = macro_f1(y_true_arr, y_pred_arr)
    prec = precision_per_class(y_true_arr, y_pred_arr)
    cm_info = confusion_matrix_report(y_true_arr, y_pred_arr, plot_path=None)

    # Support counts (number of actual samples per class)
    support: dict[str, int] = {}
    for i, name in enumerate(_LABEL_NAMES):
        support[f"support_{name}"] = int(np.sum(y_true_arr == i))

    return {
        "macro_f1": mf1,
        **prec,
        **support,
        "confusion": cm_info,
        "n_samples": int(len(y_true_arr)),
    }


# ---------------------------------------------------------------------------
# Self-verification smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    SEPARATOR = "=" * 62

    print(SEPARATOR)
    print("Smoke test: evaluation/ml_metrics.py")
    print(SEPARATOR)

    # ------------------------------------------------------------------
    # Build synthetic y_true: 100 samples — 20 大跌, 60 盤整, 20 大漲
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    y_true_base = np.array([0] * 20 + [1] * 60 + [2] * 20, dtype=np.intp)
    rng.shuffle(y_true_base)  # shuffle to simulate realistic ordering

    print(f"\nSynthetic y_true: {len(y_true_base)} samples")
    print(f"  大跌(0): {(y_true_base == 0).sum()}  "
          f"盤整(1): {(y_true_base == 1).sum()}  "
          f"大漲(2): {(y_true_base == 2).sum()}")

    # ------------------------------------------------------------------
    # Case A: perfect predictor — y_pred == y_true
    # ------------------------------------------------------------------
    print(f"\n{'-'*50}")
    print("Case A: Perfect predictor (y_pred == y_true)")
    y_pred_A = y_true_base.copy()

    mf1_A = macro_f1(y_true_base, y_pred_A)
    prec_A = precision_per_class(y_true_base, y_pred_A)
    cm_A = confusion_matrix_report(y_true_base, y_pred_A)

    print(f"  macro_f1         = {mf1_A:.6f}  (expected 1.0)")
    print(f"  precision_大跌   = {prec_A['precision_大跌']:.6f}  (expected 1.0)")
    print(f"  precision_盤整   = {prec_A['precision_盤整']:.6f}  (expected 1.0)")
    print(f"  precision_大漲   = {prec_A['precision_大漲']:.6f}  (expected 1.0)")
    print(f"  confusion matrix:\n{cm_A['matrix']}")
    print(f"  fatal_error_count = {cm_A['fatal_error_count']}  (expected 0)")

    assert abs(mf1_A - 1.0) < 1e-9, f"Case A: macro_f1 should be 1.0, got {mf1_A}"
    assert abs(prec_A["precision_大跌"] - 1.0) < 1e-9, "Case A: precision_大跌 != 1.0"
    assert abs(prec_A["precision_盤整"] - 1.0) < 1e-9, "Case A: precision_盤整 != 1.0"
    assert abs(prec_A["precision_大漲"] - 1.0) < 1e-9, "Case A: precision_大漲 != 1.0"
    # Confusion matrix must be diagonal
    cm_arr = cm_A["matrix"]
    off_diag_sum = cm_arr.sum() - np.trace(cm_arr)
    assert off_diag_sum == 0, f"Case A: confusion matrix is not diagonal, off-diag sum = {off_diag_sum}"
    assert cm_A["fatal_error_count"] == 0, "Case A: fatal_error_count != 0"
    print("  [PASSED]")

    # ------------------------------------------------------------------
    # Case B: DummyMajority — always predicts 1 (盤整)
    # ------------------------------------------------------------------
    print(f"\n{'-'*50}")
    print("Case B: DummyMajority (y_pred == 1 everywhere)")
    y_pred_B = np.ones(len(y_true_base), dtype=np.intp)

    mf1_B = macro_f1(y_true_base, y_pred_B)
    prec_B = precision_per_class(y_true_base, y_pred_B)
    cm_B = confusion_matrix_report(y_true_base, y_pred_B)

    print(f"  macro_f1         = {mf1_B:.6f}")
    print(f"  precision_大跌   = {prec_B['precision_大跌']:.6f}  (expected 0.0)")
    print(f"  precision_大漲   = {prec_B['precision_大漲']:.6f}  (expected 0.0)")
    print(f"  fatal_error_count = {cm_B['fatal_error_count']}  (expected 0, because pred never == 2)")

    assert abs(prec_B["precision_大漲"] - 0.0) < 1e-9, "Case B: precision_大漲 should be 0.0"
    assert abs(prec_B["precision_大跌"] - 0.0) < 1e-9, "Case B: precision_大跌 should be 0.0"
    assert cm_B["fatal_error_count"] == 0, (
        f"Case B: fatal_error_count should be 0 (no pred==2), got {cm_B['fatal_error_count']}"
    )
    print("  [PASSED]")

    # ------------------------------------------------------------------
    # Case C: worst-case — all 大跌 (0) predicted as 大漲 (2)
    # ------------------------------------------------------------------
    print(f"\n{'-'*50}")
    print("Case C: Worst case — all true-大跌 predicted as 大漲")
    y_pred_C = y_true_base.copy()
    y_pred_C[y_true_base == 0] = 2  # flip every 大跌 → 大漲

    cm_C = confusion_matrix_report(y_true_base, y_pred_C)
    print(f"  confusion matrix:\n{cm_C['matrix']}")
    print(f"  fatal_error_count = {cm_C['fatal_error_count']}  (expected 20)")
    print(f"  fatal_error_rate  = {cm_C['fatal_error_rate']:.6f}  (expected 1.0)")

    assert cm_C["fatal_error_count"] == 20, (
        f"Case C: fatal_error_count expected 20, got {cm_C['fatal_error_count']}"
    )
    assert abs(cm_C["fatal_error_rate"] - 1.0) < 1e-9, (
        f"Case C: fatal_error_rate expected 1.0, got {cm_C['fatal_error_rate']}"
    )
    print("  [PASSED]")

    # ------------------------------------------------------------------
    # Case D: invalid input — y_pred contains value 3
    # ------------------------------------------------------------------
    print(f"\n{'-'*50}")
    print("Case D: Invalid input — y_pred contains value 3")
    y_pred_D = y_true_base.copy()
    y_pred_D[0] = 3  # inject bad value

    caught_D = False
    try:
        _ = confusion_matrix_report(y_true_base, y_pred_D)
    except ValueError as exc:
        caught_D = True
        print(f"  ValueError raised (expected). Message: {exc}")
    assert caught_D, "Case D: expected ValueError for y_pred value=3, but none was raised"
    print("  [PASSED]")

    # ------------------------------------------------------------------
    # Case E: PNG confusion matrix — save and verify
    # ------------------------------------------------------------------
    print(f"\n{'-'*50}")
    print("Case E: Confusion matrix PNG saved to /tmp/_ml_metrics_test_cm.png")
    png_path = Path("/tmp/_ml_metrics_test_cm.png")

    # Use Case C data (most interesting matrix)
    cm_E = confusion_matrix_report(y_true_base, y_pred_C, plot_path=png_path)

    assert png_path.exists(), "Case E: PNG file was not created"
    file_size = png_path.stat().st_size
    assert file_size > 0, "Case E: PNG file is empty"
    print(f"  File exists: {png_path}")
    print(f"  File size  : {file_size:,} bytes")
    print("  [PASSED]")

    # ------------------------------------------------------------------
    # full_report round-trip
    # ------------------------------------------------------------------
    print(f"\n{'-'*50}")
    print("full_report round-trip check (Case C data)")
    report = full_report(y_true_base, y_pred_C)
    expected_keys = {
        "macro_f1", "precision_大跌", "precision_盤整", "precision_大漲",
        "support_大跌", "support_盤整", "support_大漲", "confusion", "n_samples",
    }
    missing_keys = expected_keys - set(report.keys())
    assert not missing_keys, f"full_report missing keys: {missing_keys}"
    assert report["n_samples"] == 100, f"n_samples expected 100, got {report['n_samples']}"
    assert report["support_大跌"] == 20, f"support_大跌 expected 20, got {report['support_大跌']}"
    assert report["support_盤整"] == 60, f"support_盤整 expected 60, got {report['support_盤整']}"
    assert report["support_大漲"] == 20, f"support_大漲 expected 20, got {report['support_大漲']}"
    print(f"  Keys     : {sorted(report.keys())}")
    print(f"  n_samples: {report['n_samples']}")
    print(f"  macro_f1 : {report['macro_f1']:.4f}")
    print(f"  support  : 大跌={report['support_大跌']}  盤整={report['support_盤整']}  大漲={report['support_大漲']}")
    print("  [PASSED]")

    print(f"\n{SEPARATOR}")
    print("All smoke tests PASSED.")
    print(SEPARATOR)
