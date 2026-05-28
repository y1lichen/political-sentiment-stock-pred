"""
evaluation/model_interface.py
------------------------------
Defines the abstract interface (ClassifierModel) and concrete implementations
(DummyRandomModel, DummyMajorityModel, CsvPredictionsModel) used across the
Trump-sentiment × TSMC validation pipeline.

Label convention (three-class):
    0 = 大跌 (down  > -1%)
    1 = 盤整 (flat  in [-1%, +1%])
    2 = 大漲 (up    > +1%)

predict_proba returns column order: [proba_down, proba_flat, proba_up]
"""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class ClassifierModel(ABC):
    """Abstract contract for every model in the validation pipeline.

    All concrete implementations must handle a ``pd.DatetimeIndex`` of
    *trading dates* and return predictions aligned to those dates.

    Date alignment note
    -------------------
    A model is expected to have been queried *only* for dates it was trained /
    tested on. Passing a date that is absent from the model's internal state
    should raise a descriptive ``KeyError`` listing every offending date so the
    caller can diagnose time-alignment bugs immediately.
    """

    @abstractmethod
    def predict_proba(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Return per-class probability estimates.

        Parameters
        ----------
        dates:
            Trading dates for which predictions are requested.

        Returns
        -------
        np.ndarray of shape ``(len(dates), 3)`` with columns
        ``[proba_down, proba_flat, proba_up]``. Each row sums to ~1.0
        (within floating-point tolerance). Values are in ``[0.0, 1.0]``.
        """

    @abstractmethod
    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Return discrete class labels.

        Parameters
        ----------
        dates:
            Trading dates for which predictions are requested.

        Returns
        -------
        np.ndarray of shape ``(len(dates),)`` with integer values in
        ``{0, 1, 2}``.
        """


# ---------------------------------------------------------------------------
# DummyRandomModel
# ---------------------------------------------------------------------------

class DummyRandomModel(ClassifierModel):
    """Uniform-random 3-class model for sanity checks and baseline comparisons.

    Draws each row from a symmetric Dirichlet distribution (concentration
    parameter = 1 for all classes) so the marginal expectation for every class
    is 1/3. The seed is fixed at construction time, making results fully
    reproducible across multiple calls with the *same* dates.

    Parameters
    ----------
    seed:
        Integer seed passed to ``numpy.random.default_rng``. Default: 42.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def predict_proba(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Draw Dirichlet([1, 1, 1]) probabilities, one row per date.

        Returns
        -------
        np.ndarray of shape ``(N, 3)``.
        """
        n = len(dates)
        rng = np.random.default_rng(self._seed)
        # Dirichlet([1,1,1]) is uniform over the 3-simplex
        proba = rng.dirichlet(alpha=[1.0, 1.0, 1.0], size=n)
        return proba.astype(np.float64)

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Return argmax of ``predict_proba`` as integer labels.

        Returns
        -------
        np.ndarray of shape ``(N,)`` with values in ``{0, 1, 2}``.
        """
        return np.argmax(self.predict_proba(dates), axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# DummyMajorityModel
# ---------------------------------------------------------------------------

class DummyMajorityModel(ClassifierModel):
    """Always predicts class 1 (盤整 / flat).

    Useful for demonstrating the class-imbalance pathology: a model that never
    signals 大漲 will have Precision(大漲) = 0 and CumReturn = 0 in a
    long-only strategy.
    """

    def predict_proba(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Return ``[0.0, 1.0, 0.0]`` for every date.

        Returns
        -------
        np.ndarray of shape ``(N, 3)``.
        """
        n = len(dates)
        proba = np.zeros((n, 3), dtype=np.float64)
        proba[:, 1] = 1.0
        return proba

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Return class ``1`` for every date.

        Returns
        -------
        np.ndarray of shape ``(N,)`` filled with ``1``.
        """
        return np.ones(len(dates), dtype=np.int64)


# ---------------------------------------------------------------------------
# CsvPredictionsModel
# ---------------------------------------------------------------------------

class CsvPredictionsModel(ClassifierModel):
    """Adapter that loads a teammate's ``predictions.csv`` and serves lookups.

    Expected CSV columns
    --------------------
    ``date``, ``proba_down``, ``proba_flat``, ``proba_up``, ``pred_label``

    The ``pred_label`` column encodes the teammate's *actual* decision
    (which may reflect threshold adjustment or confidence filtering) and is
    returned verbatim by ``predict()`` — **argmax is never applied**.

    Parameters
    ----------
    predictions_csv_path:
        Path (str or ``pathlib.Path``) to the predictions CSV file.

    Raises
    ------
    FileNotFoundError
        If ``predictions_csv_path`` does not exist.
    ValueError
        If required columns are missing, or if ``pred_label`` contains values
        outside ``{0, 1, 2}``.
    """

    REQUIRED_COLUMNS: tuple[str, ...] = (
        "date", "proba_down", "proba_flat", "proba_up", "pred_label"
    )

    def __init__(self, predictions_csv_path: Union[str, Path]) -> None:
        path = Path(predictions_csv_path)
        if not path.exists():
            raise FileNotFoundError(
                f"predictions CSV not found: {path.resolve()}"
            )

        df = pd.read_csv(path, parse_dates=["date"])

        # --- column presence check ---
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"predictions CSV is missing required column(s): {missing_cols}. "
                f"Found columns: {list(df.columns)}"
            )

        # --- pred_label domain check ---
        invalid_labels = df["pred_label"][~df["pred_label"].isin({0, 1, 2})]
        if not invalid_labels.empty:
            bad_values = sorted(invalid_labels.unique().tolist())
            raise ValueError(
                f"pred_label must be in {{0, 1, 2}} but found unexpected values: "
                f"{bad_values}. Check rows: {invalid_labels.index.tolist()[:10]}"
            )

        df["pred_label"] = df["pred_label"].astype(np.int64)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date").sort_index()

        self._df = df

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _check_dates(self, dates: pd.DatetimeIndex) -> None:
        """Raise a descriptive KeyError if any date is absent from the CSV.

        This is the primary guard against date-misalignment bugs (e.g., a
        weekend date slipping in, or a holiday not present in the predictions).
        """
        normalized = dates.normalize()
        missing = normalized.difference(self._df.index)
        if len(missing) > 0:
            raise KeyError(
                f"The following {len(missing)} date(s) are not present in the "
                f"predictions CSV:\n  {missing.tolist()}\n"
                "Common causes: (1) querying weekend/holiday dates, "
                "(2) test-set split mismatch, "
                "(3) predictions CSV covers a different date range."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_proba(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Look up ``[proba_down, proba_flat, proba_up]`` rows by date.

        Parameters
        ----------
        dates:
            Trading dates to query.

        Returns
        -------
        np.ndarray of shape ``(N, 3)``.

        Raises
        ------
        KeyError
            If any date in ``dates`` is not found in the CSV, with a
            descriptive message listing all missing dates.
        """
        self._check_dates(dates)
        normalized = dates.normalize()
        rows = self._df.loc[normalized, ["proba_down", "proba_flat", "proba_up"]]
        return rows.to_numpy(dtype=np.float64)

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Return the ``pred_label`` column directly (no argmax applied).

        The teammate's ``pred_label`` may reflect custom thresholding or
        confidence filtering. Respecting it verbatim avoids silently
        overriding their decision logic.

        Parameters
        ----------
        dates:
            Trading dates to query.

        Returns
        -------
        np.ndarray of shape ``(N,)`` with integer values in ``{0, 1, 2}``.

        Raises
        ------
        KeyError
            If any date in ``dates`` is not found in the CSV.
        """
        self._check_dates(dates)
        normalized = dates.normalize()
        labels = self._df.loc[normalized, "pred_label"]
        return labels.to_numpy(dtype=np.int64)


# ---------------------------------------------------------------------------
# Smoke test  (run with:  python -m evaluation.model_interface)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    SEPARATOR = "-" * 60

    print(SEPARATOR)
    print("Smoke test: evaluation/model_interface.py")
    print(SEPARATOR)

    # ------------------------------------------------------------------
    # 1. Build a fake DatetimeIndex of 10 business days
    # ------------------------------------------------------------------
    dates_10 = pd.bdate_range(start="2025-01-02", periods=10)
    print(f"\n[1] DatetimeIndex (10 business days): {dates_10.date.tolist()}")

    # ------------------------------------------------------------------
    # 2. DummyRandomModel and DummyMajorityModel
    # ------------------------------------------------------------------
    rand_model = DummyRandomModel(seed=0)
    maj_model = DummyMajorityModel()

    rp = rand_model.predict_proba(dates_10)
    rl = rand_model.predict(dates_10)
    print(f"\n[2a] DummyRandomModel(seed=0).predict_proba -> shape={rp.shape}, first row={rp[0]}")
    print(f"     DummyRandomModel(seed=0).predict       -> shape={rl.shape}, first 5={rl[:5]}")
    assert rp.shape == (10, 3), "predict_proba shape mismatch"
    assert rl.shape == (10,), "predict shape mismatch"
    assert np.all((rl >= 0) & (rl <= 2)), "predict values out of {0,1,2}"

    mp = maj_model.predict_proba(dates_10)
    ml = maj_model.predict(dates_10)
    print(f"\n[2b] DummyMajorityModel.predict_proba -> shape={mp.shape}, first row={mp[0]}")
    print(f"     DummyMajorityModel.predict       -> shape={ml.shape}, all ones={np.all(ml == 1)}")
    assert mp.shape == (10, 3), "predict_proba shape mismatch"
    assert np.all(mp[:, 1] == 1.0), "DummyMajority should always predict class 1"
    assert np.all(ml == 1), "DummyMajority predict must all be 1"

    # ------------------------------------------------------------------
    # 3. CsvPredictionsModel with a 5-row tempfile
    # ------------------------------------------------------------------
    dates_5 = pd.bdate_range(start="2025-01-02", periods=5)
    fake_data = pd.DataFrame({
        "date": dates_5.strftime("%Y-%m-%d"),
        "proba_down": [0.1, 0.2, 0.3, 0.4, 0.5],
        "proba_flat": [0.6, 0.5, 0.4, 0.3, 0.2],
        "proba_up":   [0.3, 0.3, 0.3, 0.3, 0.3],
        "pred_label": [2, 1, 0, 1, 2],
    })

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    try:
        os.close(tmp_fd)
        fake_data.to_csv(tmp_path, index=False)

        csv_model = CsvPredictionsModel(tmp_path)
        query_dates = pd.DatetimeIndex(dates_5[:3])
        cp = csv_model.predict_proba(query_dates)
        cl = csv_model.predict(query_dates)
        print(f"\n[3] CsvPredictionsModel queried 3 dates: {query_dates.date.tolist()}")
        print(f"    predict_proba -> shape={cp.shape}")
        print(f"    predict_proba[0] = {cp[0]}")
        print(f"    predict       -> {cl}")
        assert cp.shape == (3, 3), "CsvPredictionsModel predict_proba shape mismatch"
        assert cl.tolist() == [2, 1, 0], f"Expected [2,1,0], got {cl.tolist()}"

    finally:
        os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # 4. Reproducibility: same seed → identical output
    # ------------------------------------------------------------------
    model_a = DummyRandomModel(seed=0)
    model_b = DummyRandomModel(seed=0)
    pa = model_a.predict_proba(dates_10)
    pb = model_b.predict_proba(dates_10)
    assert np.allclose(pa, pb), "Reproducibility failure: same seed produced different output"
    print("\n[4] Reproducibility check PASSED (seed=0 → identical predict_proba)")

    # ------------------------------------------------------------------
    # 5. CsvPredictionsModel raises KeyError for a missing date
    # ------------------------------------------------------------------
    tmp_fd2, tmp_path2 = tempfile.mkstemp(suffix=".csv")
    try:
        os.close(tmp_fd2)
        fake_data.to_csv(tmp_path2, index=False)
        csv_model2 = CsvPredictionsModel(tmp_path2)

        missing_date = pd.DatetimeIndex(["2099-01-01"])  # definitely not in CSV
        caught = False
        try:
            _ = csv_model2.predict_proba(missing_date)
        except KeyError as exc:
            caught = True
            print(f"\n[5] KeyError raised for missing date (PASSED). Message snippet:")
            print(f"    {str(exc)[:200]}")
        assert caught, "Expected KeyError for missing date, but none was raised"
    finally:
        os.unlink(tmp_path2)

    print(f"\n{SEPARATOR}")
    print("All smoke tests PASSED.")
    print(SEPARATOR)
