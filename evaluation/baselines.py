"""
evaluation/baselines.py
=======================
Financial baselines for the Trump-Sentiment × TSMC validation prototype.

Provides two non-ML baselines:

  B1 Buy & Hold  — enters on day 0, holds to end.  Default mode is
                   mark-to-market (no forced final-day sell cost);
                   pass ``liquidated=True`` for a version that sells on the
                   last day and deducts ``fee_sell + tax_sell``.

  B3 SMA 5/20    — golden-cross / death-cross crossover.  Goes long when
                   the 5-day MA crosses above the 20-day MA, sits in cash
                   otherwise.

Neither baseline produces ``大漲/盤整/大跌`` class labels, so neither
subclasses ``ClassifierModel`` from :mod:`evaluation.model_interface`.
They bypass the classifier interface entirely and return the same dict
shape as :func:`evaluation.backtest.backtest`.  The ``run_eval`` orchestrator
handles them as "Financial-only" rows in the final comparison table (ML
metric columns set to N/A).

Note on B2 (Pure-Market Model)
-------------------------------
B2 is **out of scope** for this file.  It will be a
``CsvPredictionsModel(...)`` instance loaded by ``run_eval`` when the
teammate delivers ``data/pure_market_predictions.csv``.

Public symbols
--------------
``buy_and_hold``        — B1 backtest, returns dict
``sma_signals``         — B3 position series, returns pd.Series
``run_buy_and_hold``    — thin wrapper used by run_eval, tags result with name
``run_sma``             — thin wrapper used by run_eval, tags result with name
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import backtest


# ---------------------------------------------------------------------------
# B1: Buy and Hold
# ---------------------------------------------------------------------------


def buy_and_hold(
    close: pd.Series,
    fee_buy: float = 0.001425,
    fee_sell: float = 0.001425,
    tax_sell: float = 0.003,
    initial_capital: float = 1.0,
    liquidated: bool = False,
) -> dict:
    """Simulate a buy-and-hold strategy on a single instrument.

    Enters a long position on day 0 and holds through the end of the price
    series.  Transaction costs follow the Taiwan cost model.

    Parameters
    ----------
    close:
        Date-indexed ``pd.Series`` of daily close prices.  Must have at
        least 2 rows.
    fee_buy:
        One-way brokerage fee on the initial buy (default 0.1425 %).
    fee_sell:
        One-way brokerage fee on the final sell — charged only when
        ``liquidated=True`` (default 0.1425 %).
    tax_sell:
        Securities transaction tax on the final sell — charged only when
        ``liquidated=True`` (default 0.3 %).
    initial_capital:
        Starting portfolio value (default 1.0, i.e., fully normalised).
    liquidated:
        ``False`` (default) — mark-to-market: hold is evaluated at the
        final close without incurring a sell cost.  Only the initial buy
        cost is deducted (``n_trades == 1``).

        ``True`` — liquidated mode: a forced sell is recorded on the last
        day, deducting ``fee_sell + tax_sell`` from the final-day return
        (``n_trades == 2``).

    Returns
    -------
    dict
        Same keys as :func:`evaluation.backtest.backtest`:

        - ``"equity_curve"``      : pd.Series indexed by ``close.index[1:]``
        - ``"daily_returns"``     : pd.Series of net returns (same index)
        - ``"cumulative_return"`` : float
        - ``"sharpe"``            : float (annualised)
        - ``"max_drawdown"``      : float in [0, 1]
        - ``"n_trades"``          : int  (1 for mark-to-market, 2 for liquidated)

    Notes
    -----
    Because the position never changes after the initial buy (all-ones
    signal), the backtest engine charges a transaction cost only on the
    day the position changes.  For all-ones signals that gives exactly one
    trade (the buy at day 0).  Adding a trailing ``0`` forces a sell on the
    last day, producing two trades.

    Implementation uses :func:`evaluation.backtest.backtest` to avoid
    duplicating cost and compounding logic.

    Raises
    ------
    ValueError
        If ``close`` has fewer than 2 rows.
    """
    n = len(close)
    if n < 2:
        raise ValueError(
            f"buy_and_hold requires at least 2 rows in close; got {n}."
        )

    # Signals must align with close.index[:-1] (lookahead guard in backtest)
    signal_index = close.index[:-1]

    if not liquidated:
        # Mark-to-market: hold for every period, never sell.
        # All-ones signal → position never changes after day 0 buy → n_trades=1.
        signals = pd.Series(
            np.ones(len(signal_index), dtype=np.int8),
            index=signal_index,
        )
    else:
        # Liquidated: hold through second-to-last period, sell on last period.
        # signals[-1] = 0 triggers a sell at the transition from day (n-2) to
        # day (n-1), deducting fee_sell + tax_sell from the final return.
        sig_values = np.ones(len(signal_index), dtype=np.int8)
        sig_values[-1] = np.int8(0)
        signals = pd.Series(sig_values, index=signal_index)

    return backtest(
        close=close,
        signals=signals,
        fee_buy=fee_buy,
        fee_sell=fee_sell,
        tax_sell=tax_sell,
        initial_capital=initial_capital,
    )


# ---------------------------------------------------------------------------
# B3: SMA crossover signal generator
# ---------------------------------------------------------------------------


def sma_signals(
    close: pd.Series,
    fast: int = 5,
    slow: int = 20,
) -> pd.Series:
    """Compute a 5/20 SMA crossover position series (B3 baseline).

    The signal at time T uses only information available at the close of day
    T (no lookahead bias).  The resulting position is intended to be held
    during the T → T+1 window.

    Parameters
    ----------
    close:
        Date-indexed ``pd.Series`` of daily close prices.  Must contain no
        NaN values.
    fast:
        Look-back window for the fast moving average (default 5).
    slow:
        Look-back window for the slow moving average (default 20).

    Returns
    -------
    pd.Series
        Integer series (dtype int8) indexed by ``close.index[:-1]``, with
        values:

        - ``1`` — fast MA > slow MA (golden cross → long)
        - ``0`` — fast MA ≤ slow MA or either MA is NaN (death cross / warm-up
          period → cash)

    Notes
    -----
    The last close is excluded from the returned index because there is no
    subsequent period to hold a position in (consistent with the lookahead
    guard in :func:`evaluation.backtest.backtest`).

    For the first ``slow - 1`` days the slow MA is undefined (NaN); those
    days default to position ``0`` (cash).
    """
    if fast <= 0 or slow <= 0:
        raise ValueError(
            f"fast and slow must be positive integers; got fast={fast}, slow={slow}."
        )
    if fast >= slow:
        raise ValueError(
            f"fast ({fast}) must be strictly less than slow ({slow}) for a "
            "meaningful crossover signal."
        )
    if len(close) < 2:
        raise ValueError(
            f"sma_signals requires at least 2 rows in close; got {len(close)}."
        )

    fast_ma: pd.Series = close.rolling(fast).mean()
    slow_ma: pd.Series = close.rolling(slow).mean()

    # Compute raw signal for every date in close.index[:-1]:
    #   1 if fast_ma > slow_ma (and both defined), else 0.
    raw = (fast_ma > slow_ma).astype(np.int8)

    # Where the slow MA is NaN (warm-up period), force cash position.
    nan_mask: pd.Series = slow_ma.isna()
    raw[nan_mask] = np.int8(0)

    # Return only the signal-generating dates (all but the very last close).
    return raw.iloc[:-1].rename("signal").astype(np.int8)


# ---------------------------------------------------------------------------
# Thin wrappers for run_eval orchestrator
# ---------------------------------------------------------------------------


def run_buy_and_hold(close: pd.Series, **kwargs) -> dict:
    """Wrapper around :func:`buy_and_hold` that tags the result with a name.

    Intended for use by ``run_eval.py`` to build the comparison table row.

    Parameters
    ----------
    close:
        Date-indexed close price series.
    **kwargs:
        Forwarded to :func:`buy_and_hold` (e.g., ``liquidated``,
        ``fee_buy``, ``fee_sell``, ``tax_sell``, ``initial_capital``).

    Returns
    -------
    dict
        Same as :func:`buy_and_hold`, with an additional key
        ``"name": "Buy & Hold"``.
    """
    result = buy_and_hold(close, **kwargs)
    result["name"] = "Buy & Hold"
    return result


def run_sma(
    close: pd.Series,
    fast: int = 5,
    slow: int = 20,
    **kwargs,
) -> dict:
    """Wrapper that builds SMA signals and runs :func:`backtest`, tagging the result.

    Intended for use by ``run_eval.py`` to build the comparison table row.

    Parameters
    ----------
    close:
        Date-indexed close price series.
    fast:
        Fast MA window (default 5).
    slow:
        Slow MA window (default 20).
    **kwargs:
        Forwarded to :func:`evaluation.backtest.backtest` (e.g., ``fee_buy``,
        ``fee_sell``, ``tax_sell``, ``initial_capital``).

    Returns
    -------
    dict
        Same as :func:`evaluation.backtest.backtest`, with an additional key
        ``"name": f"SMA {fast}/{slow}"``.
    """
    signals = sma_signals(close, fast=fast, slow=slow)
    result = backtest(close=close, signals=signals, **kwargs)
    result["name"] = f"SMA {fast}/{slow}"
    return result


# ---------------------------------------------------------------------------
# Self-verification (run with: python -m evaluation.baselines)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    from .data_module import load_tsmc_prices, temporal_split, make_labels

    _PASS = "PASS"
    _FAIL = "FAIL"

    def _check(label: str, condition: bool, detail: str = "") -> None:
        status = _PASS if condition else _FAIL
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{status}] {label}{suffix}")

    SEPARATOR = "=" * 60

    # ------------------------------------------------------------------
    # Load data — use only the test split (2025-01-01 onwards) for speed
    # ------------------------------------------------------------------
    _this_dir = Path(__file__).resolve().parent
    _repo_root = _this_dir.parent
    _prices_path = _repo_root / "data" / "taiwan_market_data" / "global_prices.csv"

    # Fallback for worktree environments: if data not found under repo root,
    # look in the canonical project root two levels up from the worktree.
    if not _prices_path.exists():
        _alt_root = _repo_root.parent.parent.parent  # .claude/worktrees/agent-xxx/ → repo
        _prices_path = _alt_root / "data" / "taiwan_market_data" / "global_prices.csv"
    if not _prices_path.exists():
        # Try relative to common worktree structure:
        # .claude/worktrees/<id>/evaluation/../data/...
        # The original repo lives 3 levels up from worktree root
        for _candidate_root in [
            _repo_root.parent.parent.parent,
            Path("/mnt/sda/home/r147250250916/2026spring/DLA/political-sentiment-stock-pred"),
        ]:
            _candidate = _candidate_root / "data" / "taiwan_market_data" / "global_prices.csv"
            if _candidate.exists():
                _prices_path = _candidate
                break

    print(f"Loading prices from: {_prices_path}")
    close_full = load_tsmc_prices(_prices_path)
    print(
        f"  Full series  : {len(close_full)} rows  "
        f"[{close_full.index.min().date()} → {close_full.index.max().date()}]"
    )

    # Test split: 2025-01-01 onwards
    TEST_START = "2025-01-01"
    close = close_full[close_full.index >= TEST_START].copy()
    print(
        f"  Test slice   : {len(close)} rows  "
        f"[{close.index.min().date()} → {close.index.max().date()}]"
    )
    print()

    # ------------------------------------------------------------------
    # Case 1: B1 mark-to-market (liquidated=False)
    # ------------------------------------------------------------------
    print(SEPARATOR)
    print("Case 1: B1 Buy & Hold — mark-to-market (liquidated=False)")

    b1_mtm = buy_and_hold(close, liquidated=False)
    spot_return = float(close.iloc[-1] / close.iloc[0] - 1)
    fee_buy_default = 0.001425

    # Expected: cum_return ≈ spot_return - fee_buy
    # (only the initial buy cost is incurred; no sell at end)
    expected_approx = spot_return - fee_buy_default
    delta = abs(b1_mtm["cumulative_return"] - expected_approx)

    print(f"  spot return        : {spot_return:.6f}")
    print(f"  expected approx    : {expected_approx:.6f}  (spot - fee_buy)")
    print(f"  actual cum_return  : {b1_mtm['cumulative_return']:.6f}")
    print(f"  difference         : {delta:.2e}  (should be tiny — compounding effect)")
    print(f"  n_trades           : {b1_mtm['n_trades']}  (expected 1)")

    # The compounding effect means the exact value isn't spot-fee_buy, but it
    # should be very close (within a few percent of spot_return for short series).
    # Assert: n_trades == 1 (only initial buy)
    _check("n_trades == 1", b1_mtm["n_trades"] == 1, f"got {b1_mtm['n_trades']}")
    # Assert: difference from (spot - fee_buy) is small relative to |spot_return|
    tolerance = max(0.005, 0.02 * abs(spot_return))  # 2% of spot return or 0.5%
    _check(
        f"|cum_return - (spot - fee_buy)| < {tolerance:.4f}",
        delta < tolerance,
        f"delta={delta:.6f}",
    )

    # ------------------------------------------------------------------
    # Case 2: B1 liquidated (liquidated=True)
    # ------------------------------------------------------------------
    print()
    print(SEPARATOR)
    print("Case 2: B1 Buy & Hold — liquidated (liquidated=True)")

    b1_liq = buy_and_hold(close, liquidated=True)
    print(f"  mark-to-market cum_return : {b1_mtm['cumulative_return']:.6f}")
    print(f"  liquidated     cum_return : {b1_liq['cumulative_return']:.6f}")
    print(f"  delta (should be negative): {b1_liq['cumulative_return'] - b1_mtm['cumulative_return']:.6f}")
    print(f"  n_trades                  : {b1_liq['n_trades']}  (expected 2)")

    _check("n_trades == 2", b1_liq["n_trades"] == 2, f"got {b1_liq['n_trades']}")
    _check(
        "liquidated cum_return < mark-to-market cum_return",
        b1_liq["cumulative_return"] < b1_mtm["cumulative_return"],
        f"liq={b1_liq['cumulative_return']:.6f}, mtm={b1_mtm['cumulative_return']:.6f}",
    )

    # ------------------------------------------------------------------
    # Case 3: B3 SMA 5/20
    # ------------------------------------------------------------------
    print()
    print(SEPARATOR)
    print("Case 3: B3 SMA 5/20 crossover — end-to-end backtest")

    b3 = run_sma(close, fast=5, slow=20)
    print(f"  name           : {b3['name']}")
    print(f"  n_trades       : {b3['n_trades']}")
    print(f"  cum_return     : {b3['cumulative_return']:.6f}")
    print(f"  sharpe         : {b3['sharpe']:.6f}")
    print(f"  max_drawdown   : {b3['max_drawdown']:.6f}")

    _check(
        "sharpe is finite",
        np.isfinite(b3["sharpe"]),
        f"got {b3['sharpe']}",
    )
    _check(
        "max_drawdown in [0, 1]",
        0.0 <= b3["max_drawdown"] <= 1.0,
        f"got {b3['max_drawdown']:.6f}",
    )
    _check(
        "equity_curve length == len(close) - 1",
        len(b3["equity_curve"]) == len(close) - 1,
        f"got {len(b3['equity_curve'])}",
    )

    # ------------------------------------------------------------------
    # Case 4: Edge case — monotonically increasing close → long position
    # ------------------------------------------------------------------
    print()
    print(SEPARATOR)
    print("Case 4: SMA edge case — monotonically increasing close series")

    _dates = pd.date_range("2020-01-01", periods=40, freq="B")
    _close_mono = pd.Series(
        [float(100 + i) for i in range(40)],  # strictly increasing
        index=_dates,
    )
    _sigs = sma_signals(_close_mono, fast=5, slow=20)

    # fast MA always equals slow MA or exceeds it once warm-up passes on a mono series
    n_long = int((_sigs == 1).sum())
    n_cash = int((_sigs == 0).sum())
    print(f"  series length  : {len(_close_mono)}")
    print(f"  signal length  : {len(_sigs)}  (should be {len(_close_mono) - 1})")
    print(f"  long periods   : {n_long}")
    print(f"  cash periods   : {n_cash}  (first {20-1}=19 days during warm-up)")

    _check(
        "signal length == len(close) - 1",
        len(_sigs) == len(_close_mono) - 1,
        f"got {len(_sigs)}",
    )
    _check(
        "at least one long position on monotonically increasing close",
        n_long >= 1,
        f"got n_long={n_long}",
    )
    # On a monotonically increasing series fast > slow once slow is defined
    # (fast MA of recent higher values > slow MA including older lower values).
    # Warm-up period (first slow-1 = 19 signals) should be cash.
    _check(
        "first slow-1=19 signals are cash (warm-up NaN → 0)",
        int((_sigs.iloc[:19] == 0).sum()) == 19,
        f"cash count in first 19={int((_sigs.iloc[:19] == 0).sum())}",
    )

    print()
    print(SEPARATOR)
    print("All self-verification cases complete.")
