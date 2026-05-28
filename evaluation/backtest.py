"""
evaluation/backtest.py
======================
Long-only PnL engine for the Trump-Sentiment × TSMC validation prototype.
All public functions are pure (no class state, no side effects).

Execution timing (no lookahead bias)
--------------------------------------
Signal generated at the close of day T uses only information available at or
before close_T.  The resulting position is held during the T → T+1 window:

    r_{T+1}^strategy = position_T × r_{T+1}^stock
    r_{T+1}^stock    = close_{T+1} / close_T − 1

Transaction costs are charged on the day the position *changes*
(position_T ≠ position_{T-1}, with position_{-1} = 0):

    Buy  (0 → 1): deduct fee_buy
    Sell (1 → 0): deduct fee_sell + tax_sell

Taiwan cost model defaults:
    fee_buy   = 0.1425% (brokerage, buy side)
    fee_sell  = 0.1425% (brokerage, sell side)
    tax_sell  = 0.300%  (securities transaction tax, sell only)
    round-trip ≈ 0.585%
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def signals_from_predictions(y_pred: np.ndarray) -> np.ndarray:
    """Map class predictions to long-only position signals.

    Parameters
    ----------
    y_pred:
        Integer array of class predictions with values in {0, 1, 2}:
        0 = 大跌, 1 = 盤整, 2 = 大漲.

    Returns
    -------
    np.ndarray
        int8 array of the same shape.  2 (大漲) → 1 (long); all other
        values → 0 (cash).
    """
    y_pred = np.asarray(y_pred)
    return np.where(y_pred == 2, np.int8(1), np.int8(0)).astype(np.int8)


def sharpe_ratio(
    daily_returns: np.ndarray | pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualised Sharpe ratio (assuming zero risk-free rate).

    Parameters
    ----------
    daily_returns:
        Array or Series of daily net returns (not cumulative).
    periods_per_year:
        Number of trading periods per year used for annualisation.
        Default is 252 (trading days).

    Returns
    -------
    float
        Annualised Sharpe = mean(r) / std(r, ddof=1) * sqrt(periods_per_year).
        Returns 0.0 when std == 0 (e.g., all-cash strategy producing a
        constant zero return) to avoid division by zero.
    """
    r = np.asarray(daily_returns, dtype=float)
    if r.size == 0:
        return 0.0
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    if sigma == 0.0:
        return 0.0
    return float(mu / sigma * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray | pd.Series) -> float:
    """Compute the maximum peak-to-trough drawdown of an equity curve.

    Parameters
    ----------
    equity_curve:
        Array or Series of portfolio values (not returns), e.g., starting at 1.0.
        All values should be positive.

    Returns
    -------
    float
        A positive number in [0, 1] representing the worst relative decline
        from any historical peak to a subsequent trough.  For example, 0.15
        means a 15% drawdown.  Returns 0.0 if the curve is monotonically
        non-decreasing (no drawdown observed).
    """
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size == 0:
        return 0.0
    # Running maximum up to and including each point
    running_peak = np.maximum.accumulate(eq)
    # Guard against zero peak values (should not happen with positive equity)
    with np.errstate(invalid="ignore", divide="ignore"):
        drawdowns = np.where(
            running_peak > 0,
            (running_peak - eq) / running_peak,
            0.0,
        )
    return float(np.max(drawdowns))


def backtest(
    close: pd.Series,
    signals: pd.Series,
    fee_buy: float = 0.001425,
    fee_sell: float = 0.001425,
    tax_sell: float = 0.003,
    initial_capital: float = 1.0,
) -> dict:
    """Run a vectorised long-only backtest on a single instrument.

    Parameters
    ----------
    close:
        Date-indexed ``pd.Series`` of daily close prices.  Must have at least
        2 rows so that at least one return can be computed.
    signals:
        Date-indexed ``pd.Series`` of *intended positions for the next period*.
        Each ``signals.iloc[t]`` is the position held during the window
        ``close.iloc[t] → close.iloc[t+1]``.  Values must be in {0, 1}.

        **The index of ``signals`` must equal ``close.index[:-1]``** (the last
        close has no future return; a lookahead guard asserts this at runtime).
    fee_buy:
        One-way brokerage fee on a buy trade (default 0.1425%).
    fee_sell:
        One-way brokerage fee on a sell trade (default 0.1425%).
    tax_sell:
        Securities transaction tax on a sell trade (default 0.3%).
    initial_capital:
        Starting portfolio value (default 1.0, i.e., fully normalised).

    Returns
    -------
    dict with keys:
        ``"equity_curve"``      : pd.Series indexed by ``close.index[1:]``
        ``"daily_returns"``     : pd.Series of net returns (same index)
        ``"cumulative_return"`` : float — ``equity_curve.iloc[-1] / initial_capital - 1``
        ``"sharpe"``            : float — annualised Sharpe of daily_returns
        ``"max_drawdown"``      : float — peak-to-trough MDD of equity_curve
        ``"n_trades"``          : int — total number of position changes (buys + sells)

    Raises
    ------
    AssertionError
        If ``signals.index`` does not exactly match ``close.index[:-1]`` —
        this would indicate a lookahead-bias bug in the caller.
    """
    # ------------------------------------------------------------------
    # Lookahead-bias guard: signal at day T can only use information
    # available at close_T; it earns the return from close_T to close_{T+1}.
    # Therefore signals must be indexed by close.index[:-1] (all days except
    # the last), whose corresponding *next* close is close.index[1:].
    # ------------------------------------------------------------------
    assert signals.index.equals(close.index[:-1]), (
        "Lookahead-bias guard failed: signals.index must exactly equal "
        "close.index[:-1].  Ensure that signals[t] was generated at or "
        "before close_T and that no signal is provided for the final close "
        "(which has no future return to trade on).\n"
        f"  signals.index : {signals.index[[0, -1]].tolist()} "
        f"(len={len(signals)})\n"
        f"  close.index[:-1]: {close.index[[-1, -1]].tolist()} "
        f"(len={len(close)-1})"
    )

    n = len(signals)
    close_arr = close.values  # shape (n+1,)
    sig_arr = signals.values.astype(np.int8)  # shape (n,)

    equity_vals = np.empty(n, dtype=float)
    net_returns = np.empty(n, dtype=float)

    equity = initial_capital
    prev_position = np.int8(0)
    n_trades = 0

    for t in range(n):
        position_t = sig_arr[t]

        # Stock return for window close[t] → close[t+1]
        r_stock = close_arr[t + 1] / close_arr[t] - 1.0

        # Gross strategy return (long-only: position ∈ {0, 1})
        r_strategy = float(position_t) * r_stock

        # Transaction cost on position change
        cost = 0.0
        if position_t != prev_position:
            n_trades += 1
            if position_t == 1:
                # Going from cash (0) to long (1) — buy
                cost = fee_buy
            else:
                # Going from long (1) to cash (0) — sell
                cost = fee_sell + tax_sell

        r_net = r_strategy - cost
        equity = equity * (1.0 + r_net)

        net_returns[t] = r_net
        equity_vals[t] = equity
        prev_position = position_t

    result_index = close.index[1:]
    equity_series = pd.Series(equity_vals, index=result_index, name="equity")
    returns_series = pd.Series(net_returns, index=result_index, name="net_return")

    return {
        "equity_curve": equity_series,
        "daily_returns": returns_series,
        "cumulative_return": float(equity_series.iloc[-1] / initial_capital - 1.0),
        "sharpe": sharpe_ratio(returns_series),
        "max_drawdown": max_drawdown(equity_series),
        "n_trades": n_trades,
    }


# ---------------------------------------------------------------------------
# Smoke tests (run with: python -m evaluation.backtest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    _PASS = "PASS"
    _FAIL = "FAIL"

    def _check(label: str, condition: bool, detail: str = "") -> None:
        status = _PASS if condition else _FAIL
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{status}] {label}{suffix}")

    def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
        return abs(a - b) <= tol

    # ------------------------------------------------------------------
    # Case 1: All-cash strategy
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Case 1: All-cash (signals = all zeros)")
    print("Expected: cum_return=0, n_trades=0, sharpe=0, mdd=0")

    dates_1 = pd.date_range("2020-01-01", periods=6, freq="B")
    close_1 = pd.Series([100.0, 105.0, 103.0, 108.0, 102.0, 110.0], index=dates_1)
    signals_1 = pd.Series([0, 0, 0, 0, 0], index=dates_1[:-1], dtype=np.int8)

    res_1 = backtest(close_1, signals_1)
    _check("cum_return == 0.0", _approx(res_1["cumulative_return"], 0.0),
           f"got {res_1['cumulative_return']}")
    _check("n_trades == 0", res_1["n_trades"] == 0, f"got {res_1['n_trades']}")
    _check("sharpe == 0.0", _approx(res_1["sharpe"], 0.0), f"got {res_1['sharpe']}")
    _check("max_drawdown == 0.0", _approx(res_1["max_drawdown"], 0.0),
           f"got {res_1['max_drawdown']}")

    # ------------------------------------------------------------------
    # Case 2: Buy day 0, hold forever (signals = all ones)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Case 2: Buy day 0, hold forever (signals = all ones)")
    print("Expected: n_trades == 1, cum_return ≈ close[-1]/close[0] - 1 - fee_buy")

    dates_2 = pd.date_range("2020-01-01", periods=6, freq="B")
    close_2 = pd.Series([100.0, 105.0, 103.0, 108.0, 102.0, 110.0], index=dates_2)
    signals_2 = pd.Series([1, 1, 1, 1, 1], index=dates_2[:-1], dtype=np.int8)

    res_2 = backtest(close_2, signals_2)
    # Compound manually (only fee_buy at entry on day 0; no selling cost)
    fee_buy_default = 0.001425
    steps = [
        (105 / 100 - 1) - fee_buy_default,
        103 / 105 - 1,
        108 / 103 - 1,
        102 / 108 - 1,
        110 / 102 - 1,
    ]
    expected_equity_2 = 1.0
    for s in steps:
        expected_equity_2 *= (1.0 + s)
    expected_cum_2 = expected_equity_2 - 1.0

    _check("n_trades == 1", res_2["n_trades"] == 1, f"got {res_2['n_trades']}")
    _check(
        f"cum_return ≈ {expected_cum_2:.6f}",
        _approx(res_2["cumulative_return"], expected_cum_2, tol=1e-10),
        f"got {res_2['cumulative_return']:.6f}",
    )

    # ------------------------------------------------------------------
    # Case 3: One buy + one sell (hand-computed)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Case 3: close=[100,110,121], signals=[1,0]")
    print("  Day 0→1: 1*(110/100-1) - fee_buy = 0.1 - 0.001425 = 0.098575")
    print("  Day 1→2: 0*(121/110-1) - (fee_sell+tax_sell) = 0 - 0.004425 = -0.004425")
    print("  Equity:  1.0 → 1.098575 → 1.098575*(1-0.004425) ≈ 1.09371...")
    print("  n_trades == 2")

    dates_3 = pd.date_range("2020-01-01", periods=3, freq="B")
    close_3 = pd.Series([100.0, 110.0, 121.0], index=dates_3)
    signals_3 = pd.Series([1, 0], index=dates_3[:-1], dtype=np.int8)

    res_3 = backtest(close_3, signals_3)

    expected_r0 = 0.1 - 0.001425          # 0.098575
    expected_r1 = 0.0 - (0.001425 + 0.003)  # -0.004425
    expected_eq0 = 1.0 * (1.0 + expected_r0)  # 1.098575
    expected_eq1 = expected_eq0 * (1.0 + expected_r1)
    expected_cum_3 = expected_eq1 - 1.0

    daily_r = res_3["daily_returns"]
    eq_curve = res_3["equity_curve"]

    _check(
        f"daily_returns[0] == {expected_r0:.6f}",
        _approx(daily_r.iloc[0], expected_r0),
        f"got {daily_r.iloc[0]:.6f}",
    )
    _check(
        f"daily_returns[1] == {expected_r1:.6f}",
        _approx(daily_r.iloc[1], expected_r1),
        f"got {daily_r.iloc[1]:.6f}",
    )
    _check(
        f"equity_curve[0] == {expected_eq0:.6f}",
        _approx(eq_curve.iloc[0], expected_eq0),
        f"got {eq_curve.iloc[0]:.6f}",
    )
    _check(
        f"equity_curve[1] == {expected_eq1:.8f}",
        _approx(eq_curve.iloc[1], expected_eq1),
        f"got {eq_curve.iloc[1]:.8f}",
    )
    _check(
        f"cum_return == {expected_cum_3:.8f}",
        _approx(res_3["cumulative_return"], expected_cum_3),
        f"got {res_3['cumulative_return']:.8f}",
    )
    _check("n_trades == 2", res_3["n_trades"] == 2, f"got {res_3['n_trades']}")

    # Detailed printout for manual verification
    print(f"\n  Computed equity_curve[0]  = {eq_curve.iloc[0]:.10f}  (expected {expected_eq0:.10f})")
    print(f"  Computed equity_curve[1]  = {eq_curve.iloc[1]:.10f}  (expected {expected_eq1:.10f})")
    print(f"  Computed cum_return       = {res_3['cumulative_return']:.10f}  (expected {expected_cum_3:.10f})")
    print(f"  Computed sharpe           = {res_3['sharpe']:.6f}")
    print(f"  Computed max_drawdown     = {res_3['max_drawdown']:.6f}")

    # ------------------------------------------------------------------
    # Case 4: Lookahead-bias guard triggers AssertionError on mismatched index
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Case 4: Mismatched signals index → expect AssertionError")

    dates_4 = pd.date_range("2020-01-01", periods=4, freq="B")
    close_4 = pd.Series([100.0, 105.0, 103.0, 108.0], index=dates_4)
    # Intentionally wrong: use close.index[1:] instead of close.index[:-1]
    bad_signals_4 = pd.Series([1, 0, 1], index=dates_4[1:], dtype=np.int8)

    try:
        backtest(close_4, bad_signals_4)
        _check("AssertionError raised", False, "no error was raised — UNEXPECTED")
    except AssertionError as exc:
        _check("AssertionError raised", True, f"message starts: {str(exc)[:60]!r}")

    # ------------------------------------------------------------------
    # Case 5: MDD sanity — always-declining equity
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Case 5: max_drawdown on known equity curve [1.0, 0.9, 0.8]")
    print("  Peak at t=0 is 1.0; trough at t=2 is 0.8 → MDD = (1.0-0.8)/1.0 = 0.2")

    eq_5 = np.array([1.0, 0.9, 0.8])
    mdd_5 = max_drawdown(eq_5)
    _check(f"max_drawdown == 0.2", _approx(mdd_5, 0.2, tol=1e-12), f"got {mdd_5:.10f}")

    # ------------------------------------------------------------------
    # Case 6: Sharpe ratio sanity — known small example
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Case 6: sharpe_ratio on known returns [0.01, 0.02, 0.03]")
    r6 = np.array([0.01, 0.02, 0.03])
    mu6 = np.mean(r6)               # 0.02
    sigma6 = np.std(r6, ddof=1)     # std([0.01,0.02,0.03], ddof=1) = 0.01
    expected_sharpe_6 = mu6 / sigma6 * math.sqrt(252)
    computed_sharpe_6 = sharpe_ratio(r6)
    print(f"  mean={mu6}, std={sigma6:.6f}, expected={expected_sharpe_6:.6f}")
    _check(
        f"sharpe ≈ {expected_sharpe_6:.6f}",
        _approx(computed_sharpe_6, expected_sharpe_6, tol=1e-9),
        f"got {computed_sharpe_6:.6f}",
    )

    print()
    print("All cases complete.")
