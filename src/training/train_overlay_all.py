from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import Paths
from src.data.build_dataset import build_modeling_table
from src.training.train_overlay import train_overlay
from src.utils.io import ensure_dir, safe_name, write_json


DEFAULT_TICKERS = [
    "2330.TW",
    "3711.TW",
    "2382.TW",
    "2454.TW",
    "0050.TW",
    "2317.TW",
    "2308.TW",
    "2303.TW",
    "2376.TW",
    "2377.TW",
    "00632R.TW",
    "00679B.TW",
    "TSM",
    "^SOX",
    "^NDX",
    "^GSPC",
    "^VIX",
    "TWD=X",
    "^TNX",
]


def overlay_stem(target: str, split: str, market_model: str, overlay_model: str) -> str:
    return f"overlay_{safe_name(target)}_{split}_{market_model}_{overlay_model}"


def collect_one(paths: Paths, target: str, split: str, market_model: str, overlay_model: str) -> tuple[list[dict], dict]:
    stem = overlay_stem(target, split, market_model, overlay_model)
    summary_path = paths.reports_dir / f"summary_{stem}.csv"
    pred_path = paths.predictions_dir / f"predictions_{stem}.csv"
    report_path = paths.reports_dir / f"report_{stem}.md"
    model_path = paths.models_dir / f"{stem}.pkl"

    summary_rows: list[dict] = []
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        test = summary[summary["split"].eq("test")].copy()
        for _, row in test.iterrows():
            payload = row.to_dict()
            payload["ticker"] = target
            payload["model_path"] = str(model_path)
            payload["report_path"] = str(report_path)
            summary_rows.append(payload)
        if {"market_only", "market_plus_trump_overlay"}.issubset(set(test["strategy"])):
            pivot = test.set_index("strategy")
            delta = {
                "ticker": target,
                "strategy": "overlay_minus_market",
                "split": "test",
                "model_path": str(model_path),
                "report_path": str(report_path),
            }
            numeric_cols = test.select_dtypes(include="number").columns
            for col in numeric_cols:
                delta[col] = pivot.loc["market_plus_trump_overlay", col] - pivot.loc["market_only", col]
            summary_rows.append(delta)

    latest: dict = {
        "ticker": target,
        "prediction_path": str(pred_path),
        "report_path": str(report_path),
        "model_path": str(model_path),
    }
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        if not pred.empty:
            row = pred.sort_values("date").iloc[-1].to_dict()
            latest.update(row)
            latest["market_signal"] = position_to_signal(latest.get("market_position"))
            latest["overlay_signal"] = position_to_signal(latest.get("overlay_position"))
    return summary_rows, latest


def position_to_signal(position) -> str:
    try:
        value = float(position)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if value > 0:
        return "LONG"
    if value < 0:
        return "SHORT"
    return "NO_TRADE"


def has_usable_target_data(target: str, min_rows: int) -> tuple[bool, str, int]:
    paths = Paths()
    dataset_path = paths.datasets_dir / f"modeling_table_{safe_name(target)}.csv"
    if dataset_path.exists():
        df = pd.read_csv(dataset_path, usecols=["target_return_1d"])
    else:
        df = build_modeling_table(paths.trump_posts, paths.market_dir, target)
    usable_rows = int(df["target_return_1d"].notna().sum())
    if usable_rows < min_rows:
        return False, f"NO_USABLE_TARGET_DATA usable_rows={usable_rows} min_rows={min_rows}", usable_rows
    return True, "OK", usable_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Market baseline + Trump overlay for many tickers.")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--split", default="regime_aware", choices=["regime_matched", "all_history", "regime_aware"])
    parser.add_argument("--market-model", default="lightgbm", choices=["logistic", "elasticnet", "random_forest", "lightgbm"])
    parser.add_argument("--overlay-model", default="elasticnet", choices=["logistic", "elasticnet", "random_forest", "lightgbm"])
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--all-overlay-features", action="store_true")
    parser.add_argument("--overlay-all-days", action="store_true")
    parser.add_argument("--objective", default="return_after_costs", choices=[
        "avg_signal_return",
        "cumulative_return",
        "compound_nav",
        "sharpe",
        "return_after_costs",
    ])
    parser.add_argument("--market-threshold-min", type=float, default=0.50)
    parser.add_argument("--market-threshold-max", type=float, default=0.90)
    parser.add_argument("--veto-min", type=float, default=0.00)
    parser.add_argument("--veto-max", type=float, default=0.70)
    parser.add_argument("--boost-min", type=float, default=0.60)
    parser.add_argument("--boost-max", type=float, default=0.90)
    parser.add_argument("--boost-size", type=float, default=1.5)
    parser.add_argument("--override-min", type=float, default=0.60)
    parser.add_argument("--override-max", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--min-val-signals", type=int, default=50)
    parser.add_argument("--min-overlay-train-samples", type=int, default=50)
    parser.add_argument("--min-target-rows", type=int, default=100)
    parser.add_argument("--transaction-cost", type=float, default=0.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    paths = Paths()
    ensure_dir(paths.reports_dir)
    ensure_dir(paths.predictions_dir)

    failures = []
    skipped = []
    for ticker in args.tickers:
        print(f"\n=== Training overlay for {ticker} ===")
        ok, reason, usable_rows = has_usable_target_data(ticker, args.min_target_rows)
        if not ok:
            payload = {"ticker": ticker, "reason": reason, "usable_rows": usable_rows}
            skipped.append(payload)
            failures.append({"ticker": ticker, "error": reason})
            print(f"SKIPPED {ticker}: {reason}")
            continue
        overlay_args = argparse.Namespace(
            target=ticker,
            split=args.split,
            dataset=None,
            rebuild_dataset=args.rebuild_dataset,
            market_model=args.market_model,
            overlay_model=args.overlay_model,
            all_overlay_features=args.all_overlay_features,
            overlay_event_only=not args.overlay_all_days,
            overlay_all_days=args.overlay_all_days,
            objective=args.objective,
            market_threshold_min=args.market_threshold_min,
            market_threshold_max=args.market_threshold_max,
            veto_min=args.veto_min,
            veto_max=args.veto_max,
            boost_min=args.boost_min,
            boost_max=args.boost_max,
            boost_size=args.boost_size,
            override_min=args.override_min,
            override_max=args.override_max,
            threshold_step=args.threshold_step,
            min_val_signals=args.min_val_signals,
            min_overlay_train_samples=args.min_overlay_train_samples,
            transaction_cost=args.transaction_cost,
            slippage=args.slippage,
        )
        try:
            train_overlay(overlay_args)
        except Exception as exc:  # keep batch running unless requested otherwise
            failures.append({"ticker": ticker, "error": repr(exc)})
            print(f"FAILED {ticker}: {exc!r}")
            if args.fail_fast:
                raise

    summary_rows: list[dict] = []
    latest_rows: list[dict] = []
    skipped_by_ticker = {item["ticker"]: item for item in skipped}
    for ticker in args.tickers:
        rows, latest = collect_one(paths, ticker, args.split, args.market_model, args.overlay_model)
        if ticker in skipped_by_ticker:
            item = skipped_by_ticker[ticker]
            latest.update(
                {
                    "status": "SKIPPED",
                    "skip_reason": item["reason"],
                    "usable_rows": item["usable_rows"],
                }
            )
        else:
            latest.setdefault("status", "OK")
        summary_rows.extend(rows)
        latest_rows.append(latest)

    run_stem = f"all_tickers_overlay_{args.split}_{args.market_model}_{args.overlay_model}"
    summary_out = paths.reports_dir / f"{run_stem}_summary.csv"
    latest_out = paths.predictions_dir / f"{run_stem}_latest.csv"
    failures_out = paths.reports_dir / f"{run_stem}_failures.json"

    pd.DataFrame(summary_rows).to_csv(summary_out, index=False)
    pd.DataFrame(latest_rows).to_csv(latest_out, index=False)
    write_json(
        {
            "tickers": args.tickers,
            "split": args.split,
            "market_model": args.market_model,
            "overlay_model": args.overlay_model,
            "transaction_cost": args.transaction_cost,
            "slippage": args.slippage,
            "summary_path": str(summary_out),
            "latest_path": str(latest_out),
            "failures": failures,
            "skipped": skipped,
        },
        failures_out,
    )

    print("\n=== Batch outputs ===")
    print(f"Summary: {summary_out}")
    print(f"Latest predictions: {latest_out}")
    print(f"Failures JSON: {failures_out}")
    if failures:
        print(f"Failures: {len(failures)}")
        for failure in failures:
            print(f"  {failure['ticker']}: {failure['error']}")


if __name__ == "__main__":
    main()
