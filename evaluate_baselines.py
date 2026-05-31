import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate direction and profitability baselines for Trump regime-fusion outputs."
    )
    parser.add_argument(
        "--predictions",
        default="data/output/deep_regime_fusion_terms_only_mlp/test_predictions.csv",
        help="Path to test_predictions.csv produced by event_combo.py.",
    )
    parser.add_argument(
        "--price-path",
        default="data/taiwan_market_data/global_prices.csv",
        help="Path to global_prices.csv.",
    )
    parser.add_argument("--target", default="2330.TW", help="Target ticker used by the model.")
    parser.add_argument(
        "--index-target",
        default="0050.TW",
        help="Index/market buy-and-hold benchmark ticker in global_prices.csv.",
    )
    parser.add_argument("--hold", type=int, default=1, help="Prediction horizon in trading days.")
    parser.add_argument(
        "--momentum-lookbacks",
        default="1,5",
        help="Comma-separated lookback horizons for momentum accuracy baselines.",
    )
    parser.add_argument("--ma-fast", type=int, default=5, help="Fast moving average window.")
    parser.add_argument("--ma-slow", type=int, default=20, help="Slow moving average window.")
    parser.add_argument(
        "--random-trials",
        type=int,
        default=10000,
        help="Number of random monkey simulations.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the predictions file directory.",
    )
    return parser.parse_args()


def load_predictions(path):
    pred = pd.read_csv(path)
    pred["date"] = pd.to_datetime(pred["date"])
    required = {"date", "actual_label", "pred_label", "future_ret", "trade_signal", "strategy_ret_no_cost"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return pred.sort_values("date").reset_index(drop=True)


def load_prices(path):
    prices = pd.read_csv(path)
    prices["Date"] = pd.to_datetime(prices["Date"])
    return prices.sort_values("Date").set_index("Date")


def total_return(returns):
    returns = pd.Series(returns).fillna(0.0)
    return float((1.0 + returns).prod() - 1.0)


def max_drawdown(returns):
    returns = pd.Series(returns).fillna(0.0)
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def sharpe_like(returns):
    returns = pd.Series(returns).fillna(0.0)
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(252) * returns.mean() / std)


def hit_rate(returns):
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())


def strategy_summary(name, signal, future_ret):
    signal = pd.Series(np.asarray(signal), dtype=float).fillna(0.0)
    future_ret = pd.Series(np.asarray(future_ret), dtype=float).fillna(0.0)
    strategy_ret = signal * future_ret
    traded = signal != 0
    return {
        "name": name,
        "n": int(len(strategy_ret)),
        "coverage": float(traded.mean()),
        "long_count": int((signal > 0).sum()),
        "short_count": int((signal < 0).sum()),
        "neutral_count": int((signal == 0).sum()),
        "mean_return": float(strategy_ret.mean()),
        "total_return": total_return(strategy_ret),
        "sharpe_like": sharpe_like(strategy_ret),
        "max_drawdown": max_drawdown(strategy_ret),
        "hit_rate_all_days": hit_rate(strategy_ret),
        "hit_rate_traded_days": hit_rate(strategy_ret[traded]) if traded.any() else 0.0,
    }


def direction_baselines(pred, prices, target, lookbacks):
    actual = pred["actual_label"].astype(int).to_numpy()
    rows = []

    majority_label = int(pd.Series(actual).mode().iloc[0])
    majority_pred = np.full(len(actual), majority_label, dtype=int)
    rows.append(
        {
            "name": f"majority_class_{'up' if majority_label == 1 else 'down'}",
            "accuracy": float(accuracy_score(actual, majority_pred)),
            "pred_up_rate": float(majority_pred.mean()),
        }
    )

    model_pred = pred["pred_label"].astype(int).to_numpy()
    rows.append(
        {
            "name": "model_direction",
            "accuracy": float(accuracy_score(actual, model_pred)),
            "pred_up_rate": float(model_pred.mean()),
        }
    )

    price = prices[target].reindex(pred["date"])
    full_price = prices[target]
    for lookback in lookbacks:
        past_ret = full_price.pct_change(lookback).reindex(pred["date"])
        momentum_pred = (past_ret > 0).astype(int).fillna(0).to_numpy()
        rows.append(
            {
                "name": f"momentum_{lookback}d",
                "accuracy": float(accuracy_score(actual, momentum_pred)),
                "pred_up_rate": float(momentum_pred.mean()),
            }
        )

    return pd.DataFrame(rows)


def future_return_from_prices(prices, ticker, dates, hold):
    if ticker not in prices.columns:
        return None
    series = prices[ticker]
    future_ret = series.shift(-hold) / series - 1.0
    return future_ret.reindex(dates)


def moving_average_signal(prices, target, dates, fast, slow):
    price = prices[target]
    ma_fast = price.rolling(fast, min_periods=fast).mean()
    ma_slow = price.rolling(slow, min_periods=slow).mean()
    signal = pd.Series(np.where(ma_fast > ma_slow, 1.0, -1.0), index=prices.index)
    signal[(ma_fast.isna()) | (ma_slow.isna())] = 0.0
    return signal.reindex(dates).fillna(0.0)


def random_monkey(pred, random_trials, seed):
    rng = np.random.default_rng(seed)
    model_signal = pred["trade_signal"].astype(float).to_numpy()
    future_ret = pred["future_ret"].astype(float).to_numpy()

    long_count = int((model_signal > 0).sum())
    short_count = int((model_signal < 0).sum())
    n = len(model_signal)

    random_totals = np.empty(random_trials, dtype=float)
    random_means = np.empty(random_trials, dtype=float)
    random_hit_rates = np.empty(random_trials, dtype=float)

    for i in range(random_trials):
        signal = np.zeros(n, dtype=float)
        if long_count + short_count > 0:
            chosen = rng.choice(n, size=long_count + short_count, replace=False)
            if long_count > 0:
                signal[chosen[:long_count]] = 1.0
            if short_count > 0:
                signal[chosen[long_count:]] = -1.0
        strategy_ret = signal * future_ret
        random_totals[i] = np.prod(1.0 + strategy_ret) - 1.0
        random_means[i] = strategy_ret.mean()
        traded = signal != 0
        random_hit_rates[i] = (strategy_ret[traded] > 0).mean() if traded.any() else 0.0

    model_total = total_return(pred["strategy_ret_no_cost"])
    p_value = float((np.sum(random_totals >= model_total) + 1) / (random_trials + 1))
    summary = {
        "mode": "same_long_short_neutral_counts_as_model",
        "trials": random_trials,
        "model_total_return": model_total,
        "p_value_random_total_ge_model": p_value,
        "random_total_mean": float(random_totals.mean()),
        "random_total_std": float(random_totals.std(ddof=1)),
        "random_total_p05": float(np.quantile(random_totals, 0.05)),
        "random_total_p50": float(np.quantile(random_totals, 0.50)),
        "random_total_p95": float(np.quantile(random_totals, 0.95)),
        "random_mean_return_mean": float(random_means.mean()),
        "random_hit_rate_mean": float(random_hit_rates.mean()),
    }
    samples = pd.DataFrame(
        {
            "trial": np.arange(random_trials),
            "total_return": random_totals,
            "mean_return": random_means,
            "hit_rate_traded_days": random_hit_rates,
        }
    )
    return summary, samples


def main():
    args = parse_args()
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pred = load_predictions(predictions_path)
    prices = load_prices(args.price_path)
    lookbacks = [int(x.strip()) for x in args.momentum_lookbacks.split(",") if x.strip()]

    if args.target not in prices.columns:
        raise ValueError(f"Target {args.target!r} not found in {args.price_path}.")

    direction_df = direction_baselines(pred, prices, args.target, lookbacks)

    dates = pred["date"]
    target_future_ret = pred["future_ret"].astype(float)
    model_signal = pred["trade_signal"].astype(float)
    rows = [
        strategy_summary("model_strategy", model_signal, target_future_ret),
        strategy_summary("target_buy_and_hold", np.ones(len(pred)), target_future_ret),
        strategy_summary(
            f"ma_{args.ma_fast}_{args.ma_slow}_long_short",
            moving_average_signal(prices, args.target, dates, args.ma_fast, args.ma_slow).to_numpy(),
            target_future_ret,
        ),
    ]

    index_future_ret = future_return_from_prices(prices, args.index_target, dates, args.hold)
    if index_future_ret is not None:
        rows.append(
            strategy_summary(
                f"index_buy_and_hold_{args.index_target}",
                np.ones(len(pred)),
                index_future_ret,
            )
        )

    profitability_df = pd.DataFrame(rows)
    random_summary, random_samples = random_monkey(pred, args.random_trials, args.seed)

    model_actual = pred["actual_label"].astype(int)
    model_pred = pred["pred_label"].astype(int)
    model_report = classification_report(
        model_actual,
        model_pred,
        labels=[0, 1],
        target_names=["down", "up"],
        zero_division=0,
        output_dict=True,
    )
    model_cm = confusion_matrix(model_actual, model_pred, labels=[0, 1]).tolist()

    summary = {
        "predictions": str(predictions_path),
        "target": args.target,
        "index_target": args.index_target,
        "hold": args.hold,
        "date_range": {
            "start": pred["date"].min().strftime("%Y-%m-%d"),
            "end": pred["date"].max().strftime("%Y-%m-%d"),
        },
        "n": int(len(pred)),
        "direction_baselines": direction_df.to_dict(orient="records"),
        "profitability_baselines": profitability_df.to_dict(orient="records"),
        "random_monkey": random_summary,
        "model_classification_report": model_report,
        "model_confusion_matrix_labels": ["down", "up"],
        "model_confusion_matrix": model_cm,
    }

    direction_df.to_csv(output_dir / "baseline_direction_accuracy.csv", index=False)
    profitability_df.to_csv(output_dir / "baseline_profitability.csv", index=False)
    random_samples.to_csv(output_dir / "random_monkey_returns.csv", index=False)
    (output_dir / "evaluation_baselines.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nDirection accuracy baselines")
    print(direction_df.to_string(index=False))
    print("\nProfitability baselines")
    print(profitability_df.to_string(index=False))
    print("\nRandom monkey baseline")
    print(json.dumps(random_summary, indent=2))
    print(f"\nSaved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
