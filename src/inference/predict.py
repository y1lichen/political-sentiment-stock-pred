from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import DEFAULT_TARGET, Paths
from src.data.build_dataset import build_modeling_table
from src.models.torch_models import EventGatedMLP, SmallMLP
from src.training.common import predict_model
from src.utils.io import ensure_dir, read_pickle, safe_name, write_json


def load_model(model_path: Path):
    if model_path.suffix == ".pkl":
        obj = read_pickle(model_path)
        model_name = obj.get("model_name")
        if model_name is None:
            stem = model_path.stem
            for candidate in ("random_forest", "lightgbm", "elasticnet", "logistic"):
                if stem.startswith(candidate):
                    model_name = candidate
                    break
        return obj["model"], obj["features"], model_name or "logistic"

    try:
        state = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    model_name = state["model_name"]
    weights = state["state_dict"]
    first_weight = next(v for v in weights.values() if getattr(v, "ndim", 0) == 2)
    hidden_dim = int(first_weight.shape[0])
    input_dim = int(first_weight.shape[1])
    if model_name == "small_mlp":
        model = SmallMLP(input_dim, hidden_dim=hidden_dim)
    elif model_name == "event_gated_mlp":
        model = EventGatedMLP(input_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unsupported torch model: {model_name}")
    model.load_state_dict(state["state_dict"])
    model.eval()
    bundle = {
        "model_name": model_name,
        "model": model,
        "imputer": state["imputer"],
        "scaler": state["scaler"],
    }
    return bundle, state["features"], model_name


def classify_signal(
    proba: np.ndarray,
    event_gate: np.ndarray,
    threshold: float,
    trade_gate: np.ndarray | None = None,
    trade_gate_threshold: float = 0.5,
) -> list[str]:
    signals = []
    for i, p in enumerate(proba):
        allow = bool(event_gate[i])
        if trade_gate is not None:
            allow = allow and trade_gate[i] >= trade_gate_threshold
        if not allow:
            signals.append("NO_TRADE")
        elif p >= threshold:
            signals.append("LONG")
        elif p <= 1 - threshold:
            signals.append("SHORT")
        else:
            signals.append("NO_TRADE")
    return signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Deep Trump Code inference.")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--latest", action="store_true", help="Only emit the latest available row.")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--trade-gate-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    paths = Paths()
    dataset_path = args.dataset or (paths.datasets_dir / f"modeling_table_{safe_name(args.target)}.csv")
    if args.rebuild_dataset or not dataset_path.exists():
        df = build_modeling_table(paths.trump_posts, paths.market_dir, args.target)
        ensure_dir(dataset_path.parent)
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path, parse_dates=["date"])

    bundle, features, model_name = load_model(args.model_path)
    data = df.sort_values("date").copy()
    if args.latest:
        data = data.tail(1).copy()

    X = data[features].replace([np.inf, -np.inf], np.nan)
    proba, ret_pred, trade_gate = predict_model(bundle, model_name, X)
    event_gate = data.get("event_gate_default", pd.Series(0, index=data.index)).fillna(0).astype(int).to_numpy()
    signals = classify_signal(
        proba,
        event_gate,
        args.threshold,
        trade_gate=trade_gate,
        trade_gate_threshold=args.trade_gate_threshold,
    )

    out = data[["date"]].copy()
    if "target_return_1d" in data.columns:
        out["target_return_1d"] = data["target_return_1d"].to_numpy()
    out["pred_direction_proba"] = proba
    out["pred_return"] = ret_pred
    out["event_gate_default"] = event_gate
    if trade_gate is not None:
        out["pred_trade_gate"] = trade_gate
    out["signal"] = signals

    output = args.output or (paths.predictions_dir / f"inference_{safe_name(args.target)}.csv")
    ensure_dir(output.parent)
    out.to_csv(output, index=False)
    write_json(
        {
            "target": args.target,
            "model_path": str(args.model_path),
            "dataset_path": str(dataset_path),
            "output": str(output),
            "rows": int(len(out)),
            "latest": bool(args.latest),
            "signal_counts": out["signal"].value_counts().to_dict(),
        },
        output.with_suffix(".metadata.json"),
    )
    print(f"Wrote inference: {output}")
    print(out.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
