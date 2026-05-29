from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DEFAULT_TARGET, RANDOM_SEED, Paths
from src.data.build_dataset import build_modeling_table
from src.evaluation.metrics import classification_metrics, regression_metrics, signal_metrics
from src.features.feature_sets import (
    CANONICAL_FEATURE_SETS,
    FEATURE_SET_ALIASES,
    audit_feature_set,
    build_feature_set,
)
from src.models.torch_models import EventGatedMLP, SmallMLP
from src.training.common import (
    SKLEARN_MODELS,
    artifact_stem,
    build_sample_weights,
    fit_sklearn_model,
    predict_model,
    seed_everything,
    split_by_mode,
)
from src.utils.io import ensure_dir, safe_name, write_json, write_pickle


def prepare_arrays(
    df: pd.DataFrame,
    features: list[str],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    clean = df.dropna(subset=["target_return_1d", "target_direction_1d"]).copy()
    index = clean.index.to_numpy()
    masks = {
        "train": np.isin(index, df.index[train_mask]),
        "val": np.isin(index, df.index[val_mask]),
        "test": np.isin(index, df.index[test_mask]),
    }
    X = clean[features].replace([np.inf, -np.inf], np.nan)
    y_cls = clean["target_direction_1d"].astype(int).to_numpy()
    y_ret = clean["target_return_1d"].astype(float).to_numpy()
    return {"clean": clean, "X": X, "y_cls": y_cls, "y_ret": y_ret, **masks}


def fit_torch_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    ret_train: np.ndarray,
    w_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> dict:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(imputer.fit_transform(X_train)).astype(np.float32)
    Xva = scaler.transform(imputer.transform(X_val)).astype(np.float32)
    ytr = y_train.astype(np.float32)
    rtr = ret_train.astype(np.float32)
    wtr = w_train.astype(np.float32)
    yva = y_val.astype(np.float32)

    if model_name == "small_mlp":
        model = SmallMLP(Xtr.shape[1], hidden_dim=hidden_dim)
    elif model_name == "event_gated_mlp":
        model = EventGatedMLP(Xtr.shape[1], hidden_dim=hidden_dim)
    else:
        raise ValueError(model_name)

    ds = TensorDataset(
        torch.from_numpy(Xtr),
        torch.from_numpy(ytr),
        torch.from_numpy(rtr),
        torch.from_numpy(wtr),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    huber = nn.HuberLoss(reduction="none", delta=0.01)
    best_state = None
    best_loss = math.inf
    patience = 12
    stale = 0

    for _ in range(epochs):
        model.train()
        for xb, yb, rb, wb in loader:
            opt.zero_grad()
            out = model(xb)
            direction_logit, ret_pred = out[0], out[1]
            cls_loss = bce(direction_logit, yb)
            reg_loss = huber(ret_pred, rb)
            loss = ((cls_loss + 0.25 * reg_loss) * wb).mean()
            if model_name == "event_gated_mlp":
                gate_logit = out[2]
                gate_target = (torch.abs(rb) > torch.quantile(torch.abs(torch.from_numpy(rtr)), 0.6)).float()
                loss = loss + 0.15 * bce(gate_logit, gate_target).mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(Xva))
            val_loss = bce(out[0], torch.from_numpy(yva)).mean().item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "imputer": imputer, "scaler": scaler, "model_name": model_name}


def save_torch_bundle(bundle: dict, path: Path, features: list[str], metadata: dict) -> None:
    first_weight = next(iter(bundle["model"].state_dict().values()))
    state = {
        "model_name": bundle["model_name"],
        "state_dict": bundle["model"].state_dict(),
        "input_dim": int(first_weight.shape[1]) if first_weight.ndim == 2 else len(features),
        "features": features,
        "imputer": bundle["imputer"],
        "scaler": bundle["scaler"],
        "metadata": metadata,
    }
    ensure_dir(path.parent)
    torch.save(state, path)


def train(args: argparse.Namespace) -> None:
    seed_everything(RANDOM_SEED)
    paths = Paths()
    dataset_path = args.dataset or (paths.datasets_dir / f"modeling_table_{safe_name(args.target)}.csv")
    if not Path(dataset_path).exists() or args.rebuild_dataset:
        df = build_modeling_table(paths.trump_posts, paths.market_dir, args.target)
        ensure_dir(Path(dataset_path).parent)
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path, parse_dates=["date"])

    train_mask, val_mask, test_mask = split_by_mode(df, args.split)
    canonical_feature_set, features = build_feature_set(
        df,
        args.target,
        args.feature_set,
        feature_budget=args.feature_budget,
        all_features=args.all_features,
    )
    feature_audit = audit_feature_set(features)
    arrays = prepare_arrays(df, features, train_mask, val_mask, test_mask)
    clean = arrays["clean"]
    X = arrays["X"]
    y = arrays["y_cls"]
    y_ret = arrays["y_ret"]
    weights = build_sample_weights(clean, args.split)

    tr, va, te = arrays["train"], arrays["val"], arrays["test"]
    if tr.sum() < 50 or va.sum() < 20:
        raise RuntimeError(f"Not enough samples. train={tr.sum()} val={va.sum()} test={te.sum()}")

    stem = artifact_stem(args.model, args.target, args.split, canonical_feature_set)
    feature_path = paths.features_dir / f"selected_features_{stem}.json"
    write_json(
        {
            "target": args.target,
            "model": args.model,
            "split": args.split,
            "requested_feature_set": args.feature_set,
            "feature_set": canonical_feature_set,
            "feature_budget": args.feature_budget,
            "all_features": bool(args.all_features),
            "feature_count": len(features),
            "features": features,
            "audit": feature_audit,
        },
        feature_path,
    )

    if args.model in SKLEARN_MODELS:
        bundle = fit_sklearn_model(args.model, X[tr], y[tr], weights[tr])
        model_path = paths.models_dir / f"{stem}.pkl"
        write_pickle(
            {
                "model": bundle,
                "features": features,
                "model_name": args.model,
                "requested_feature_set": args.feature_set,
                "feature_set": canonical_feature_set,
                "feature_audit": feature_audit,
            },
            model_path,
        )
    else:
        bundle = fit_torch_model(
            args.model,
            X[tr],
            y[tr],
            y_ret[tr],
            weights[tr],
            X[va],
            y[va],
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        model_path = paths.models_dir / f"{stem}.pt"
        metadata = {
            **vars(args),
            "requested_feature_set": args.feature_set,
            "feature_set": canonical_feature_set,
            "feature_audit": feature_audit,
        }
        save_torch_bundle(bundle, model_path, features, metadata)

    pred_frames = []
    metrics = {
        "target": args.target,
        "model": args.model,
        "split": args.split,
        "requested_feature_set": args.feature_set,
        "feature_set": canonical_feature_set,
        "feature_count": len(features),
        "feature_budget": args.feature_budget,
        "all_features": bool(args.all_features),
        "feature_path": str(feature_path),
        "feature_audit": feature_audit,
        "features": features,
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "split_counts": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
    }

    for name, mask in [("train", tr), ("val", va), ("test", te)]:
        if mask.sum() == 0:
            continue
        proba, ret_pred, gate = predict_model(bundle, args.model, X[mask])
        part = clean.loc[mask, ["date", "target_return_1d", "target_direction_1d", "event_gate_default"]].copy()
        part["split"] = name
        part["pred_direction_proba"] = proba
        part["pred_return"] = ret_pred
        if gate is not None:
            part["pred_trade_gate"] = gate
        pred_frames.append(part)
        metrics[name] = {
            **classification_metrics(part["target_direction_1d"].to_numpy(), proba),
            **regression_metrics(part["target_return_1d"].to_numpy(), ret_pred),
            **{f"signal_{k}": v for k, v in signal_metrics(part, threshold=args.signal_threshold).items()},
        }
        try:
            metrics[name]["log_loss"] = float(log_loss(part["target_direction_1d"], proba))
        except ValueError:
            metrics[name]["log_loss"] = float("nan")

    pred = pd.concat(pred_frames, ignore_index=True)
    pred_path = paths.predictions_dir / f"predictions_{stem}.csv"
    ensure_dir(pred_path.parent)
    pred.to_csv(pred_path, index=False)
    metrics["prediction_path"] = str(pred_path)
    metrics_path = paths.reports_dir / f"metrics_{stem}.json"
    write_json(metrics, metrics_path)
    print(f"Wrote model: {model_path}")
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metrics: {metrics_path}")
    for split_name in ("train", "val", "test"):
        if split_name in metrics:
            m = metrics[split_name]
            print(
                f"{split_name}: "
                f"auc={m.get('auc', float('nan')):.4f} "
                f"acc={m.get('accuracy', float('nan')):.4f} "
                f"coverage={m.get('signal_coverage', float('nan')):.4f} "
                f"hit={m.get('signal_hit_rate', float('nan')):.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Deep Trump Code models.")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--model", default="event_gated_mlp", choices=[
        "logistic",
        "elasticnet",
        "random_forest",
        "lightgbm",
        "small_mlp",
        "event_gated_mlp",
    ])
    parser.add_argument("--split", default="regime_aware", choices=["regime_matched", "all_history", "regime_aware"])
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--feature-budget", type=int, default=80)
    parser.add_argument("--all-features", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--signal-threshold", type=float, default=0.55)
    parser.add_argument(
        "--feature-set",
        default="full",
        choices=[*CANONICAL_FEATURE_SETS, *FEATURE_SET_ALIASES],
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
