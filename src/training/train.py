from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DEFAULT_TARGET, RANDOM_SEED, Paths
from src.data.build_dataset import build_modeling_table
from src.evaluation.metrics import classification_metrics, regression_metrics, signal_metrics
from src.models.torch_models import EventGatedMLP, SmallMLP
from src.utils.io import ensure_dir, read_pickle, safe_name, write_json, write_pickle


TARGET_COLUMNS = {"target_return_1d", "target_direction_1d", "target_big_move_1d", "target_close"}
NON_FEATURE_COLUMNS = {"date", *TARGET_COLUMNS}


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_by_mode(df: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = pd.to_datetime(df["date"])
    if mode == "regime_matched":
        train = (d >= "2017-01-20") & (d <= "2019-12-31")
        val = (d >= "2020-01-01") & (d <= "2021-01-20")
        test = d >= "2025-01-20"
    elif mode == "all_history":
        train = (d >= "2017-01-20") & (d <= "2022-12-31")
        val = (d >= "2023-01-01") & (d <= "2024-12-31")
        test = d >= "2025-01-20"
    elif mode == "regime_aware":
        train = (d >= "2017-01-20") & (d <= "2022-12-31")
        val = (d >= "2023-01-01") & (d <= "2024-12-31")
        test = d >= "2025-01-20"
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    return train.to_numpy(), val.to_numpy(), test.to_numpy()


def build_sample_weights(df: pd.DataFrame, split_mode: str) -> np.ndarray:
    weights = np.ones(len(df), dtype=np.float32)
    if split_mode != "regime_aware":
        return weights
    if "is_president" in df.columns:
        weights *= np.where(df["is_president"].fillna(0).to_numpy() == 1, 1.5, 0.65)
    if "trump_sum_tc_tariff" in df.columns:
        weights *= np.where(df["trump_sum_tc_tariff"].fillna(0).to_numpy() > 0, 1.25, 1.0)
    if "covid_policy_period" in df.columns:
        weights *= np.where(df["covid_policy_period"].fillna(0).to_numpy() == 1, 0.75, 1.0)
    return weights


def select_feature_columns(df: pd.DataFrame, feature_budget: int = 80) -> list[str]:
    candidates = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().any()
    ]
    priority_prefixes = (
        "trump_",
        "mkt_",
        "vol_",
        "tx_night_",
        "inst_",
        "margin_",
    )
    priority_names = {
        "is_president",
        "first_term",
        "post_presidency",
        "second_term",
        "campaign_period",
        "covid_crash_period",
        "covid_recovery_liquidity_period",
        "covid_policy_period",
        "policy_power_score",
        "tariff_regime_intensity",
        "high_vix_regime",
        "market_stress_score",
        "event_gate_default",
    }
    ordered = [
        c
        for c in candidates
        if c in priority_names or c.startswith(priority_prefixes)
    ]
    ordered += [c for c in candidates if c not in ordered]
    return ordered[:feature_budget]


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
    return {
        "clean": clean,
        "X": X,
        "y_cls": y_cls,
        "y_ret": y_ret,
        **masks,
    }


def fit_sklearn_model(model_name: str, X_train: pd.DataFrame, y_train: np.ndarray, sample_weight: np.ndarray):
    if model_name == "logistic":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    elif model_name == "elasticnet":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.25,
            random_state=RANDOM_SEED,
        )
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    elif model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("lightgbm is not installed. Use --model logistic or install lightgbm.") from exc
        clf = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=RANDOM_SEED,
            verbose=-1,
        )
    else:
        raise ValueError(model_name)

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clf),
        ]
    )
    pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
    return pipe


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


def predict_model(bundle, model_name: str, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if model_name in {"logistic", "elasticnet", "random_forest", "lightgbm"}:
        proba = bundle.predict_proba(X)[:, 1]
        ret = (proba - 0.5) * 0.02
        return proba, ret, None

    model = bundle["model"]
    imputer = bundle["imputer"]
    scaler = bundle["scaler"]
    Xn = scaler.transform(imputer.transform(X)).astype(np.float32)
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(Xn))
        proba = torch.sigmoid(out[0]).numpy()
        ret = out[1].numpy()
        gate = torch.sigmoid(out[2]).numpy() if model_name == "event_gated_mlp" else None
    return proba, ret, gate


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
    features = select_feature_columns(df, args.feature_budget)
    arrays = prepare_arrays(df, features, train_mask, val_mask, test_mask)
    clean = arrays["clean"]
    X = arrays["X"]
    y = arrays["y_cls"]
    y_ret = arrays["y_ret"]
    weights = build_sample_weights(clean, args.split)

    tr, va, te = arrays["train"], arrays["val"], arrays["test"]
    if tr.sum() < 50 or va.sum() < 20:
        raise RuntimeError(f"Not enough samples. train={tr.sum()} val={va.sum()} test={te.sum()}")

    if args.model in {"logistic", "elasticnet", "random_forest", "lightgbm"}:
        bundle = fit_sklearn_model(args.model, X[tr], y[tr], weights[tr])
        model_path = paths.models_dir / f"{args.model}_{safe_name(args.target)}_{args.split}.pkl"
        write_pickle({"model": bundle, "features": features, "model_name": args.model}, model_path)
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
        model_path = paths.models_dir / f"{args.model}_{safe_name(args.target)}_{args.split}.pt"
        save_torch_bundle(bundle, model_path, features, vars(args))

    pred_frames = []
    metrics = {
        "target": args.target,
        "model": args.model,
        "split": args.split,
        "feature_count": len(features),
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
    pred_path = paths.predictions_dir / f"predictions_{args.model}_{safe_name(args.target)}_{args.split}.csv"
    ensure_dir(pred_path.parent)
    pred.to_csv(pred_path, index=False)
    metrics["prediction_path"] = str(pred_path)
    metrics_path = paths.reports_dir / f"metrics_{args.model}_{safe_name(args.target)}_{args.split}.json"
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
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--signal-threshold", type=float, default=0.55)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
