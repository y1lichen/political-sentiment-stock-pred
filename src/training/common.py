from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED
from src.utils.io import safe_name


SKLEARN_MODELS = {"logistic", "elasticnet", "random_forest", "lightgbm"}


def seed_everything(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_by_mode(df: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.to_datetime(df["date"])
    if mode == "regime_matched":
        train = (dates >= "2017-01-20") & (dates <= "2019-12-31")
        val = (dates >= "2020-01-01") & (dates <= "2021-01-20")
        test = dates >= "2025-01-20"
    elif mode == "all_history":
        train = (dates >= "2017-01-20") & (dates <= "2022-12-31")
        val = (dates >= "2023-01-01") & (dates <= "2024-12-31")
        test = dates >= "2025-01-20"
    elif mode == "regime_aware":
        train = (dates >= "2017-01-20") & (dates <= "2022-12-31")
        val = (dates >= "2023-01-01") & (dates <= "2024-12-31")
        test = dates >= "2025-01-20"
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    return train.to_numpy(), val.to_numpy(), test.to_numpy()


def artifact_stem(model: str, target: str, split: str, feature_set: str) -> str:
    return f"{model}_{safe_name(target)}_{split}_{feature_set}"


def build_sample_weights(df: pd.DataFrame, split_mode: str) -> np.ndarray:
    weights = np.ones(len(df), dtype=np.float32)
    if split_mode == "regime_aware" and "covid_policy_period" in df.columns:
        weights *= np.where(df["covid_policy_period"].fillna(0).to_numpy() == 1, 0.75, 1.0)
    return weights


def fit_sklearn_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
):
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
        raise ValueError(f"Unknown sklearn model: {model_name}")

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clf),
        ]
    )
    pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
    return pipe


def predict_model(bundle, model_name: str, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if model_name in SKLEARN_MODELS:
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
