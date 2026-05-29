from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from .data_loader import CustomDataset, load_market_features, prepare_text_dataframe


TARGETS = [
    "0050.TW",
    "00632R.TW",
    "00679B.TW",
    "2303.TW",
    "2308.TW",
    "2317.TW",
    "2330.TW",
    "2376.TW",
    "2377.TW",
    "2382.TW",
    "2454.TW",
    "3711.TW",
]


@dataclass(frozen=True)
class Config:
    window_size: int = 20
    seed: int = 42
    max_splits: int = 7
    min_split_samples: int = 600
    epochs: int = 18
    batch_size: int = 512
    hidden_dim: int = 96
    target_embedding_dim: int = 8
    dropout: float = 0.25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    high_conf_quantile: float = 0.85


@dataclass
class TargetBundle:
    target: str
    target_id: int
    trading_index: pd.DatetimeIndex
    market_features: np.ndarray
    text_features: np.ndarray
    labels: np.ndarray
    next_returns: np.ndarray


@dataclass(frozen=True)
class SampleRecord:
    bundle_id: int
    target: str
    target_id: int
    date: pd.Timestamp
    pos: int
    label: int
    next_return: float


@dataclass
class Normalizer:
    market_mean: np.ndarray
    market_std: np.ndarray
    text_mean: np.ndarray
    text_std: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generic_market_df(market_df: pd.DataFrame, target: str) -> pd.DataFrame:
    rename = {
        f"close_{target}": "close_target",
        f"close_ret_{target}": "close_ret_target",
        f"volume_{target}": "volume_target",
    }
    market_df = market_df.rename(columns=rename).copy()
    seen: dict[str, int] = {}
    columns: list[str] = []
    for col in market_df.columns:
        count = seen.get(col, 0)
        columns.append(col if count == 0 else f"{col}__dup{count}")
        seen[col] = count + 1
    market_df.columns = columns
    return market_df


def _text_path(base_dir: Path) -> Path:
    candidates = [
        base_dir / "data/trump_nlp/trump_posts_features_2017_2026_relabel.csv",
        base_dir / "data/trump_nlp/trump_posts_features_2017_2026.csv",
        base_dir / "data/text/trump_posts_features_2017_2026.csv",
    ]
    return next((p for p in candidates if p.exists()), candidates[1])


def load_bundles(base_dir: Path, targets: Iterable[str], config: Config) -> tuple[list[TargetBundle], list[str], list[str]]:
    text_path = _text_path(base_dir)
    print(f"text_features={text_path}")
    text_df = prepare_text_dataframe(text_path)

    raw: list[tuple[str, pd.DataFrame, CustomDataset]] = []
    market_cols: set[str] = set()
    text_cols: set[str] = set()

    for target in targets:
        try:
            market_df = load_market_features(
                base_dir / "data/taiwan_market_data/global_prices.csv",
                base_dir / "data/taiwan_market_data/global_volumes.csv",
                base_dir / "data/taiwan_market_data/institutional_investors.csv",
                base_dir / "data/taiwan_market_data/margin_trading.csv",
                base_dir / "data/taiwan_market_data/tx_futures_night.csv",
                target,
            )
            market_df = _generic_market_df(market_df, target)
            if "close_target" not in market_df.columns:
                print(f"[skip] {target}: close_target missing")
                continue
            ds = CustomDataset(
                market_df=market_df,
                text_df=text_df,
                window_size=config.window_size,
                close_price_col="close_target",
                open_price_col="close_target",
                aggregation="trumpcode_daily",
                label_mode="event_binary",
                high_signal_only=False,
            )
        except Exception as exc:
            print(f"[skip] {target}: {exc}")
            continue

        raw.append((target, market_df, ds))
        market_cols.update(c for c in market_df.columns if pd.api.types.is_numeric_dtype(market_df[c]))
        text_cols.update(ds.text_feature_cols)

    if not raw:
        raise RuntimeError("no target bundles could be loaded")

    market_col_list = sorted(market_cols)
    text_col_list = sorted(text_cols)
    bundles: list[TargetBundle] = []

    for target_id, (target, market_df, ds) in enumerate(raw):
        market_features = (
            market_df.reindex(columns=market_col_list, fill_value=0.0)
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
            .astype(np.float32)
            .to_numpy()
        )
        text_daily = pd.DataFrame(ds.text_features, index=ds.trading_index, columns=ds.text_feature_cols)
        text_features = (
            text_daily.reindex(columns=text_col_list, fill_value=0.0)
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
            .astype(np.float32)
            .to_numpy()
        )
        close = market_df["close_target"].to_numpy(dtype=np.float32)
        next_returns = np.full(len(close), np.nan, dtype=np.float32)
        valid_close = np.isfinite(close[:-1]) & np.isfinite(close[1:]) & (close[:-1] > 0)
        next_returns[:-1][valid_close] = close[1:][valid_close] / close[:-1][valid_close] - 1.0
        bundles.append(
            TargetBundle(
                target=target,
                target_id=target_id,
                trading_index=ds.trading_index,
                market_features=market_features,
                text_features=text_features,
                labels=ds.labels.astype(np.float32),
                next_returns=next_returns,
            )
        )

    return bundles, market_col_list, text_col_list


def build_samples(bundles: list[TargetBundle], window_size: int) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for bundle_id, bundle in enumerate(bundles):
        for pos in range(window_size - 1, len(bundle.trading_index)):
            label = bundle.labels[pos]
            next_return = bundle.next_returns[pos]
            if not np.isfinite(label) or not np.isfinite(next_return):
                continue
            records.append(
                SampleRecord(
                    bundle_id=bundle_id,
                    target=bundle.target,
                    target_id=bundle.target_id,
                    date=pd.Timestamp(bundle.trading_index[pos]).normalize(),
                    pos=pos,
                    label=int(label),
                    next_return=float(next_return),
                )
            )
    records.sort(key=lambda r: (r.date, r.target))
    return records


def make_splits(records: list[SampleRecord], config: Config) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    dates = pd.DatetimeIndex([r.date for r in records])
    split_defs = [
        ("2018-12-31", "2019-12-31", "2020-12-31"),
        ("2019-12-31", "2020-12-31", "2021-12-31"),
        ("2020-12-31", "2021-12-31", "2022-12-31"),
        ("2021-12-31", "2022-12-31", "2023-12-31"),
        ("2022-12-31", "2023-12-31", "2024-12-31"),
        ("2023-12-31", "2024-12-31", "2025-12-31"),
        ("2024-12-31", "2025-12-31", "2026-12-31"),
    ]
    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    for train_end_s, val_end_s, test_end_s in split_defs:
        train_end = pd.Timestamp(train_end_s)
        val_end = pd.Timestamp(val_end_s)
        test_end = pd.Timestamp(test_end_s)
        train_idx = np.where(dates <= train_end)[0]
        val_idx = np.where((dates > train_end) & (dates <= val_end))[0]
        test_idx = np.where((dates > val_end) & (dates <= test_end))[0]
        if min(len(train_idx), len(val_idx), len(test_idx)) < config.min_split_samples:
            continue
        split_name = f"train_to_{train_end.year}_val_{val_end.year}_test_{test_end.year}"
        splits.append((train_idx, val_idx, test_idx, split_name))
        if len(splits) >= config.max_splits:
            break
    return splits


def fit_normalizer(records: list[SampleRecord], indices: np.ndarray, bundles: list[TargetBundle], window_size: int) -> Normalizer:
    market_chunks: list[np.ndarray] = []
    text_chunks: list[np.ndarray] = []
    for idx in indices:
        record = records[int(idx)]
        bundle = bundles[record.bundle_id]
        start = record.pos - window_size + 1
        market_chunks.append(bundle.market_features[start : record.pos + 1])
        text_chunks.append(bundle.text_features[start : record.pos + 1])
    market = np.concatenate(market_chunks, axis=0)
    text = np.concatenate(text_chunks, axis=0)
    market_std = market.std(axis=0, keepdims=True)
    text_std = text.std(axis=0, keepdims=True)
    return Normalizer(
        market_mean=market.mean(axis=0, keepdims=True),
        market_std=np.where(market_std < 1e-6, 1.0, market_std),
        text_mean=text.mean(axis=0, keepdims=True),
        text_std=np.where(text_std < 1e-6, 1.0, text_std),
    )


class SequenceDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        indices: np.ndarray,
        bundles: list[TargetBundle],
        normalizer: Normalizer,
        window_size: int,
        use_text: bool,
    ) -> None:
        self.records = records
        self.indices = np.asarray(indices, dtype=np.int64)
        self.bundles = bundles
        self.normalizer = normalizer
        self.window_size = window_size
        self.use_text = use_text

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[int(self.indices[idx])]
        bundle = self.bundles[record.bundle_id]
        start = record.pos - self.window_size + 1
        market = (bundle.market_features[start : record.pos + 1] - self.normalizer.market_mean) / self.normalizer.market_std
        text = (bundle.text_features[start : record.pos + 1] - self.normalizer.text_mean) / self.normalizer.text_std
        if not self.use_text:
            text = np.zeros_like(text, dtype=np.float32)
        return {
            "market": torch.tensor(market, dtype=torch.float32),
            "text": torch.tensor(text, dtype=torch.float32),
            "target_id": torch.tensor(record.target_id, dtype=torch.long),
            "label": torch.tensor(record.label, dtype=torch.long),
        }


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class EventSequenceNet(nn.Module):
    def __init__(
        self,
        market_dim: int,
        text_dim: int,
        n_targets: int,
        hidden_dim: int,
        target_embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.market_lstm = nn.LSTM(market_dim, hidden_dim, batch_first=True)
        self.text_lstm = nn.LSTM(text_dim, hidden_dim, batch_first=True)
        self.market_pool = AttentionPool(hidden_dim)
        self.text_pool = AttentionPool(hidden_dim)
        self.market_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.target_embedding = nn.Embedding(n_targets, target_embedding_dim)
        self.text_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        nn.init.constant_(self.text_gate[0].bias, -1.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + target_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        market_seq, _ = self.market_lstm(batch["market"])
        text_seq, _ = self.text_lstm(batch["text"])
        market_feat = self.market_norm(self.market_pool(market_seq))
        text_feat = self.text_norm(self.text_pool(text_seq))
        gate = self.text_gate(torch.cat([market_feat, text_feat], dim=1))
        fused = market_feat + gate * text_feat
        features = torch.cat([fused, self.target_embedding(batch["target_id"])], dim=1)
        return self.head(features)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_model(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    config: Config,
    market_dim: int,
    text_dim: int,
    n_targets: int,
    device: torch.device,
) -> EventSequenceNet:
    model = EventSequenceNet(
        market_dim=market_dim,
        text_dim=text_dim,
        n_targets=n_targets,
        hidden_dim=config.hidden_dim,
        target_embedding_dim=config.target_embedding_dim,
        dropout=config.dropout,
    ).to(device)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    labels = np.array([train_ds.records[int(i)].label for i in train_ds.indices], dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state = None
    best_f1 = -1.0
    for _epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            batch = _move_batch(batch, device)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["label"], weight=weights)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_pred = predict(model, val_loader, device)
        score = f1_score(val_pred["label"], val_pred["pred"], average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []
    offset = 0
    dataset: SequenceDataset = loader.dataset  # type: ignore[assignment]
    with torch.no_grad():
        for batch in loader:
            batch_size = batch["label"].shape[0]
            proba = torch.softmax(model(_move_batch(batch, device)), dim=1).cpu().numpy()
            pred = proba.argmax(axis=1)
            for j in range(batch_size):
                record = dataset.records[int(dataset.indices[offset + j])]
                rows.append(
                    {
                        "target": record.target,
                        "target_id": record.target_id,
                        "date": record.date.strftime("%Y-%m-%d"),
                        "label": int(record.label),
                        "pred": int(pred[j]),
                        "proba_down": float(proba[j, 0]),
                        "proba_up": float(proba[j, 1]),
                        "confidence": float(proba[j].max()),
                        "next_return": float(record.next_return),
                    }
                )
            offset += batch_size
    return pd.DataFrame(rows)


def summarize_predictions(pred: pd.DataFrame, split: str, model_name: str, threshold: float | None = None) -> dict[str, object]:
    label = pred["label"].to_numpy()
    yhat = pred["pred"].to_numpy()
    row: dict[str, object] = {
        "split": split,
        "model": model_name,
        "n": len(pred),
        "accuracy": accuracy_score(label, yhat),
        "macro_f1": f1_score(label, yhat, average="macro", zero_division=0),
        "precision_down": precision_score(label, yhat, pos_label=0, zero_division=0),
        "precision_up": precision_score(label, yhat, pos_label=1, zero_division=0),
        "recall_down": recall_score(label, yhat, pos_label=0, zero_division=0),
        "recall_up": recall_score(label, yhat, pos_label=1, zero_division=0),
        "mean_proba_up": pred["proba_up"].mean(),
    }
    if threshold is not None:
        selected = pred[(pred["pred"] == 1) & (pred["confidence"] >= threshold)].copy()
        row["high_conf_threshold"] = threshold
        row["n_high_conf_long"] = len(selected)
        row["high_conf_hit_rate"] = float((selected["next_return"] > 0).mean()) if len(selected) else 0.0
        row["high_conf_avg_return"] = float(selected["next_return"].mean()) if len(selected) else 0.0
        row["high_conf_sum_return"] = float(selected["next_return"].sum()) if len(selected) else 0.0
    return row


def run_event_sequence(base_dir: Path, output_dir: Path, config: Config | None = None) -> None:
    config = config or Config()
    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    bundles, market_cols, text_cols = load_bundles(base_dir, TARGETS, config)
    records = build_samples(bundles, config.window_size)
    splits = make_splits(records, config)
    if not splits:
        raise RuntimeError(f"not enough sequence samples for walk-forward: n={len(records)}")

    pd.DataFrame({"market_feature": market_cols}).to_csv(output_dir / "event_sequence_market_features.csv", index=False)
    pd.DataFrame({"text_feature": text_cols}).to_csv(output_dir / "event_sequence_text_features.csv", index=False)

    all_predictions: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    market_dim = len(market_cols)
    text_dim = len(text_cols)
    n_targets = len(bundles)

    for split_id, (train_idx, val_idx, test_idx, split_name) in enumerate(splits, start=1):
        normalizer = fit_normalizer(records, train_idx, bundles, config.window_size)
        print(f"\n=== Split {split_id}: {split_name} | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} ===")

        for model_name, use_text in [("market_only", False), ("event_sequence", True)]:
            train_ds = SequenceDataset(records, train_idx, bundles, normalizer, config.window_size, use_text=use_text)
            val_ds = SequenceDataset(records, val_idx, bundles, normalizer, config.window_size, use_text=use_text)
            test_ds = SequenceDataset(records, test_idx, bundles, normalizer, config.window_size, use_text=use_text)
            model = train_model(train_ds, val_ds, config, market_dim, text_dim, n_targets, device)
            val_pred = predict(model, DataLoader(val_ds, batch_size=config.batch_size, shuffle=False), device)
            threshold = float(val_pred["confidence"].quantile(config.high_conf_quantile))
            test_pred = predict(model, DataLoader(test_ds, batch_size=config.batch_size, shuffle=False), device)
            test_pred.insert(0, "model", model_name)
            test_pred.insert(0, "split", split_name)
            all_predictions.append(test_pred)
            metrics = summarize_predictions(test_pred, split_name, model_name, threshold=threshold)
            summary_rows.append(metrics)
            print(
                f"{model_name}: macro_f1={metrics['macro_f1']:.4f} "
                f"acc={metrics['accuracy']:.4f} "
                f"high_conf_long={metrics['n_high_conf_long']} "
                f"hit={metrics['high_conf_hit_rate']:.4f} "
                f"avg_ret={metrics['high_conf_avg_return']:.5f}"
            )

    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    target_summary = (
        predictions.groupby(["model", "target"])
        .apply(lambda g: pd.Series({
            "n": len(g),
            "accuracy": accuracy_score(g["label"], g["pred"]),
            "macro_f1": f1_score(g["label"], g["pred"], average="macro", zero_division=0),
            "precision_up": precision_score(g["label"], g["pred"], pos_label=1, zero_division=0),
            "avg_next_return_when_pred_up": g.loc[g["pred"] == 1, "next_return"].mean(),
        }))
        .reset_index()
    )
    overall = (
        predictions.groupby("model")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "accuracy": accuracy_score(g["label"], g["pred"]),
            "macro_f1": f1_score(g["label"], g["pred"], average="macro", zero_division=0),
            "precision_up": precision_score(g["label"], g["pred"], pos_label=1, zero_division=0),
            "recall_up": recall_score(g["label"], g["pred"], pos_label=1, zero_division=0),
        }))
        .reset_index()
    )

    predictions.to_csv(output_dir / "event_sequence_predictions.csv", index=False)
    summary.to_csv(output_dir / "event_sequence_summary.csv", index=False)
    target_summary.to_csv(output_dir / "event_sequence_target_summary.csv", index=False)
    overall.to_csv(output_dir / "event_sequence_overall.csv", index=False)

    print(f"\nSaved predictions to {output_dir / 'event_sequence_predictions.csv'}")
    print(f"Saved split summary to {output_dir / 'event_sequence_summary.csv'}")
    print(f"Saved overall summary to {output_dir / 'event_sequence_overall.csv'}")
