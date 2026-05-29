from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score
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
    horizons: tuple[int, ...] = (1, 2, 3)
    seed: int = 42
    max_splits: int = 8
    min_split_candidates: int = 1000
    epochs: int = 12
    batch_size: int = 512
    hidden_dim: int = 64
    dropout: float = 0.30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    min_val_rule_trades: int = 8
    min_val_rule_hit_rate: float = 0.58
    min_val_rule_avg_return: float = 0.0
    min_val_selected_hit_rate: float = 0.52
    min_val_selected_avg_return: float = 0.0
    max_selected_per_split: int = 500
    threshold_grid: tuple[float, ...] = (0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72)


@dataclass
class TargetBundle:
    target: str
    target_id: int
    trading_index: pd.DatetimeIndex
    close: np.ndarray
    market_features: np.ndarray
    text_features: np.ndarray
    text_daily: pd.DataFrame


@dataclass(frozen=True)
class CandidateRecord:
    bundle_id: int
    target: str
    target_id: int
    date: pd.Timestamp
    pos: int
    rule_id: int
    rule_name: str
    horizon: int
    horizon_id: int
    label: int
    future_return: float
    rule_features: tuple[float, ...]


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


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(0.0)
    return pd.Series(0.0, index=df.index)


def event_rule_masks(text_daily: pd.DataFrame) -> dict[str, pd.Series]:
    intensity = _safe_col(text_daily, "tc_event_intensity_sum")
    pressure = _safe_col(text_daily, "tc_directional_pressure_max")
    if pressure.abs().sum() == 0:
        pressure = _safe_col(text_daily, "tc_directional_pressure_mean")

    return {
        "pre_tariff": _safe_col(text_daily, "tc_pre_tariff_sum") > 0,
        "pre_deal": _safe_col(text_daily, "tc_pre_deal_sum") > 0,
        "pre_relief": _safe_col(text_daily, "tc_pre_relief_sum") > 0,
        "pre_action": _safe_col(text_daily, "tc_pre_action_sum") > 0,
        "open_tariff": _safe_col(text_daily, "tc_open_tariff_sum") > 0,
        "open_deal": _safe_col(text_daily, "tc_open_deal_sum") > 0,
        "night_tariff": _safe_col(text_daily, "tc_night_tariff_sum") > 0,
        "tariff_streak": _safe_col(text_daily, "tc_tariff_streak_3d") > 0,
        "tariff_only": _safe_col(text_daily, "tc_tariff_only_day") > 0,
        "deal_over_tariff": _safe_col(text_daily, "tc_deal_over_tariff_day") > 0,
        "relief_positive": _safe_col(text_daily, "tc_relief_positive_day") > 0,
        "action_positive": _safe_col(text_daily, "tc_action_positive_day") > 0,
        "post_volume_spike": _safe_col(text_daily, "tc_post_volume_spike") > 0,
        "high_intensity_3": intensity >= 3,
        "high_intensity_5": intensity >= 5,
        "positive_pressure": pressure > 0.5,
        "negative_pressure": pressure < -0.5,
        "china_chip": (_safe_col(text_daily, "kw_china_sum") + _safe_col(text_daily, "kw_chips_sum")) > 0,
        "market_brag": _safe_col(text_daily, "tc_market_brag_sum") > 0,
        "fed_or_rates": _safe_col(text_daily, "tc_fed_sum") > 0,
    }


def rule_feature_frame(text_daily: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "post_count": _safe_col(text_daily, "post_count"),
        "event_intensity": _safe_col(text_daily, "tc_event_intensity_sum"),
        "directional_pressure": _safe_col(text_daily, "tc_directional_pressure_max"),
        "post_count_z7": _safe_col(text_daily, "tc_post_count_z7"),
        "tariff_count": _safe_col(text_daily, "tc_tariff_sum"),
        "deal_count": _safe_col(text_daily, "tc_deal_sum"),
        "relief_count": _safe_col(text_daily, "tc_relief_sum"),
        "attack_count": _safe_col(text_daily, "tc_attack_sum"),
        "weighted_vader": _safe_col(text_daily, "weighted_vader_weighted"),
        "viral_score": _safe_col(text_daily, "viral_score_max"),
    }
    return pd.DataFrame(cols, index=text_daily.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _generic_market_df(market_df: pd.DataFrame, target: str) -> pd.DataFrame:
    rename = {
        f"close_{target}": "close_target",
        f"close_ret_{target}": "close_ret_target",
        f"volume_{target}": "volume_target",
    }
    market_df = market_df.rename(columns=rename)
    seen: dict[str, int] = {}
    columns = []
    for col in market_df.columns:
        count = seen.get(col, 0)
        columns.append(col if count == 0 else f"{col}__dup{count}")
        seen[col] = count + 1
    market_df = market_df.copy()
    market_df.columns = columns
    return market_df


def load_bundles(base_dir: Path, targets: Iterable[str], config: Config) -> tuple[list[TargetBundle], list[str], list[str]]:
    text_candidates = [
        base_dir / "data/trump_nlp/trump_posts_features_2017_2026.csv",
        base_dir / "data/text/trump_posts_features_2017_2026.csv",
    ]
    text_path = next((p for p in text_candidates if p.exists()), text_candidates[0])
    print(f"text_features={text_path}")
    text_df = prepare_text_dataframe(text_path)
    raw: list[tuple[str, pd.DataFrame, CustomDataset]] = []
    market_cols: set[str] = set()
    text_cols: list[str] | None = None

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
                print(f"[skip] {target}: close_target missing after generic rename")
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
        if text_cols is None:
            text_cols = list(ds.text_feature_cols)

    if not raw:
        raise RuntimeError("no target bundles could be loaded")

    market_col_list = sorted(market_cols)
    text_col_list = text_cols or []
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
        text_daily = text_daily.reindex(columns=text_col_list, fill_value=0.0)
        bundles.append(
            TargetBundle(
                target=target,
                target_id=target_id,
                trading_index=ds.trading_index,
                close=market_df["close_target"].to_numpy(dtype=np.float32),
                market_features=market_features,
                text_features=text_daily.astype(np.float32).to_numpy(),
                text_daily=text_daily,
            )
        )
    return bundles, market_col_list, text_col_list


def build_candidates(bundles: list[TargetBundle], config: Config) -> tuple[list[CandidateRecord], list[str], list[str]]:
    records: list[CandidateRecord] = []
    rule_names = sorted(event_rule_masks(bundles[0].text_daily).keys())
    rule_feature_names = list(rule_feature_frame(bundles[0].text_daily).columns)

    for bundle_id, bundle in enumerate(bundles):
        masks = event_rule_masks(bundle.text_daily)
        feature_df = rule_feature_frame(bundle.text_daily)
        for rule_id, rule_name in enumerate(rule_names):
            mask = masks[rule_name].reindex(bundle.trading_index, fill_value=False).to_numpy(dtype=bool)
            trigger_positions = np.where(mask)[0]
            for pos in trigger_positions:
                if pos < config.window_size - 1:
                    continue
                for horizon_id, horizon in enumerate(config.horizons):
                    if pos + horizon >= len(bundle.close):
                        continue
                    close_t = float(bundle.close[pos])
                    close_h = float(bundle.close[pos + horizon])
                    if not np.isfinite(close_t) or not np.isfinite(close_h) or close_t <= 0:
                        continue
                    future_return = close_h / close_t - 1.0
                    features = tuple(float(v) for v in feature_df.iloc[pos].to_numpy(dtype=np.float32))
                    records.append(
                        CandidateRecord(
                            bundle_id=bundle_id,
                            target=bundle.target,
                            target_id=bundle.target_id,
                            date=pd.Timestamp(bundle.trading_index[pos]).normalize(),
                            pos=int(pos),
                            rule_id=rule_id,
                            rule_name=rule_name,
                            horizon=int(horizon),
                            horizon_id=horizon_id,
                            label=int(future_return >= 0.0),
                            future_return=float(future_return),
                            rule_features=features,
                        )
                    )

    records.sort(key=lambda r: (r.date, r.target, r.rule_id, r.horizon))
    return records, rule_names, rule_feature_names


def make_splits(records: list[CandidateRecord], config: Config) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
    splits = []
    for train_end_s, val_end_s, test_end_s in split_defs:
        train_end = pd.Timestamp(train_end_s)
        val_end = pd.Timestamp(val_end_s)
        test_end = pd.Timestamp(test_end_s)
        train_idx = np.where(dates <= train_end)[0]
        val_idx = np.where((dates > train_end) & (dates <= val_end))[0]
        test_idx = np.where((dates > val_end) & (dates <= test_end))[0]
        if min(len(train_idx), len(val_idx), len(test_idx)) < config.min_split_candidates:
            continue
        splits.append((train_idx, val_idx, test_idx))
        if len(splits) >= config.max_splits:
            break
    return splits


def fit_normalizer(records: list[CandidateRecord], indices: np.ndarray, bundles: list[TargetBundle], window_size: int) -> Normalizer:
    market_chunks = []
    text_chunks = []
    for i in indices:
        r = records[int(i)]
        b = bundles[r.bundle_id]
        s = r.pos - window_size + 1
        market_chunks.append(b.market_features[s : r.pos + 1])
        text_chunks.append(b.text_features[s : r.pos + 1])
    market = np.concatenate(market_chunks, axis=0)
    text = np.concatenate(text_chunks, axis=0)
    return Normalizer(
        market_mean=market.mean(axis=0, keepdims=True),
        market_std=np.where(market.std(axis=0, keepdims=True) < 1e-6, 1.0, market.std(axis=0, keepdims=True)),
        text_mean=text.mean(axis=0, keepdims=True),
        text_std=np.where(text.std(axis=0, keepdims=True) < 1e-6, 1.0, text.std(axis=0, keepdims=True)),
    )


class CandidateDataset(Dataset):
    def __init__(
        self,
        records: list[CandidateRecord],
        indices: np.ndarray,
        bundles: list[TargetBundle],
        normalizer: Normalizer,
        window_size: int,
    ) -> None:
        self.records = records
        self.indices = np.asarray(indices, dtype=np.int64)
        self.bundles = bundles
        self.normalizer = normalizer
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        record = self.records[int(self.indices[idx])]
        bundle = self.bundles[record.bundle_id]
        start = record.pos - self.window_size + 1
        market = (bundle.market_features[start : record.pos + 1] - self.normalizer.market_mean) / self.normalizer.market_std
        text = (bundle.text_features[start : record.pos + 1] - self.normalizer.text_mean) / self.normalizer.text_std
        return {
            "market": torch.tensor(market, dtype=torch.float32),
            "text": torch.tensor(text, dtype=torch.float32),
            "rule_id": torch.tensor(record.rule_id, dtype=torch.long),
            "target_id": torch.tensor(record.target_id, dtype=torch.long),
            "horizon_id": torch.tensor(record.horizon_id, dtype=torch.long),
            "rule_features": torch.tensor(record.rule_features, dtype=torch.float32),
            "label": torch.tensor(record.label, dtype=torch.long),
        }


class DeepTrumpCodeNet(nn.Module):
    def __init__(
        self,
        market_dim: int,
        text_dim: int,
        rule_feature_dim: int,
        n_rules: int,
        n_targets: int,
        n_horizons: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.market_lstm = nn.LSTM(market_dim, hidden_dim, batch_first=True)
        self.text_lstm = nn.LSTM(text_dim, hidden_dim, batch_first=True)
        self.rule_embedding = nn.Embedding(n_rules, 16)
        self.target_embedding = nn.Embedding(n_targets, 8)
        self.horizon_embedding = nn.Embedding(n_horizons, 4)
        self.rule_mlp = nn.Sequential(nn.Linear(rule_feature_dim, 16), nn.LayerNorm(16), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        nn.init.constant_(self.gate[0].bias, -1.5)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 16 + 8 + 4 + 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        _, (market_h, _) = self.market_lstm(batch["market"])
        _, (text_h, _) = self.text_lstm(batch["text"])
        market_feat = market_h[-1]
        text_feat = text_h[-1]
        gate = self.gate(torch.cat([market_feat, text_feat], dim=1))
        fused = gate * text_feat + (1.0 - gate) * market_feat
        features = torch.cat(
            [
                fused,
                self.rule_embedding(batch["rule_id"]),
                self.target_embedding(batch["target_id"]),
                self.horizon_embedding(batch["horizon_id"]),
                self.rule_mlp(batch["rule_features"]),
            ],
            dim=1,
        )
        return self.head(features)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train_one_model(
    train_ds: CandidateDataset,
    val_ds: CandidateDataset,
    config: Config,
    market_dim: int,
    text_dim: int,
    rule_feature_dim: int,
    n_rules: int,
    n_targets: int,
    device: torch.device,
) -> DeepTrumpCodeNet:
    model = DeepTrumpCodeNet(
        market_dim=market_dim,
        text_dim=text_dim,
        rule_feature_dim=rule_feature_dim,
        n_rules=n_rules,
        n_targets=n_targets,
        n_horizons=len(config.horizons),
        hidden_dim=config.hidden_dim,
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

        val_pred = predict_candidates(model, val_loader, device)
        f1 = f1_score(val_pred["label"], val_pred["pred"], average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_candidates(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    offset = 0
    dataset: CandidateDataset = loader.dataset  # type: ignore[assignment]
    with torch.no_grad():
        for batch in loader:
            batch_size = batch["label"].shape[0]
            batch_on_device = _move_batch(batch, device)
            proba = torch.softmax(model(batch_on_device), dim=1).cpu().numpy()
            pred = proba.argmax(axis=1)
            labels = batch["label"].numpy()
            for j in range(batch_size):
                record = dataset.records[int(dataset.indices[offset + j])]
                rows.append(
                    {
                        "target": record.target,
                        "target_id": record.target_id,
                        "date": record.date.strftime("%Y-%m-%d"),
                        "rule_id": record.rule_id,
                        "rule_name": record.rule_name,
                        "horizon": record.horizon,
                        "label": int(labels[j]),
                        "pred": int(pred[j]),
                        "proba_down": float(proba[j, 0]),
                        "proba_up": float(proba[j, 1]),
                        "confidence": float(proba[j].max()),
                        "future_return": record.future_return,
                    }
                )
            offset += batch_size
    return pd.DataFrame(rows)


def _long_only_candidates(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    selected = pred[(pred["pred"] == 1) & (pred["confidence"] >= threshold)].copy()
    selected["strategy_return"] = selected["future_return"]
    return selected


def choose_threshold(val_pred: pd.DataFrame, grid: Iterable[float], config: Config) -> float | None:
    best_threshold = None
    best_score = -1e9
    for threshold in grid:
        selected = _long_only_candidates(val_pred, threshold)
        if len(selected) < 25:
            continue
        selected = selected.sort_values("confidence", ascending=False).head(config.max_selected_per_split)
        strategy_return = selected["strategy_return"].to_numpy()
        hit_rate = (strategy_return > 0).mean()
        avg_return = strategy_return.mean()
        if hit_rate < config.min_val_selected_hit_rate or avg_return <= config.min_val_selected_avg_return:
            continue
        score = avg_return * math.sqrt(len(selected)) + 0.0025 * (hit_rate - 0.5)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def survivor_rules(val_pred: pd.DataFrame, threshold: float, config: Config) -> pd.DataFrame:
    columns = ["rule_name", "n_val_trades", "val_hit_rate", "val_avg_strategy_return"]
    selected = _long_only_candidates(val_pred, threshold)
    if selected.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for rule_name, group in selected.groupby("rule_name"):
        group = group.sort_values("confidence", ascending=False).head(config.max_selected_per_split)
        n = len(group)
        hit_rate = float((group["strategy_return"] > 0).mean())
        avg_return = float(group["strategy_return"].mean())
        if (
            n >= config.min_val_rule_trades
            and hit_rate >= config.min_val_rule_hit_rate
            and avg_return > config.min_val_rule_avg_return
        ):
            rows.append(
                {
                    "rule_name": rule_name,
                    "n_val_trades": n,
                    "val_hit_rate": hit_rate,
                    "val_avg_strategy_return": avg_return,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["val_hit_rate", "val_avg_strategy_return"], ascending=False)


def score_predictions(pred: pd.DataFrame, threshold: float | None, allowed_rules: set[str], config: Config) -> tuple[pd.DataFrame, dict[str, float]]:
    out = pred.copy()
    allowed = out["rule_name"].isin(allowed_rules).to_numpy()
    out["selected"] = False
    if threshold is not None and allowed_rules:
        long_mask = (out["pred"] == 1) & (out["confidence"] >= threshold) & allowed
        selected_idx = (
            out[long_mask]
            .sort_values("confidence", ascending=False)
            .head(config.max_selected_per_split)
            .index
        )
        out.loc[selected_idx, "selected"] = True
    out["direction"] = np.where(out["selected"], 1, 0)
    out["strategy_return"] = np.where(out["selected"], out["future_return"], 0.0)
    selected = out[out["selected"]].copy()
    if selected.empty:
        metrics = {
            "n_candidates": float(len(out)),
            "n_selected": 0.0,
            "hit_rate": 0.0,
            "avg_strategy_return": 0.0,
            "sum_strategy_return": 0.0,
            "macro_f1_all_candidates": f1_score(out["label"], out["pred"], average="macro", zero_division=0),
            "accuracy_all_candidates": accuracy_score(out["label"], out["pred"]),
            "precision_up_all_candidates": precision_score(out["label"], out["pred"], pos_label=1, zero_division=0),
        }
        return out, metrics
    metrics = {
        "n_candidates": float(len(out)),
        "n_selected": float(len(selected)),
        "hit_rate": float((selected["strategy_return"] > 0).mean()),
        "avg_strategy_return": float(selected["strategy_return"].mean()),
        "sum_strategy_return": float(selected["strategy_return"].sum()),
        "macro_f1_all_candidates": f1_score(out["label"], out["pred"], average="macro", zero_division=0),
        "accuracy_all_candidates": accuracy_score(out["label"], out["pred"]),
        "precision_up_all_candidates": precision_score(out["label"], out["pred"], pos_label=1, zero_division=0),
    }
    return out, metrics


def run_deep_trump_code(base_dir: Path, output_dir: Path, config: Config | None = None) -> None:
    config = config or Config()
    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    bundles, market_cols, text_cols = load_bundles(base_dir, TARGETS, config)
    records, rule_names, rule_feature_names = build_candidates(bundles, config)
    splits = make_splits(records, config)
    if not splits:
        raise RuntimeError(f"not enough candidate records for walk-forward: n={len(records)}")

    pd.DataFrame({"rule_id": range(len(rule_names)), "rule_name": rule_names}).to_csv(output_dir / "deep_trump_code_rules.csv", index=False)
    pd.DataFrame({"market_feature": market_cols}).to_csv(output_dir / "deep_trump_code_market_features.csv", index=False)
    pd.DataFrame({"text_feature": text_cols}).to_csv(output_dir / "deep_trump_code_text_features.csv", index=False)
    pd.DataFrame({"rule_feature": rule_feature_names}).to_csv(output_dir / "deep_trump_code_rule_features.csv", index=False)

    all_predictions = []
    summary_rows = []
    survivor_rows = []
    print(f"bundles={len(bundles)} candidates={len(records)} rules={len(rule_names)} splits={len(splits)}")

    for split_id, (train_idx, val_idx, test_idx) in enumerate(splits, start=1):
        normalizer = fit_normalizer(records, train_idx, bundles, config.window_size)
        train_ds = CandidateDataset(records, train_idx, bundles, normalizer, config.window_size)
        val_ds = CandidateDataset(records, val_idx, bundles, normalizer, config.window_size)
        test_ds = CandidateDataset(records, test_idx, bundles, normalizer, config.window_size)

        model = train_one_model(
            train_ds=train_ds,
            val_ds=val_ds,
            config=config,
            market_dim=bundles[0].market_features.shape[1],
            text_dim=bundles[0].text_features.shape[1],
            rule_feature_dim=len(rule_feature_names),
            n_rules=len(rule_names),
            n_targets=len(bundles),
            device=device,
        )

        val_pred = predict_candidates(model, DataLoader(val_ds, batch_size=config.batch_size), device)
        threshold = choose_threshold(val_pred, config.threshold_grid, config)
        survivors = survivor_rules(val_pred, threshold, config) if threshold is not None else pd.DataFrame()
        allowed_rules = set(survivors["rule_name"]) if not survivors.empty else set()
        if threshold is None:
            print(f"split={split_id}: validation regime filter failed; test will not trade")
        elif survivors.empty:
            print(f"split={split_id}: no survivor rules; test will not trade")

        test_pred = predict_candidates(model, DataLoader(test_ds, batch_size=config.batch_size), device)
        test_scored, metrics = score_predictions(test_pred, threshold, allowed_rules, config)
        test_scored.insert(0, "split", split_id)
        test_scored["threshold"] = np.nan if threshold is None else threshold
        all_predictions.append(test_scored)

        metrics.update(
            {
                "split": split_id,
                "threshold": np.nan if threshold is None else threshold,
                "n_survivor_rules": len(allowed_rules),
                "train_start": records[int(train_idx[0])].date.strftime("%Y-%m-%d"),
                "train_end": records[int(train_idx[-1])].date.strftime("%Y-%m-%d"),
                "val_start": records[int(val_idx[0])].date.strftime("%Y-%m-%d"),
                "val_end": records[int(val_idx[-1])].date.strftime("%Y-%m-%d"),
                "test_start": records[int(test_idx[0])].date.strftime("%Y-%m-%d"),
                "test_end": records[int(test_idx[-1])].date.strftime("%Y-%m-%d"),
            }
        )
        summary_rows.append(metrics)
        if not survivors.empty:
            survivors = survivors.copy()
            survivors.insert(0, "split", split_id)
            survivors["threshold"] = threshold
            survivor_rows.append(survivors)

        print(
            f"split={split_id} threshold={threshold:.2f} selected={int(metrics['n_selected'])}/"
            f"{int(metrics['n_candidates'])} hit={metrics['hit_rate']:.3f} "
            f"avg_ret={metrics['avg_strategy_return']:.4f} survivors={metrics['n_survivor_rules']}"
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(output_dir / "deep_trump_code_predictions.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "deep_trump_code_summary.csv", index=False)
    if survivor_rows:
        pd.concat(survivor_rows, ignore_index=True).to_csv(output_dir / "deep_trump_code_rule_survivors.csv", index=False)
    else:
        pd.DataFrame(columns=["split", "rule_name", "n_val_trades", "val_hit_rate", "val_avg_strategy_return", "threshold"]).to_csv(
            output_dir / "deep_trump_code_rule_survivors.csv", index=False
        )

    target_summary = (
        predictions[predictions["selected"]]
        .groupby("target")
        .agg(
            n_selected=("selected", "size"),
            hit_rate=("strategy_return", lambda x: float((x > 0).mean())),
            avg_strategy_return=("strategy_return", "mean"),
            sum_strategy_return=("strategy_return", "sum"),
        )
        .reset_index()
    )
    target_summary.to_csv(output_dir / "deep_trump_code_target_summary.csv", index=False)

    overall = {
        "n_splits": len(summary),
        "n_candidates": int(summary["n_candidates"].sum()),
        "n_selected": int(summary["n_selected"].sum()),
        "hit_rate_selected": float((predictions.loc[predictions["selected"], "strategy_return"] > 0).mean())
        if predictions["selected"].any()
        else 0.0,
        "avg_strategy_return_selected": float(predictions.loc[predictions["selected"], "strategy_return"].mean())
        if predictions["selected"].any()
        else 0.0,
        "sum_strategy_return_selected": float(predictions.loc[predictions["selected"], "strategy_return"].sum())
        if predictions["selected"].any()
        else 0.0,
        "macro_f1_all_candidates": float(f1_score(predictions["label"], predictions["pred"], average="macro", zero_division=0)),
        "accuracy_all_candidates": float(accuracy_score(predictions["label"], predictions["pred"])),
    }
    pd.DataFrame([overall]).to_csv(output_dir / "deep_trump_code_overall.csv", index=False)
    print("saved deep_trump_code outputs")
