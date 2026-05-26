import math
import os
from pathlib import Path
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score


@dataclass
class WalkForwardSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class CustomDataset(Dataset):
    """
    Dual-branch dataset that aligns:
      - Market branch: past N trading days of market features
      - Text branch: aggregated text features aligned to each trading day

    The critical logic is the "holiday aggregation":
    - Taiwan stock market has weekends/holidays, but text events can happen any day.
    - For each trading day D, we aggregate all text events from the previous trading
      day's close (13:30) up to the current trading day's open (09:00). If there are
      non-trading days (weekends/holidays), those events are also aggregated into D.
    - Implementation detail:
        1) For each text timestamp, we map it to a "target trading day".
        2) If the timestamp is on a trading day and after market close, it is assigned
           to the next trading day.
        3) If the timestamp is on a non-trading day, it is assigned to the next trading day.
        4) Otherwise (trading day and before close), it is assigned to the same day.
    - After mapping, we aggregate (mean / max / weighted mean) all text vectors that
      land on the same trading day, producing a single event-impact vector for that day.
    """

    def __init__(
        self,
        market_df: pd.DataFrame,
        text_df: pd.DataFrame,
        window_size: int,
        market_feature_cols: Optional[List[str]] = None,
        text_feature_cols: Optional[List[str]] = None,
        text_timestamp_col: str = "Timestamp",
        text_timezone: str = "UTC",
        market_timezone: str = "Asia/Taipei",
        market_open_time: str = "09:00",
        market_close_time: str = "13:30",
        aggregation: str = "weighted_mean",
        weight_col: Optional[str] = "Likes",
        label_series: Optional[pd.Series] = None,
        close_price_col: Optional[str] = None,
        open_price_col: Optional[str] = None,
        volatility_window: int = 20,
        z_score: float = 1.0,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")

        self.window_size = window_size

        market_df = market_df.copy()
        if not isinstance(market_df.index, pd.DatetimeIndex):
            market_df.index = pd.to_datetime(market_df.index)
        market_df = market_df.sort_index()

        self.trading_index = market_df.index.normalize()
        self.trading_dates = [d.date() for d in self.trading_index]
        self.trading_date_set = set(self.trading_dates)

        if market_feature_cols is None:
            market_feature_cols = [c for c in market_df.columns if pd.api.types.is_numeric_dtype(market_df[c])]
        self.market_feature_cols = market_feature_cols
        self.market_features = market_df[self.market_feature_cols].astype(np.float32).to_numpy()

        if label_series is None:
            if close_price_col is None or open_price_col is None:
                raise ValueError("Either label_series or close_price_col/open_price_col must be provided.")
            label_series = self._build_dynamic_gap_labels(
                close_series=market_df[close_price_col],
                open_series=market_df[open_price_col],
                volatility_window=volatility_window,
                z_score=z_score,
            )
        label_series = label_series.reindex(self.trading_index)

        self.labels = label_series.to_numpy()

        text_daily = self._aggregate_text_features(
            text_df=text_df,
            text_feature_cols=text_feature_cols,
            text_timestamp_col=text_timestamp_col,
            text_timezone=text_timezone,
            market_timezone=market_timezone,
            market_open_time=market_open_time,
            market_close_time=market_close_time,
            aggregation=aggregation,
            weight_col=weight_col,
        )
        self.text_features = text_daily.to_numpy(dtype=np.float32)

        valid_mask = ~pd.isna(self.labels)
        valid_indices = np.where(valid_mask)[0]

        # Need at least window_size days of history and a valid label.
        self.valid_indices = valid_indices[valid_indices >= (window_size - 1)]
        self.sample_index = self.trading_index[self.valid_indices]

    @staticmethod
    def _build_dynamic_gap_labels(
        close_series: pd.Series,
        open_series: pd.Series,
        volatility_window: int,
        z_score: float,
    ) -> pd.Series:
        # Step 1: 計算隔日跳空報酬率 (Open(T+1) vs Close(T))
        gap_returns = (open_series.shift(-1) - close_series) / close_series

        # Step 2: 計算跳空報酬率的滾動標準差
        rolling_std = gap_returns.rolling(window=volatility_window).std()

        # Step 3: 填補初期 NaN，避免標籤全遺失
        rolling_std = rolling_std.bfill()
        rolling_std = rolling_std.replace(0.0, np.nan).bfill()

        # Step 4: 動態門檻分類
        threshold = z_score * rolling_std

        labels = pd.Series(index=close_series.index, dtype="float32")
        labels[gap_returns < -threshold] = 0
        labels[(gap_returns >= -threshold) & (gap_returns <= threshold)] = 1
        labels[gap_returns > threshold] = 2
        return labels

    def _aggregate_text_features(
        self,
        text_df: pd.DataFrame,
        text_feature_cols: Optional[List[str]],
        text_timestamp_col: str,
        text_timezone: str,
        market_timezone: str,
        market_open_time: str,
        market_close_time: str,
        aggregation: str,
        weight_col: Optional[str],
    ) -> pd.DataFrame:
        if text_df is None or text_df.empty:
            return pd.DataFrame(
                np.zeros((len(self.trading_index), 1), dtype=np.float32),
                index=self.trading_index,
                columns=["text_placeholder"],
            )

        df = text_df.copy()
        if text_timestamp_col not in df.columns:
            raise ValueError(f"text_timestamp_col '{text_timestamp_col}' not found in text_df.")

        # Prepare one-hot for emotion_label if present.
        if "emotion_label" in df.columns and "emotion_label" not in (text_feature_cols or []):
            emotion_ohe = pd.get_dummies(df["emotion_label"], prefix="emo")
            df = pd.concat([df.drop(columns=["emotion_label"]), emotion_ohe], axis=1)

        if text_feature_cols is None:
            text_feature_cols = [
                c for c in df.columns if c != text_timestamp_col and pd.api.types.is_numeric_dtype(df[c])
            ]
        if not text_feature_cols:
            raise ValueError("No numeric text feature columns found for aggregation.")

        df[text_timestamp_col] = pd.to_datetime(df[text_timestamp_col], errors="coerce")
        df = df.dropna(subset=[text_timestamp_col])

        ts = df[text_timestamp_col]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(text_timezone)
        ts = ts.dt.tz_convert(market_timezone)

        market_open = pd.to_datetime(market_open_time).time()
        market_close = pd.to_datetime(market_close_time).time()

        trading_dates = self.trading_dates

        def next_trading_date(d) -> Optional[pd.Timestamp]:
            pos = bisect_right(trading_dates, d)
            if pos >= len(trading_dates):
                return None
            return trading_dates[pos]

        target_dates: List[Optional[pd.Timestamp]] = []
        for t in ts:
            d = t.date()
            tm = t.time()
            if d in self.trading_date_set:
                if tm >= market_close:
                    target_dates.append(next_trading_date(d))
                else:
                    target_dates.append(d)
            else:
                target_dates.append(next_trading_date(d))

        df["trade_date"] = target_dates
        df = df.dropna(subset=["trade_date"])

        def weighted_mean(group: pd.DataFrame) -> pd.Series:
            feats = group[text_feature_cols].to_numpy(dtype=np.float32)
            if weight_col is None or weight_col not in group.columns:
                return pd.Series(feats.mean(axis=0), index=text_feature_cols)
            weights = group[weight_col].to_numpy(dtype=np.float32)
            weights = np.maximum(weights, 0.0)
            if weights.sum() == 0:
                return pd.Series(feats.mean(axis=0), index=text_feature_cols)
            weighted = (feats.T * weights).T
            return pd.Series(weighted.sum(axis=0) / weights.sum(), index=text_feature_cols)

        if aggregation == "max":
            agg_df = df.groupby("trade_date")[text_feature_cols].max()
        elif aggregation == "mean":
            agg_df = df.groupby("trade_date")[text_feature_cols].mean()
        elif aggregation == "weighted_mean":
            agg_df = df.groupby("trade_date").apply(weighted_mean)
        else:
            raise ValueError("aggregation must be one of: 'mean', 'max', 'weighted_mean'")

        agg_df.index = pd.to_datetime(agg_df.index)
        agg_df = agg_df.sort_index()

        # Reindex to all trading days, fill missing with zeros (no events).
        agg_df = agg_df.reindex(self.trading_index, fill_value=0.0)
        return agg_df

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.valid_indices[idx]

        # Market sequence: [Seq, FeatureDim]
        market_seq = self.market_features[(t - self.window_size + 1) : (t + 1)]
        # Text vector: [TextDim]
        text_vec = self.text_features[t]
        label = int(self.labels[t])

        # Shape comments:
        # market_seq: [Seq, FeatureDim] -> torch [Seq, FeatureDim]
        # text_vec:   [TextDim] -> torch [TextDim]
        # label: scalar -> torch []
        return (
            torch.from_numpy(market_seq),
            torch.from_numpy(text_vec),
            torch.tensor(label, dtype=torch.long),
        )


class DualBranchNet(nn.Module):
    def __init__(
        self,
        text_dim: int,
        market_dim: int,
        hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.market_lstm = nn.LSTM(
            input_size=market_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, market_x: torch.Tensor, text_x: torch.Tensor) -> torch.Tensor:
        # market_x: [Batch, Seq, MarketDim]
        # text_x:   [Batch, TextDim]
        text_feat = self.text_mlp(text_x)  # [Batch, Hidden]

        lstm_out, (h_n, _) = self.market_lstm(market_x)
        market_feat = h_n[-1]  # [Batch, Hidden]

        fused = torch.cat([text_feat, market_feat], dim=1)  # [Batch, Hidden * 2]
        logits = self.fusion_head(fused)  # [Batch, 3]
        return logits


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Iterable[float]] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [Batch, 3], targets: [Batch]
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp().clamp(min=1e-8, max=1.0)

        targets = targets.long()
        num_classes = logits.size(1)
        true_dist = torch.zeros_like(probs)
        true_dist.fill_(self.label_smoothing / max(1, num_classes - 1))
        true_dist.scatter_(1, targets.view(-1, 1), 1.0 - self.label_smoothing)

        focal_weight = (1.0 - probs).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha.view(1, -1)
            focal_weight = focal_weight * alpha_t

        loss = -(true_dist * focal_weight * probs.log()).sum(dim=1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LogitAdjustedLoss(nn.Module):
    def __init__(
        self,
        class_priors: Iterable[float],
        tau: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.tau = tau
        self.reduction = reduction
        priors = torch.tensor(class_priors, dtype=torch.float32)
        self.register_buffer("log_prior", torch.log(priors))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.tau * self.log_prior
        return F.cross_entropy(adjusted, targets, reduction=self.reduction)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int = 20,
    grad_clip: Optional[float] = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> dict:
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for market_x, text_x, y in train_loader:
            market_x = market_x.to(device)
            text_x = text_x.to(device)
            y = y.to(device)

            logits = model(market_x, text_x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_count += y.size(0)

        avg_train_loss = total_loss / max(1, total_count)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_count = 0
        val_targets = []
        val_preds = []
        with torch.no_grad():
            for market_x, text_x, y in val_loader:
                market_x = market_x.to(device)
                text_x = text_x.to(device)
                y = y.to(device)

                logits = model(market_x, text_x)
                loss = loss_fn(logits, y)
                preds = logits.argmax(dim=1)

                val_loss += loss.item() * y.size(0)
                val_count += y.size(0)
                val_targets.append(y.cpu().numpy())
                val_preds.append(preds.cpu().numpy())

        avg_val_loss = val_loss / max(1, val_count)
        y_true = np.concatenate(val_targets) if val_targets else np.array([], dtype=int)
        y_pred = np.concatenate(val_preds) if val_preds else np.array([], dtype=int)
        # Macro F1（處理類別不平衡）
        val_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true.size else 0.0
        history["val_loss"].append(avg_val_loss)
        history["val_macro_f1"].append(val_macro_f1)

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_f1={val_macro_f1:.4f}"
        )

    return history


def compute_class_weights(labels: np.ndarray, num_classes: int = 3) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / counts
    return weights / weights.mean()


def compute_class_balanced_weights(labels: np.ndarray, num_classes: int = 3, beta: float = 0.99) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    return weights / weights.mean()


def evaluate_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for market_x, text_x, y in data_loader:
            market_x = market_x.to(device)
            text_x = text_x.to(device)
            y = y.to(device)

            logits = model(market_x, text_x)
            batch_preds = logits.argmax(dim=1)
            preds.append(batch_preds.cpu().numpy())
            targets.append(y.cpu().numpy())

    return np.concatenate(targets), np.concatenate(preds)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def compute_f1_scores(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3
) -> float:
    if y_true.size == 0:
        return 0.0

    f1s = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for cls in range(num_classes):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        tp_total += tp
        fp_total += fp
        fn_total += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    return macro_f1


def expanding_window_walk_forward(
    index: pd.DatetimeIndex,
    initial_train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
) -> List[WalkForwardSplit]:
    splits: List[WalkForwardSplit] = []
    total = len(index)
    train_end = initial_train_size

    while True:
        val_end = train_end + val_size
        test_end = val_end + test_size
        if test_end > total:
            break

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, test_end)
        splits.append(WalkForwardSplit(train_idx, val_idx, test_idx))

        train_end += step_size

    return splits


def load_market_features(
    prices_path: str,
    volumes_path: str,
    opens_path: str,
    inst_path: str,
    margin_path: str,
    tx_path: str,
    target_ticker: str = "2330.TW",
) -> pd.DataFrame:
    def read_market_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df = df.set_index("Date")
        elif "Unnamed: 0" in df.columns:
            df = df.set_index("Unnamed: 0")
        else:
            df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        return df

    prices = read_market_csv(prices_path)
    volumes = read_market_csv(volumes_path)
    opens = read_market_csv(opens_path)

    target_col = target_ticker
    if target_col not in prices.columns:
        raise ValueError(f"{target_col} not found in prices columns.")

    # === 價量特徵（台股目標標的） ===
    # 將價格/成交量轉為日變化率，避免非平穩序列影響模型
    close_returns = prices[[target_col]].pct_change().replace([np.inf, -np.inf], 0.0)  # 安全處理 inf
    open_returns = opens[[target_col]].pct_change().replace([np.inf, -np.inf], 0.0)  # 安全處理 inf
    volume_returns = volumes[[target_col]].pct_change().replace([np.inf, -np.inf], 0.0)  # 安全處理 inf

    # === 美股/總經特徵（共用） ===
    us_features = ["TSM", "^SOX", "^NDX", "^GSPC", "^VIX", "TWD=X", "^TNX"]
    us_features = [c for c in us_features if c in prices.columns]

    us_price_features = [c for c in us_features if c not in ["^VIX", "^TNX"]]
    us_returns = prices[us_price_features].pct_change().replace([np.inf, -np.inf], 0.0)  # 安全處理 inf
    us_diff = pd.DataFrame(index=prices.index)
    if "^VIX" in prices.columns:
        us_diff["^VIX"] = prices["^VIX"].diff()
    if "^TNX" in prices.columns:
        us_diff["^TNX"] = prices["^TNX"].diff()

    us_shifted = pd.concat([us_returns, us_diff], axis=1).shift(1)

    # === 三大法人（台股目標標的） ===
    inst_pivot = pd.DataFrame(index=prices.index)
    if target_ticker.endswith(".TW"):
        stock_id = target_ticker.replace(".TW", "")
        inst_df = pd.read_csv(inst_path, parse_dates=["date"])
        inst_df = inst_df[inst_df["stock_id"].astype(str) == stock_id]
        inst_df["net_buy"] = inst_df["buy"] - inst_df["sell"]
        inst_pivot = (
            inst_df.pivot_table(index="date", columns="name", values="net_buy", aggfunc="sum")
            .sort_index()
        )
        inst_pivot.columns = [f"inst_{c}_net_buy" for c in inst_pivot.columns]

    # === 融資融券（台股目標標的） ===
    margin_df = pd.DataFrame(index=prices.index)
    if target_ticker.endswith(".TW"):
        stock_id = target_ticker.replace(".TW", "")
        margin_raw = pd.read_csv(margin_path, parse_dates=["date"])
        margin_raw = margin_raw[margin_raw["stock_id"].astype(str) == stock_id]
        margin_raw = margin_raw.set_index("date").sort_index()
        margin_cols = ["MarginPurchaseTodayBalance", "ShortSaleTodayBalance"]
        margin_df = margin_raw[margin_cols].pct_change().replace([np.inf, -np.inf], 0.0)  # 安全處理 inf
        margin_df = margin_df.rename(
            columns={
                "MarginPurchaseTodayBalance": "margin_purchase_change",
                "ShortSaleTodayBalance": "short_sale_change",
            }
        )

    # === 台指期夜盤（總經/情緒） ===
    tx_df = pd.DataFrame(index=prices.index)
    if target_ticker.endswith(".TW"):
        tx_df = pd.read_csv(tx_path, parse_dates=["date"])
        if "trading_session" in tx_df.columns:
            tx_df = tx_df[tx_df["trading_session"] == "after_market"]
        tx_df = tx_df.set_index("date").sort_index()
        tx_df = tx_df[[c for c in ["spread_per", "volume"] if c in tx_df.columns]]
        if "volume" in tx_df.columns:
            tx_df["volume"] = tx_df["volume"].pct_change().replace([np.inf, -np.inf], 0.0)  # 安全處理 inf
        tx_df = tx_df.rename(columns={"spread_per": "tx_spread_per", "volume": "tx_volume_change"})

    features = [
        close_returns.rename(columns={target_col: f"close_{target_ticker}"}),
        open_returns.rename(columns={target_col: f"open_{target_ticker}"}),
        volume_returns.rename(columns={target_col: f"volume_{target_ticker}"}),
        us_shifted,
        inst_pivot,
        margin_df,
        tx_df,
    ]

    market_df = pd.concat(features, axis=1).sort_index()
    market_df = market_df.ffill().dropna()
    return market_df


def prepare_text_dataframe(text_path: str) -> pd.DataFrame:
    df = pd.read_csv(text_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, format="mixed")
    return df


if __name__ == "__main__":
    # === Example usage (paths follow the repo structure) ===
    VOLATILITY_WINDOW = 20
    Z_SCORE = 1.25
    TAU = 0.75  # 放寬極端類別的決策邊界以提升 Recall
    base_dir = Path(__file__).resolve().parents[1]
    text_paths = [
        base_dir / "data/trump_nlp/trump_posts_features_2017_2026.csv",
        base_dir / "data/trump_post_data/trump_posts_features_2017_2026.csv",
    ]
    text_path = next((p for p in text_paths if p.exists()), None)
    if text_path is None:
        raise FileNotFoundError(
            "Missing text features CSV. Expected one of: "
            + ", ".join(str(p) for p in text_paths)
        )
    text_df = prepare_text_dataframe(str(text_path))

    # 使用貼文資料集中所有數值型特徵（包含 kw_ 關鍵字與情緒 one-hot）
    text_feature_cols = None

    # 目標清單：台股預測標的
    target_list = ["2330.TW", "2454.TW", "0050.TW"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_rows = []
    metrics_rows = []
    output_dir = base_dir / "output"
    split_output_dir = output_dir / "split_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_output_dir.mkdir(parents=True, exist_ok=True)

    for target_ticker in target_list:
        market_df = load_market_features(
            prices_path=str(base_dir / "data/taiwan_market_data/global_prices.csv"),
            volumes_path=str(base_dir / "data/taiwan_market_data/global_volumes.csv"),
            opens_path=str(base_dir / "data/taiwan_market_data/global_opens.csv"),
            inst_path=str(base_dir / "data/taiwan_market_data/institutional_investors.csv"),
            margin_path=str(base_dir / "data/taiwan_market_data/margin_trading.csv"),
            tx_path=str(base_dir / "data/taiwan_market_data/tx_futures_night.csv"),
            target_ticker=target_ticker,
        )

        dataset = CustomDataset(
            market_df=market_df,
            text_df=text_df,
            window_size=20,
            text_feature_cols=text_feature_cols,
            close_price_col=f"close_{target_ticker}",
            open_price_col=f"open_{target_ticker}",
            volatility_window=VOLATILITY_WINDOW,
            z_score=Z_SCORE,
            aggregation="weighted_mean",
            weight_col="Retweets",
        )

        base_market_features = dataset.market_features.copy()
        base_text_features = dataset.text_features.copy()

        # Walk-forward validation setup
        splits = expanding_window_walk_forward(
            index=dataset.sample_index,
            initial_train_size=800,
            val_size=100,
            test_size=100,
            step_size=100,
        )

        print(f"=== Target {target_ticker} ===")
        all_val_targets = []
        all_val_preds = []
        for split_idx, split in enumerate(splits, start=1):
            dataset.market_features = base_market_features.copy()
            dataset.text_features = base_text_features.copy()

            train_subset = torch.utils.data.Subset(dataset, split.train_idx)
            val_subset = torch.utils.data.Subset(dataset, split.val_idx)

            train_labels = dataset.labels[dataset.valid_indices[split.train_idx]]
            val_labels = dataset.labels[dataset.valid_indices[split.val_idx]]

            train_day_indices = dataset.valid_indices[split.train_idx]
            market_train = dataset.market_features[train_day_indices]
            market_mean = market_train.mean(axis=0)
            market_std = market_train.std(axis=0)
            market_std = np.where(market_std < 1e-8, 1.0, market_std)
            dataset.market_features = (dataset.market_features - market_mean) / market_std

            text_train = dataset.text_features[train_day_indices]
            text_mean = text_train.mean(axis=0)
            text_std = text_train.std(axis=0)
            text_std = np.where(text_std < 1e-8, 1.0, text_std)
            dataset.text_features = (dataset.text_features - text_mean) / text_std

            majority_class = int(np.bincount(train_labels.astype(int), minlength=3).argmax())
            baseline_preds = np.full_like(val_labels, majority_class)
            baseline_macro_f1 = compute_f1_scores(
                val_labels.astype(int), baseline_preds.astype(int), num_classes=3
            )
            print(
                f"Split {split_idx} baseline_macro_f1={baseline_macro_f1:.4f}"
            )
            class_dist = np.bincount(train_labels.astype(int), minlength=3)
            class_props = class_dist / max(1, class_dist.sum())
            class_dist = np.bincount(train_labels.astype(int), minlength=3)
            class_props = class_dist / max(1, class_dist.sum())
            print(f"Split {split_idx} class_dist={class_dist.tolist()}")

            # 使用 shuffle=True 保留真實市場分佈，避免過度矯正決策邊界
            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

            sample_market, sample_text, _ = dataset[0]
            model = DualBranchNet(
                text_dim=sample_text.shape[0],
                market_dim=sample_market.shape[1],
                hidden_dim=128,
                lstm_layers=1,
                dropout=0.2,
            )

            # 降低 tau 以減少 Logit 修正強度，避免過度激進的決策邊界
            loss_fn = LogitAdjustedLoss(class_priors=class_props, tau=TAU)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

            # 單一 DataLoader 訓練流程（不使用 sampler）
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                epochs=5,
            )

            y_true, y_pred = evaluate_predictions(model, val_loader, device)
            all_val_targets.append(y_true)
            all_val_preds.append(y_pred)
            # 該 split 最終 Macro F1
            split_test_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
            pred_path = split_output_dir / f"preds_{target_ticker.replace('/', '_')}_split{split_idx}.csv"
            pred_df.to_csv(pred_path, index=False)

            cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
            cm_df = pd.DataFrame(cm, index=["true_0", "true_1", "true_2"], columns=["pred_0", "pred_1", "pred_2"])
            cm_path = split_output_dir / f"cm_{target_ticker.replace('/', '_')}_split{split_idx}.csv"
            cm_df.to_csv(cm_path, index=True)

            best_idx = int(np.argmax(history["val_macro_f1"]))
            results_rows.append(
                {
                    "target": target_ticker,
                    "split": split_idx,
                    "best_epoch": best_idx + 1,
                    "best_val_macro_f1": history["val_macro_f1"][best_idx],
                    "best_val_loss": history["val_loss"][best_idx],
                    "split_test_macro_f1": split_test_macro_f1,
                }
            )

        target_rows = [r for r in results_rows if r["target"] == target_ticker]
        if target_rows:
            avg_best_f1 = sum(r["best_val_macro_f1"] for r in target_rows) / len(target_rows)
            print(f"=== Target {target_ticker} avg_best_f1={avg_best_f1:.4f} ===")
        if all_val_targets:
            y_true = np.concatenate(all_val_targets)
            y_pred = np.concatenate(all_val_preds)
            # 整體 Macro F1
            macro_avg_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            metrics_rows.append(
                {"target": target_ticker, "macro_avg_f1": macro_avg_f1}
            )
            print(f"=== Target {target_ticker} macro_avg_f1={macro_avg_f1:.4f} ===")

    if metrics_rows:
        metrics_path = output_dir / "training_metrics_avg.csv"
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")
