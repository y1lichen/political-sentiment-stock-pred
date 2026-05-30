import argparse
import json
import random
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


PRICE_PATH = "data/taiwan_market_data/global_prices.csv"
EVENT_PATH = "data/output/trump_daily_binary_event_features.csv"
INSTITUTION_PATH = "data/taiwan_market_data/institutional_investors.csv"
MARGIN_PATH = "data/taiwan_market_data/margin_trading.csv"
FUTURES_NIGHT_PATH = "data/taiwan_market_data/tx_futures_night.csv"

COUNT_EVENT_COLS = {
    "post_count",
    "tariff_count",
    "deal_count",
    "relief_count",
    "china_count",
    "taiwan_count",
    "chips_count",
    "ai_count",
    "night_post_count",
    "pre_post_count",
    "open_post_count",
    "total_excl",
    "avg_post_len",
}

REGIME_LABELS = {
    0: "calm_bull",
    1: "risk_off",
    2: "oversold",
    3: "neutral",
}

DIRECTION_LABELS = {
    0: "down",
    1: "up",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trump event brute-force + deep learning market-regime fusion model "
            "for Taiwan stock next-day return classification."
        )
    )
    parser.add_argument("--target", default="2330.TW", help="Target ticker in global_prices.csv.")
    parser.add_argument("--hold", type=int, default=1, help="Prediction horizon in trading days.")
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=0.0,
        help="Future return threshold for binary label. > threshold is up, otherwise down.",
    )
    parser.add_argument("--window", type=int, default=20, help="LSTM lookback window.")
    parser.add_argument("--min-n", type=int, default=20, help="Minimum samples for a brute-force event combo.")
    parser.add_argument("--min-abs-mean-ret", type=float, default=0.001, help="Minimum absolute mean return.")
    parser.add_argument("--min-hit-rate", type=float, default=0.55, help="Minimum directional hit rate.")
    parser.add_argument("--top-k-events", type=int, default=80, help="Max brute-force event combos used by DL.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/output/deep_regime_fusion")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_price_data(path, target):
    price_df = pd.read_csv(path)
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    price_df = price_df.sort_values("Date").set_index("Date")
    if target not in price_df.columns:
        raise ValueError(f"Target {target!r} not found in {path}.")
    return price_df


def read_event_data(path):
    event_df = pd.read_csv(path)
    event_df["trump_date"] = pd.to_datetime(event_df["trump_date"])
    return event_df.sort_values("trump_date").set_index("trump_date")


def binary_event_columns(event_df):
    cols = []
    for col in event_df.columns:
        if col in COUNT_EVENT_COLS:
            continue
        values = event_df[col].dropna().unique()
        if len(values) > 0 and set(values).issubset({0, 1, 0.0, 1.0}):
            cols.append(col)
    return cols


def make_event_combos(event_df, max_combo_size=2):
    event_cols = binary_event_columns(event_df)
    events = event_df[event_cols].astype("int8")
    combo_data = {col: events[col] for col in event_cols}

    if max_combo_size >= 2:
        for a, b in combinations(event_cols, 2):
            combo_data[f"{a}&{b}"] = ((events[a] == 1) & (events[b] == 1)).astype("int8")

    combo_df = pd.DataFrame(combo_data, index=events.index)
    return combo_df, event_cols


def brute_force_events(combo_df, price, hold, min_n, min_abs_mean_ret, min_hit_rate, top_k):
    future_ret = price.shift(-hold) / price - 1
    data = combo_df.join(future_ret.rename("future_ret"), how="inner")

    rows = []
    selected = []
    for col in combo_df.columns:
        mask = data[col] == 1
        n = int(mask.sum())
        if n < min_n:
            continue

        returns = data.loc[mask, "future_ret"].dropna()
        if returns.empty:
            continue

        mean_ret = float(returns.mean())
        hit_up = float((returns > 0).mean())
        hit_down = float((returns < 0).mean())
        direction = 1 if mean_ret >= 0 else -1
        directional_hit = hit_up if direction > 0 else hit_down
        score = abs(mean_ret) * np.sqrt(n) * directional_hit

        row = {
            "event": col,
            "n": n,
            "mean_ret": mean_ret,
            "median_ret": float(returns.median()),
            "std": float(returns.std()),
            "hit_up": hit_up,
            "hit_down": hit_down,
            "direction": "long" if direction > 0 else "short",
            "directional_hit": directional_hit,
            "score": float(score),
        }
        rows.append(row)

        if abs(mean_ret) >= min_abs_mean_ret and directional_hit >= min_hit_rate:
            selected.append(row)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("No event combo has enough samples. Lower --min-n.")

    result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)
    selected_df = pd.DataFrame(selected).sort_values("score", ascending=False)
    if selected_df.empty:
        selected_df = result_df.head(top_k).copy()
    else:
        selected_df = selected_df.head(top_k).copy()

    return selected_df["event"].tolist(), result_df, selected_df


def add_institution_features(base, path, target):
    stock_id = target.split(".")[0]
    p = Path(path)
    if not p.exists():
        return base

    inst = pd.read_csv(p)
    inst = inst[inst["stock_id"].astype(str) == stock_id].copy()
    if inst.empty:
        return base

    inst["date"] = pd.to_datetime(inst["date"])
    inst["net_buy"] = inst["buy"] - inst["sell"]
    wide = inst.pivot_table(index="date", columns="name", values="net_buy", aggfunc="sum")
    wide = wide.add_prefix("inst_net_")
    wide["inst_net_total"] = wide.sum(axis=1)
    return base.join(wide, how="left")


def add_margin_features(base, path, target):
    stock_id = target.split(".")[0]
    p = Path(path)
    if not p.exists():
        return base

    margin = pd.read_csv(p)
    margin = margin[margin["stock_id"].astype(str) == stock_id].copy()
    if margin.empty:
        return base

    margin["date"] = pd.to_datetime(margin["date"])
    margin = margin.set_index("date").sort_index()
    cols = [
        "MarginPurchaseTodayBalance",
        "ShortSaleTodayBalance",
        "MarginPurchaseBuy",
        "MarginPurchaseSell",
        "ShortSaleBuy",
        "ShortSaleSell",
    ]
    cols = [c for c in cols if c in margin.columns]
    margin = margin[cols].add_prefix("margin_")
    margin["margin_balance_chg"] = margin["margin_MarginPurchaseTodayBalance"].pct_change()
    margin["short_balance_chg"] = margin["margin_ShortSaleTodayBalance"].pct_change()
    return base.join(margin, how="left")


def add_futures_night_features(base, path):
    p = Path(path)
    if not p.exists():
        return base

    futures = pd.read_csv(p)
    futures["date"] = pd.to_datetime(futures["date"])
    futures = futures.sort_values("date").set_index("date")
    cols = [c for c in ["spread", "spread_per", "volume"] if c in futures.columns]
    futures = futures[cols].add_prefix("tx_night_")
    futures["tx_night_volume_z20"] = (
        futures["tx_night_volume"] - futures["tx_night_volume"].rolling(20).mean()
    ) / futures["tx_night_volume"].rolling(20).std()
    return base.join(futures, how="left")


def make_market_features(price_df, target):
    price = price_df[target].dropna()
    out = pd.DataFrame(index=price_df.index)
    out["target_ret_1d"] = price.pct_change()
    out["target_ret_3d"] = price.pct_change(3)
    out["target_ret_5d"] = price.pct_change(5)
    out["target_ret_10d"] = price.pct_change(10)
    out["target_vol_10d"] = out["target_ret_1d"].rolling(10).std()
    out["target_vol_20d"] = out["target_ret_1d"].rolling(20).std()
    out["target_ma_gap_5"] = price / price.rolling(5).mean() - 1
    out["target_ma_gap_20"] = price / price.rolling(20).mean() - 1
    out["target_ma_gap_60"] = price / price.rolling(60).mean() - 1
    rolling_high = price.rolling(60, min_periods=20).max()
    out["target_drawdown_60"] = price / rolling_high - 1

    for col in ["TSM", "^SOX", "^NDX", "^GSPC", "TWD=X"]:
        if col in price_df.columns:
            out[f"{col}_ret_1d"] = price_df[col].pct_change()
            out[f"{col}_ret_5d"] = price_df[col].pct_change(5)

    if "^VIX" in price_df.columns:
        vix = price_df["^VIX"]
        out["vix_level"] = vix
        out["vix_diff_1d"] = vix.diff()
        out["vix_pct_chg"] = vix.pct_change()
        out["vix_z20"] = (vix - vix.rolling(20).mean()) / vix.rolling(20).std()
        out["vix_low"] = (vix < vix.rolling(252, min_periods=60).quantile(0.35)).astype(float)
        out["vix_high"] = (vix > vix.rolling(252, min_periods=60).quantile(0.75)).astype(float)

    if "^TNX" in price_df.columns:
        out["tnx_diff_1d"] = price_df["^TNX"].diff()

    out = add_institution_features(out, INSTITUTION_PATH, target)
    out = add_margin_features(out, MARGIN_PATH, target)
    out = add_futures_night_features(out, FUTURES_NIGHT_PATH)
    return out


def add_regime_features(df):
    out = df.copy()
    vix_high = out.get("vix_high", pd.Series(0, index=out.index)).fillna(0) > 0
    vix_low = out.get("vix_low", pd.Series(0, index=out.index)).fillna(0) > 0
    bull = out["target_ma_gap_60"].fillna(0) > 0
    weak = out["target_ma_gap_20"].fillna(0) < -0.03
    oversold = out["target_drawdown_60"].fillna(0) < -0.08
    recent_drop = out["target_ret_5d"].fillna(0) < -0.04

    out["regime_calm_bull"] = (bull & vix_low).astype(float)
    out["regime_risk_off"] = (vix_high & (weak | recent_drop)).astype(float)
    out["regime_oversold"] = (oversold | recent_drop).astype(float)
    out["regime_neutral"] = (
        (out["regime_calm_bull"] == 0)
        & (out["regime_risk_off"] == 0)
        & (out["regime_oversold"] == 0)
    ).astype(float)

    out["regime_label"] = 3
    out.loc[out["regime_calm_bull"] == 1, "regime_label"] = 0
    out.loc[out["regime_risk_off"] == 1, "regime_label"] = 1
    out.loc[(out["regime_oversold"] == 1) & (out["regime_risk_off"] == 0), "regime_label"] = 2
    return out


def add_event_regime_interactions(df, selected_events):
    out = df.copy()
    regimes = ["regime_calm_bull", "regime_risk_off", "regime_oversold"]
    seed_events = [
        c
        for c in selected_events
        if any(key in c for key in ["tariff", "china", "taiwan", "chips", "deal", "relief"])
    ][:30]
    for event_col in seed_events:
        for regime_col in regimes:
            out[f"{event_col}__x__{regime_col}"] = out[event_col] * out[regime_col]
    return out


def build_model_frame(price_df, event_df, selected_events, combo_df, target, hold, binary_threshold):
    market = make_market_features(price_df, target)
    events = combo_df[selected_events].copy()
    counts = event_df[[c for c in COUNT_EVENT_COLS if c in event_df.columns]].copy()
    frame = market.join(counts, how="left").join(events, how="left")
    event_like_cols = list(counts.columns) + selected_events
    frame[event_like_cols] = frame[event_like_cols].fillna(0)

    frame = add_regime_features(frame)
    frame = add_event_regime_interactions(frame, selected_events)

    price = price_df[target].dropna()
    future_ret = price.shift(-hold) / price - 1
    frame["future_ret"] = future_ret
    frame["direction_label"] = (frame["future_ret"] > binary_threshold).astype(int)

    frame = frame.replace([np.inf, -np.inf], np.nan).sort_index()
    frame = frame.dropna(subset=["future_ret", "direction_label", "regime_label"])
    return frame


class FusionDataset(Dataset):
    def __init__(self, df, market_cols, event_cols, window):
        self.market_x = df[market_cols].astype(np.float32).values
        self.event_x = df[event_cols].astype(np.float32).values
        self.y = df["direction_label"].astype(np.int64).values
        self.regime_y = df["regime_label"].astype(np.int64).values
        self.future_ret = df["future_ret"].astype(np.float32).values
        self.dates = df.index.astype(str).to_numpy()
        self.window = window

    def __len__(self):
        return len(self.y) - self.window

    def __getitem__(self, idx):
        end = idx + self.window
        return {
            "market_x": torch.tensor(self.market_x[idx:end]),
            "event_x": torch.tensor(self.event_x[end]),
            "y": torch.tensor(self.y[end]),
            "regime_y": torch.tensor(self.regime_y[end]),
            "future_ret": torch.tensor(self.future_ret[end]),
            "date": self.dates[end],
        }


class RegimeFusionLSTM(nn.Module):
    def __init__(self, market_dim, event_dim, hidden_dim, dropout):
        super().__init__()
        self.market_lstm = nn.LSTM(
            market_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.regime_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, len(REGIME_LABELS)),
        )
        self.event_encoder = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.direction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(DIRECTION_LABELS)),
        )

    def forward(self, market_x, event_x):
        seq_out, _ = self.market_lstm(market_x)
        market_state = seq_out[:, -1, :]
        event_state = self.event_encoder(event_x)
        gate = self.gate(torch.cat([market_state, event_state], dim=1))
        fused_event = gate * event_state
        logits = self.direction_head(torch.cat([market_state, fused_event], dim=1))
        regime_logits = self.regime_head(market_state)
        return logits, regime_logits, gate


def split_and_scale(frame, market_cols, event_cols, window):
    n = len(frame)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = frame.iloc[:train_end].copy()
    val_df = frame.iloc[train_end - window : val_end].copy()
    test_df = frame.iloc[val_end - window :].copy()

    market_scaler = StandardScaler()
    event_scaler = StandardScaler()

    train_df[market_cols] = market_scaler.fit_transform(train_df[market_cols].fillna(0))
    val_df[market_cols] = market_scaler.transform(val_df[market_cols].fillna(0))
    test_df[market_cols] = market_scaler.transform(test_df[market_cols].fillna(0))

    train_df[event_cols] = event_scaler.fit_transform(train_df[event_cols].fillna(0))
    val_df[event_cols] = event_scaler.transform(val_df[event_cols].fillna(0))
    test_df[event_cols] = event_scaler.transform(test_df[event_cols].fillna(0))

    return train_df, val_df, test_df, market_scaler, event_scaler


def batch_to_device(batch, device):
    return {
        "market_x": batch["market_x"].to(device),
        "event_x": batch["event_x"].to(device),
        "y": batch["y"].to(device),
        "regime_y": batch["regime_y"].to(device),
        "future_ret": batch["future_ret"].to(device),
        "date": batch["date"],
    }


def evaluate(model, loader, criterion, regime_criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    preds = []
    labels = []
    regime_preds = []
    regime_labels = []
    returns = []
    dates = []
    probs = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = batch_to_device(raw_batch, device)
            logits, regime_logits, _ = model(batch["market_x"], batch["event_x"])
            loss = criterion(logits, batch["y"]) + 0.25 * regime_criterion(
                regime_logits, batch["regime_y"]
            )
            prob = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            regime_pred = regime_logits.argmax(dim=1)

            total_loss += float(loss.item()) * len(batch["y"])
            total += len(batch["y"])
            preds.extend(pred.cpu().numpy().tolist())
            labels.extend(batch["y"].cpu().numpy().tolist())
            regime_preds.extend(regime_pred.cpu().numpy().tolist())
            regime_labels.extend(batch["regime_y"].cpu().numpy().tolist())
            returns.extend(batch["future_ret"].cpu().numpy().tolist())
            dates.extend(list(raw_batch["date"]))
            probs.extend(prob.cpu().numpy().tolist())

    acc = accuracy_score(labels, preds) if labels else 0.0
    regime_acc = accuracy_score(regime_labels, regime_preds) if regime_labels else 0.0
    return {
        "loss": total_loss / max(total, 1),
        "acc": acc,
        "regime_acc": regime_acc,
        "preds": preds,
        "labels": labels,
        "regime_preds": regime_preds,
        "regime_labels": regime_labels,
        "returns": returns,
        "dates": dates,
        "probs": probs,
    }


def train_model(args, train_ds, val_ds, market_dim, event_dim, device):
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = RegimeFusionLSTM(
        market_dim=market_dim,
        event_dim=event_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    regime_criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for raw_batch in train_loader:
            batch = batch_to_device(raw_batch, device)
            optimizer.zero_grad()
            logits, regime_logits, _ = model(batch["market_x"], batch["event_x"])
            loss = criterion(logits, batch["y"]) + 0.25 * regime_criterion(
                regime_logits, batch["regime_y"]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["y"])
            total += len(batch["y"])

        train_loss = total_loss / max(total, 1)
        val_metrics = evaluate(model, val_loader, criterion, regime_criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_regime_acc": val_metrics["regime_acc"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={row['val_loss']:.4f} | val_acc={row['val_acc']:.4f} | "
            f"val_regime_acc={row['val_regime_acc']:.4f}"
        )

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stop at epoch {epoch}; best_val_loss={best_val_loss:.4f}")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predictions_frame(metrics):
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(metrics["dates"]),
            "actual_label": metrics["labels"],
            "pred_label": metrics["preds"],
            "actual_regime": metrics["regime_labels"],
            "pred_regime": metrics["regime_preds"],
            "future_ret": metrics["returns"],
        }
    )
    probs = pd.DataFrame(
        metrics["probs"],
        columns=[f"prob_{DIRECTION_LABELS[i]}" for i in range(len(DIRECTION_LABELS))],
    )
    out = pd.concat([out, probs], axis=1)
    out["actual_label_name"] = out["actual_label"].map(DIRECTION_LABELS)
    out["pred_label_name"] = out["pred_label"].map(DIRECTION_LABELS)
    out["actual_regime_name"] = out["actual_regime"].map(REGIME_LABELS)
    out["pred_regime_name"] = out["pred_regime"].map(REGIME_LABELS)
    signal = out["pred_label"].map({0: -1, 1: 1}).astype(float)
    out["strategy_ret_no_cost"] = signal * out["future_ret"]
    return out


def regime_conditioned_stats(frame, selected_events):
    rows = []
    regime_cols = ["regime_calm_bull", "regime_risk_off", "regime_oversold", "regime_neutral"]
    focus_events = [
        c
        for c in selected_events
        if any(key in c for key in ["tariff", "china", "taiwan", "chips", "deal", "relief"])
    ][:40]

    for event_col in focus_events:
        for regime_col in regime_cols:
            mask = (frame[event_col] > 0) & (frame[regime_col] > 0)
            n = int(mask.sum())
            if n < 5:
                continue
            r = frame.loc[mask, "future_ret"]
            rows.append(
                {
                    "event": event_col,
                    "regime": regime_col.replace("regime_", ""),
                    "n": n,
                    "mean_ret": float(r.mean()),
                    "median_ret": float(r.median()),
                    "hit_up": float((r > 0).mean()),
                    "hit_down": float((r < 0).mean()),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["event", "regime"]).reset_index(drop=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    price_df = read_price_data(PRICE_PATH, args.target)
    event_df = read_event_data(EVENT_PATH)
    price = price_df[args.target].dropna()
    combo_df, base_event_cols = make_event_combos(event_df)
    selected_events, brute_df, selected_df = brute_force_events(
        combo_df=combo_df,
        price=price,
        hold=args.hold,
        min_n=args.min_n,
        min_abs_mean_ret=args.min_abs_mean_ret,
        min_hit_rate=args.min_hit_rate,
        top_k=args.top_k_events,
    )

    print(f"Base binary event count: {len(base_event_cols)}")
    print(f"Event + combo count: {combo_df.shape[1]}")
    print(f"Selected events for DL: {len(selected_events)}")
    print("\nTop selected brute-force events:")
    print(selected_df.head(20).to_string(index=False))

    frame = build_model_frame(
        price_df=price_df,
        event_df=event_df,
        selected_events=selected_events,
        combo_df=combo_df,
        target=args.target,
        hold=args.hold,
        binary_threshold=args.binary_threshold,
    )

    all_event_cols = selected_events + [
        c for c in frame.columns if "__x__regime_" in c
    ] + [c for c in COUNT_EVENT_COLS if c in frame.columns]
    all_event_cols = list(dict.fromkeys(all_event_cols))
    excluded = set(all_event_cols + ["future_ret", "direction_label", "regime_label"])
    market_cols = [c for c in frame.columns if c not in excluded]

    usable = frame[market_cols + all_event_cols + ["future_ret", "direction_label", "regime_label"]].copy()
    usable[market_cols + all_event_cols] = usable[market_cols + all_event_cols].fillna(0)
    usable = usable.dropna()

    if len(usable) <= args.window + 50:
        raise ValueError(
            f"Not enough usable rows ({len(usable)}) for window={args.window}. "
            "Lower --window or check input data."
        )

    train_df, val_df, test_df, market_scaler, event_scaler = split_and_scale(
        usable, market_cols, all_event_cols, args.window
    )
    train_ds = FusionDataset(train_df, market_cols, all_event_cols, args.window)
    val_ds = FusionDataset(val_df, market_cols, all_event_cols, args.window)
    test_ds = FusionDataset(test_df, market_cols, all_event_cols, args.window)

    print("\nDataset summary:")
    print(f"Rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Market features: {len(market_cols)} | Event/regime features: {len(all_event_cols)}")
    print("Direction label counts:")
    print(usable["direction_label"].map(DIRECTION_LABELS).value_counts().to_string())
    print("Regime label counts:")
    print(usable["regime_label"].map(REGIME_LABELS).value_counts().to_string())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on device: {device}")
    model, history = train_model(
        args=args,
        train_ds=train_ds,
        val_ds=val_ds,
        market_dim=len(market_cols),
        event_dim=len(all_event_cols),
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    regime_criterion = nn.CrossEntropyLoss()
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_metrics = evaluate(model, test_loader, criterion, regime_criterion, device)
    pred_df = predictions_frame(test_metrics)

    y_true = pred_df["actual_label"].to_numpy()
    y_pred = pred_df["pred_label"].to_numpy()
    report = classification_report(
        y_true,
        y_pred,
        labels=list(DIRECTION_LABELS.keys()),
        target_names=[DIRECTION_LABELS[i] for i in DIRECTION_LABELS],
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(DIRECTION_LABELS.keys()))

    brute_df.to_csv(output_dir / "brute_force_all_events.csv", index=False)
    selected_df.to_csv(output_dir / "brute_force_selected_events.csv", index=False)
    history.to_csv(output_dir / "training_history.csv", index=False)
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
    regime_stats = regime_conditioned_stats(usable, selected_events)
    regime_stats.to_csv(output_dir / "event_regime_conditioned_stats.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "market_cols": market_cols,
            "event_cols": all_event_cols,
            "direction_labels": DIRECTION_LABELS,
            "regime_labels": REGIME_LABELS,
        },
        output_dir / "regime_fusion_model.pt",
    )
    joblib.dump(market_scaler, output_dir / "market_scaler.joblib")
    joblib.dump(event_scaler, output_dir / "event_scaler.joblib")

    summary = {
        "target": args.target,
        "hold": args.hold,
        "binary_threshold": args.binary_threshold,
        "window": args.window,
        "device": device,
        "rows": {
            "usable": int(len(usable)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "features": {
            "market": len(market_cols),
            "event_regime": len(all_event_cols),
            "selected_events": len(selected_events),
        },
        "test": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["acc"],
            "regime_accuracy": test_metrics["regime_acc"],
            "strategy_mean_return_no_cost": float(pred_df["strategy_ret_no_cost"].mean()),
            "strategy_total_return_no_cost": float((1 + pred_df["strategy_ret_no_cost"]).prod() - 1),
        },
        "classification_report": report,
        "confusion_matrix_labels": [DIRECTION_LABELS[i] for i in DIRECTION_LABELS],
        "confusion_matrix": cm.tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n==============================")
    print(f"Target: {args.target}")
    print(f"Selected brute-force events: {len(selected_events)}")
    print(f"Test acc: {test_metrics['acc']:.4f}")
    print(f"Test regime acc: {test_metrics['regime_acc']:.4f}")
    print(
        "Strategy return no cost: "
        f"mean={summary['test']['strategy_mean_return_no_cost']:.6f}, "
        f"total={summary['test']['strategy_total_return_no_cost']:.4f}"
    )
    print("Confusion matrix rows=true, cols=pred [down, up]:")
    print(cm)
    print(f"Saved outputs to: {output_dir}")
    print("==============================")


if __name__ == "__main__":
    main()
