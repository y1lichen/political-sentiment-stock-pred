from itertools import combinations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================================
# 0. Config
# =====================================================

PRICE_PATH = "data/taiwan_market_data/global_prices.csv"
EVENT_PATH = "data/output/trump_daily_binary_event_features.csv"

TARGET = "2330.TW"   # 先測台積電，可改成 0050.TW 等
WINDOW = 20
HOLD = 1

MIN_N = 20
MIN_HIT_RATE = 0.55
MIN_MEAN_RET = 0.001

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

# =====================================================
# 1. Load data
# =====================================================

price_df = pd.read_csv(PRICE_PATH)
price_df["Date"] = pd.to_datetime(price_df["Date"])
price_df = price_df.sort_values("Date").set_index("Date")

event_df = pd.read_csv(EVENT_PATH)
event_df["trump_date"] = pd.to_datetime(event_df["trump_date"])
event_df = event_df.sort_values("trump_date").set_index("trump_date")

price = price_df[TARGET].dropna()

# =====================================================
# 2. Select binary event columns
# =====================================================

exclude_cols = {
    "post_count", "tariff_count", "deal_count", "relief_count",
    "china_count", "taiwan_count", "chips_count", "ai_count",
    "night_post_count", "pre_post_count", "open_post_count",
    "total_excl", "avg_post_len"
}

event_cols = [
    c for c in event_df.columns
    if c not in exclude_cols
    and event_df[c].dropna().isin([0, 1]).all()
]

events = event_df[event_cols].copy()

# =====================================================
# 3. Create event combos
# =====================================================
combo_data = {}

for a, b in combinations(event_cols, 2):
    combo_name = f"{a}&{b}"

    combo_data[combo_name] = (
        (events[a] == 1) &
        (events[b] == 1)
    ).astype("int8")

combo_df = pd.DataFrame(combo_data, index=events.index)
print("Original event count:", len(event_cols))
print("Event + combo count:", combo_df.shape[1])

# =====================================================
# 4. Brute force event filtering
# =====================================================

returns = pd.DataFrame(index=price.index)
returns["future_ret"] = price.shift(-HOLD) / price - 1

data = combo_df.join(returns, how="inner")

selected_events = []
brute_results = []

for col in combo_df.columns:
    mask = data[col] == 1
    n = int(mask.sum())

    if n < MIN_N:
        continue

    r = data.loc[mask, "future_ret"].dropna()

    mean_ret = r.mean()
    hit_rate = (r > 0).mean()

    brute_results.append({
        "event": col,
        "n": n,
        "mean_ret": mean_ret,
        "hit_rate": hit_rate,
        "std": r.std(),
    })

    if mean_ret > MIN_MEAN_RET and hit_rate > MIN_HIT_RATE:
        selected_events.append(col)

brute_df = pd.DataFrame(brute_results).sort_values("mean_ret", ascending=False)

print("Selected events:", len(selected_events))
print("\nTop 30 event combos:")
print(brute_df.head(30).to_string(index=False))

if len(selected_events) == 0:
    raise ValueError("No event passed brute-force filtering. Lower thresholds.")

# =====================================================
# 5. Build DL dataframe
# =====================================================

dl_df = pd.DataFrame(index=price.index)

dl_df["ret_1d"] = price.pct_change()
dl_df["ret_3d"] = price.pct_change(3)
dl_df["ret_5d"] = price.pct_change(5)
dl_df["ma_gap_5"] = price / price.rolling(5).mean() - 1
dl_df["ma_gap_20"] = price / price.rolling(20).mean() - 1
dl_df["volatility_10"] = dl_df["ret_1d"].rolling(10).std()

# 加入外部市場特徵
for col in ["TSM", "^SOX", "^NDX", "^GSPC", "^VIX", "TWD=X", "^TNX"]:
    if col in price_df.columns:
        if col in ["^VIX", "^TNX"]:
            dl_df[f"{col}_diff"] = price_df[col].diff()
        else:
            dl_df[f"{col}_ret"] = price_df[col].pct_change()

# 加入 brute-force 選出的事件特徵
dl_df = dl_df.join(combo_df[selected_events], how="left")
dl_df[selected_events] = dl_df[selected_events].fillna(0)

# 建立三分類 label：跌 / 平 / 漲
future_ret = price.shift(-1) / price - 1
threshold = future_ret.rolling(20).std().bfill() * 0.5

dl_df["label"] = 1
dl_df.loc[future_ret > threshold, "label"] = 2
dl_df.loc[future_ret < -threshold, "label"] = 0

dl_df = dl_df.replace([np.inf, -np.inf], np.nan).dropna()

feature_cols = [c for c in dl_df.columns if c != "label"]

print("DL feature count:", len(feature_cols))
print("DL samples:", len(dl_df))
print(dl_df["label"].value_counts())

# =====================================================
# 6. Dataset
# =====================================================

class StockEventDataset(Dataset):
    def __init__(self, df, feature_cols, window):
        self.X = df[feature_cols].astype(np.float32).values
        self.y = df["label"].astype(np.int64).values
        self.window = window

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.window]
        y = self.y[idx + self.window]
        return torch.tensor(x), torch.tensor(y)

# time split
n = len(dl_df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_df = dl_df.iloc[:train_end]
val_df = dl_df.iloc[train_end - WINDOW:val_end]
test_df = dl_df.iloc[val_end - WINDOW:]

train_ds = StockEventDataset(train_df, feature_cols, WINDOW)
val_ds = StockEventDataset(val_df, feature_cols, WINDOW)
test_ds = StockEventDataset(test_df, feature_cols, WINDOW)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# =====================================================
# 7. LSTM model
# =====================================================

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LSTMClassifier(input_dim=len(feature_cols)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# =====================================================
# 8. Train / evaluate
# =====================================================

def evaluate(loader):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            pred = logits.argmax(dim=1)

            total_loss += loss.item() * len(y)
            correct += (pred == y).sum().item()
            total += len(y)

    return total_loss / total, correct / total

best_val_acc = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()

    total_loss = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        total += len(y)

    train_loss = total_loss / total
    val_loss, val_acc = evaluate(val_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f}"
    )

# =====================================================
# 9. Test
# =====================================================

model.load_state_dict(best_state)

test_loss, test_acc = evaluate(test_loader)

print("\n==============================")
print(f"Target: {TARGET}")
print(f"Selected brute-force events: {len(selected_events)}")
print(f"Best val acc: {best_val_acc:.4f}")
print(f"Test acc: {test_acc:.4f}")
print("==============================")

print("\nSelected events:")
for e in selected_events[:50]:
    print(e)