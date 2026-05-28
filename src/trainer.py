import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from .models import DualBranchNet
from .losses import LogitAdjustedLoss
from .metrics import compute_f1_scores

def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=20, grad_clip=1.0, verbose=True, restore_best=True):
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    model.to(device)
    best_score = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_count = 0.0, 0
        for market_x, text_x, y in train_loader:
            market_x, text_x, y = market_x.to(device), text_x.to(device), y.to(device)
            loss = loss_fn(model(market_x, text_x), y)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_count += y.size(0)

        model.eval()
        val_loss, val_count, val_targets, val_preds = 0.0, 0, [], []
        with torch.no_grad():
            for market_x, text_x, y in val_loader:
                market_x, text_x, y = market_x.to(device), text_x.to(device), y.to(device)
                logits = model(market_x, text_x)
                val_loss += loss_fn(logits, y).item() * y.size(0)
                val_count += y.size(0)
                val_targets.append(y.cpu().numpy())
                val_preds.append(logits.argmax(dim=1).cpu().numpy())

        y_true = np.concatenate(val_targets) if val_targets else np.array([], dtype=int)
        y_pred = np.concatenate(val_preds) if val_preds else np.array([], dtype=int)
        history["train_loss"].append(total_loss / max(1, total_count))
        history["val_loss"].append(val_loss / max(1, val_count))
        history["val_macro_f1"].append(compute_f1_scores(y_true, y_pred) if y_true.size else 0.0)
        if history["val_macro_f1"][-1] > best_score:
            best_score = history["val_macro_f1"][-1]
            best_state = copy.deepcopy(model.state_dict())

        if verbose: print(f"Epoch {epoch:02d}/{epochs} | train_loss={history['train_loss'][-1]:.4f} | val_loss={history['val_loss'][-1]:.4f} | val_f1={history['val_macro_f1'][-1]:.4f}")

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return history


def _dates_from_dataset(dataset):
    if isinstance(dataset, Subset):
        base_dates = _dates_from_dataset(dataset.dataset)
        return base_dates[np.asarray(dataset.indices)]
    if not hasattr(dataset, "sample_index"):
        raise ValueError("dataset must expose sample_index to align predictions with trading dates.")
    return pd.DatetimeIndex(dataset.sample_index)


def evaluate_predictions(model, data_loader, device):
    model.eval()
    preds, probas, targets = [], [], []
    with torch.no_grad():
        for market_x, text_x, y in data_loader:
            logits = model(market_x.to(device), text_x.to(device))
            probas.append(torch.softmax(logits, dim=1).cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(y.numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    y_proba = np.concatenate(probas)
    dates = _dates_from_dataset(data_loader.dataset)
    if len(dates) != len(y_true):
        raise ValueError(
            f"prediction date count mismatch: dates={len(dates)} vs y_true={len(y_true)}"
        )
    return y_true, y_pred, y_proba, dates

def train_and_eval_ablation(train_subset, val_subset, class_props, device, sample_market, sample_text, zero_text=False, eval_subset=None, epochs=20):
    text_dim = sample_text.shape[-1]
    model = DualBranchNet(text_dim, sample_market.shape[-1])
    loader_train = DataLoader(train_subset, batch_size=64, shuffle=True, drop_last=True)
    loader_val = DataLoader(val_subset, batch_size=64, shuffle=False)
    loader_eval = DataLoader(eval_subset if eval_subset is not None else val_subset, batch_size=64, shuffle=False)

    if zero_text:
        original_text_train = train_subset.dataset.text_features.copy()
        train_subset.dataset.text_features = np.zeros_like(original_text_train)

    loss_fn = LogitAdjustedLoss(class_priors=class_props, tau=0.75).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    _ = train_model(model, loader_train, loader_val, optimizer, loss_fn, device, epochs=epochs, verbose=False)
    y_true, y_pred, y_proba, dates = evaluate_predictions(model, loader_eval, device)
    
    if zero_text: train_subset.dataset.text_features = original_text_train
    return y_true, y_pred, y_proba, dates
