# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from __future__ import annotations

import argparse
from pathlib import Path
try:
    from .model import RNNClassifier
except ImportError:
    from model import RNNClassifier
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import os
import pandas as pd
from typing import Optional, Tuple
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


class TimeSeriesDataset(Dataset):

    def __init__(
        self,
        data,
        labels,
        normalize=True,
        stats=None,
        eps=1e-8,
    ):

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        assert data.ndim == 3, "data must be (N, seq_len, n_features)"
        assert len(data) == len(labels)
        self.data = data.float()
        self.labels = labels.long()
        self.normalize = normalize
        self.eps = eps

        if self.normalize:
            if stats is None:
                # compute mean/std over all time‑steps *per feature*
                mean = self.data.mean(dim=(0, 1), keepdim=True)  # (1,1,F)
                std = self.data.std(dim=(0, 1), keepdim=True)
            else:
                mean, std = stats
                if isinstance(mean, np.ndarray):
                    mean = torch.from_numpy(mean)
                if isinstance(std, np.ndarray):
                    std = torch.from_numpy(std)
                mean, std = mean.float(), std.float()
            self.register_buffer("_mean",
                                 mean)  # cached on device when .to(...)
            self.register_buffer("_std", std.clamp_min(self.eps))

    # tiny helper so buffers exist even on CPU tensors
    def register_buffer(self, name: str, tensor: torch.Tensor):
        object.__setattr__(self, name, tensor)

    # expose stats to reuse on other splits
    @property
    def stats(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.normalize:
            return None
        return self._mean, self._std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.normalize:
            x = (x - self._mean) / self._std
        x = x.squeeze(0)
        return x, self.labels[idx]


# -----------------------------------------------------------------------------
# Train / Eval helpers
# -----------------------------------------------------------------------------


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.dim() == 1:  # binary with BCEWithLogits
        preds = (torch.sigmoid(logits) > 0.5).long()
    else:  # multi‑class with CE
        preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def _ensure_ntf(data: np.ndarray, input_dim: Optional[int] = None) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"data must be 3D, got {data.shape}")
    if input_dim is not None:
        input_dim = int(input_dim)
        if data.shape[1] == input_dim:
            return data.transpose(0, 2, 1)
        if data.shape[2] == input_dim:
            return data
        raise ValueError(
            f"expected {input_dim}-feature data in (N,{input_dim},T) or (N,T,{input_dim}), got {data.shape}"
        )
    if data.shape[1] <= data.shape[2]:
        return data.transpose(0, 2, 1)
    if data.shape[2] < data.shape[1]:
        return data
    raise ValueError(f"Cannot infer tuple orientation; pass --input_dim for shape {data.shape}")


def _load_tuple(path: str, input_dim: Optional[int] = None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tuple data not found: {path}")
    x, y = pd.read_pickle(path)
    x = _ensure_ntf(x, input_dim=input_dim)
    y = np.asarray(y, dtype=np.int64)
    if len(x) != len(y):
        raise ValueError(f"Data/label length mismatch in {path}: {len(x)} vs {len(y)}")
    return x, y


def _metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int):
    logits_np = logits.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    if num_classes == 1:
        probs = 1.0 / (1.0 + np.exp(-logits_np))
        preds = (probs >= 0.5).astype(np.int64)
        out = {
            "accuracy": float(accuracy_score(labels_np, preds)),
            "f1": float(f1_score(labels_np, preds, zero_division=0)),
        }
        if len(np.unique(labels_np)) == 2:
            out["auroc"] = float(roc_auc_score(labels_np, probs))
            out["auprc"] = float(average_precision_score(labels_np, probs))
        else:
            out["auroc"] = float("nan")
            out["auprc"] = float("nan")
        return out

    probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    preds = probs.argmax(axis=1)
    out = {
        "accuracy": float(accuracy_score(labels_np, preds)),
        "f1": float(f1_score(labels_np, preds, average="macro", zero_division=0)),
    }
    one_hot = np.eye(num_classes, dtype=np.int64)[labels_np]
    try:
        out["auroc"] = float(
            roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(one_hot, probs, average="macro"))
    except ValueError:
        out["auprc"] = float("nan")
    return out


def _run_epoch(model, loader, criterion, optimizer, device, train=True, num_classes=1):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_acc = 0.0
    all_logits = []
    all_labels = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y.float() if logits.dim() == 1 else y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc += _accuracy(logits, y) * x.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
    n = len(loader.dataset)
    metrics = _metrics_from_logits(torch.cat(all_logits), torch.cat(all_labels), num_classes)
    metrics["loss"] = total_loss / n
    metrics["accuracy_epoch"] = total_acc / n
    return metrics


# -----------------------------------------------------------------------------
# Main – quick demo on synthetic data
# -----------------------------------------------------------------------------


def main(args):
    X_train, y_train = _load_tuple(args.train_data, input_dim=args.input_dim)
    if args.input_dim is None:
        args.input_dim = int(X_train.shape[-1])
    X_val, y_val = _load_tuple(args.val_data, input_dim=args.input_dim)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_set = TimeSeriesDataset(X_train, y_train)
    val_set = TimeSeriesDataset(X_val, y_val, stats=train_set.stats)

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNClassifier(
        input_dim=int(args.input_dim),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)

    criterion = (nn.BCEWithLogitsLoss()
                 if args.num_classes == 1 else nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1.0
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(1, args.epochs + 1)):
        tr = _run_epoch(model,
                        train_loader,
                        criterion,
                        optimizer,
                        device,
                        train=True,
                        num_classes=args.num_classes)
        va = _run_epoch(model,
                        val_loader,
                        criterion,
                        optimizer,
                        device,
                        train=False,
                        num_classes=args.num_classes)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} auprc {tr['auprc']:.4f} auroc {tr['auroc']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f} auprc {va['auprc']:.4f} auroc {va['auroc']:.4f} f1 {va['f1']:.4f}"
        )
        score = va["auprc"] if not np.isnan(va["auprc"]) else va["accuracy"]
        if score > best_val:
            best_val = score
            print(f"New best val score: {best_val:.4f} -> saving model")
            torch.save({"model_state": model.state_dict(), "stats": train_set.stats, "metrics": va},
                       Path(args.ckpt_dir) / "best_model.pt")
    print(f"Train Finished. Best val score: {best_val:.4f}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Bidirectional LSTM/GRU time‑series classifier")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--rnn_type", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--num_classes",
                   type=int,
                   default=1,
                   help="1 for binary, >1 for multi‑class")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--input_dim", type=int, default=None, help="optional tuple feature dimension; inferred when omitted")
    p.add_argument("--train_data", type=str, default="data/train_data.npy")
    p.add_argument("--val_data", type=str, default="data/val_data.npy")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    main(args)
