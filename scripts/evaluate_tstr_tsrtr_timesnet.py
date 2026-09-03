#!/usr/bin/env python
import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


def ensure_ntf(x, input_dim=None):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D time-series tensor, got {x.shape}")
    if input_dim is not None:
        input_dim = int(input_dim)
        if x.shape[1] == input_dim:
            return x.transpose(0, 2, 1)
        if x.shape[2] == input_dim:
            return x
        raise ValueError(
            f"Expected {input_dim}-feature tensor in (N,{input_dim},T) or (N,T,{input_dim}), got {x.shape}"
        )
    if x.shape[1] <= x.shape[2]:
        return x.transpose(0, 2, 1)
    if x.shape[2] < x.shape[1]:
        return x
    raise ValueError(f"Cannot infer tuple orientation; pass --input-dim for shape {x.shape}")


def load_tuple(path, input_dim=None):
    x, y = pd.read_pickle(path)
    x = ensure_ntf(x, input_dim=input_dim)
    y = np.asarray(y, dtype=np.int64)
    if len(x) != len(y):
        raise ValueError(f"{path}: len(X)={len(x)} != len(y)={len(y)}")
    return x, y


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TupleDataset(Dataset):
    def __init__(self, x, y, stats=None):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        if stats is None:
            mean = self.x.mean(axis=(0, 1), keepdims=True)
            std = self.x.std(axis=(0, 1), keepdims=True)
            stats = {"mean": mean.astype(np.float32), "std": np.maximum(std, 1e-6).astype(np.float32)}
        self.stats = stats
        self.x = (self.x - self.stats["mean"]) / self.stats["std"]
        self.x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6):
        super().__init__()
        self.kernels = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
                for i in range(num_kernels)
            ]
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return torch.stack([kernel(x) for kernel in self.kernels], dim=-1).mean(-1)


def fft_for_period(x, top_k):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = xf.abs().mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, min(top_k, len(frequency_list)))
    period = torch.clamp(x.shape[1] // torch.clamp(top_list, min=1), min=1)
    return period.detach().cpu().tolist(), xf.abs().mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, top_k, num_kernels, dropout):
        super().__init__()
        self.seq_len = int(seq_len)
        self.top_k = int(top_k)
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            nn.Dropout(dropout),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        batch, length_in, dim = x.size()
        period_list, period_weight = fft_for_period(x, self.top_k)
        outputs = []
        for period in period_list:
            period = max(int(period), 1)
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros(batch, length - self.seq_len, dim, device=x.device, dtype=x.dtype)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = out.reshape(batch, length // period, period, dim).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(batch, -1, dim)
            outputs.append(out[:, :length_in, :])
        stacked = torch.stack(outputs, dim=-1)
        weights = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1)
        weights = weights.repeat(1, length_in, dim, 1)
        return torch.sum(stacked * weights, dim=-1) + x


class TimesNetClassifier(nn.Module):
    """TimesNet-style classifier following the THUML TSLib classification path."""

    def __init__(self, seq_len, input_dim, num_classes, d_model, d_ff, e_layers, top_k, num_kernels, dropout):
        super().__init__()
        self.seq_len = int(seq_len)
        self.embedding = DataEmbedding(input_dim, d_model, dropout)
        self.blocks = nn.ModuleList(
            [TimesBlock(seq_len, d_model, d_ff, top_k, num_kernels, dropout) for _ in range(e_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model * seq_len, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        for block in self.blocks:
            out = self.layer_norm(block(out))
        out = F.gelu(out)
        out = self.dropout(out)
        return self.projection(out.reshape(out.shape[0], -1))


def metrics_from_logits(logits, labels, num_classes):
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)
    per_class_f1 = f1_score(
        labels,
        preds,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="binary" if num_classes == 2 else "macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "minority_f1": float(np.min(per_class_f1)) if len(per_class_f1) else float("nan"),
    }
    if len(np.unique(labels)) == 2 and probs.shape[1] == 2:
        out["auroc"] = float(roc_auc_score(labels, probs[:, 1]))
        out["auprc"] = float(average_precision_score(labels, probs[:, 1]))
    else:
        one_hot = np.eye(num_classes, dtype=np.int64)[labels]
        try:
            out["auroc"] = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            )
        except ValueError:
            out["auroc"] = float("nan")
        try:
            out["auprc"] = float(average_precision_score(one_hot, probs, average="macro"))
        except ValueError:
            out["auprc"] = float("nan")
    return out


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss = 0.0
    logits_all = []
    labels_all = []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                if optimizer is not None:
                    optimizer.step()
            total_loss += loss.item() * x.shape[0]
            logits_all.append(logits.detach().cpu().numpy())
            labels_all.append(y.detach().cpu().numpy())
    metrics = metrics_from_logits(
        np.concatenate(logits_all),
        np.concatenate(labels_all),
        num_classes=int(criterion.num_classes) if hasattr(criterion, "num_classes") else 2,
    )
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def make_criterion(y_train, mode, device, num_classes):
    if mode == "none":
        criterion = nn.CrossEntropyLoss()
        criterion.num_classes = int(num_classes)
        return criterion
    if mode == "balanced":
        counts = np.bincount(y_train.astype(np.int64), minlength=int(num_classes)).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        criterion.num_classes = int(num_classes)
        return criterion
    raise ValueError(f"Unsupported class_weight={mode}")


def train_eval(x_train, y_train, x_val, y_val, x_test, y_test, args, seed):
    seed_all(seed)
    train_set = TupleDataset(x_train, y_train)
    val_set = TupleDataset(x_val, y_val, stats=train_set.stats)
    test_set = TupleDataset(x_test, y_test, stats=train_set.stats)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = TimesNetClassifier(
        seq_len=x_train.shape[1],
        input_dim=x_train.shape[-1],
        num_classes=int(args.num_classes),
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        top_k=args.top_k,
        num_kernels=args.num_kernels,
        dropout=args.dropout,
    ).to(device)
    criterion = make_criterion(y_train, args.class_weight, device, args.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = -1.0
    patience_left = args.patience
    for epoch in range(args.epochs):
        run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, None, device, train=False)
        score = val_metrics["auprc"]
        if score > best_val:
            best_val = score
            patience_left = args.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if args.patience > 0 and patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return run_epoch(model, test_loader, criterion, None, device, train=False)


def sample_synthetic(x_syn, y_syn, n_samples, seed):
    rng = np.random.default_rng(seed)
    parts = []
    labels = []
    classes = np.unique(y_syn)
    counts = {int(cls): int(round(n_samples * np.mean(y_syn == cls))) for cls in classes}
    short = n_samples - sum(counts.values())
    if short and len(classes):
        counts[int(classes[0])] += short
    for cls, count in counts.items():
        idx = np.where(y_syn == cls)[0]
        if len(idx) == 0 or count <= 0:
            continue
        chosen = rng.choice(idx, size=count, replace=count > len(idx))
        parts.append(x_syn[chosen])
        labels.append(y_syn[chosen])
    x = np.concatenate(parts, axis=0)
    y = np.concatenate(labels, axis=0)
    order = rng.permutation(len(y))
    return x[order], y[order]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "protocol",
        "alpha",
        "seed",
        "classifier",
        "auprc",
        "auroc",
        "f1",
        "macro_f1",
        "weighted_f1",
        "minority_f1",
        "accuracy",
        "loss",
    ]
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    parser = argparse.ArgumentParser(description="TimesNet TRTR/TSTR/TSRTR evaluator for MIMIC tuple data.")
    parser.add_argument("--data-dir", default="data/processed/mimic_mortality_reasonable_7of7")
    parser.add_argument("--input-dim", type=int, default=None, help="optional tuple feature dimension; inferred when omitted")
    parser.add_argument("--num-classes", type=int, default=None, help="number of classes; inferred from tuple labels when omitted")
    parser.add_argument("--synthetic", action="append", default=[], help="method_name=path/to/synthetic.pkl")
    parser.add_argument("--out", default="results/mimic_mortality_reasonable_timesnet.csv")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--num-kernels", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--skip-trtr", action="store_true", help="skip real-only TRTR and only evaluate provided synthetic sets")
    parser.add_argument("--skip-tsrtr", action="store_true", help="skip TSRTR augmentation runs")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    x_train, y_train = load_tuple(data_dir / "train_tuple.pkl", input_dim=args.input_dim)
    if args.input_dim is None:
        args.input_dim = int(x_train.shape[-1])
    x_val, y_val = load_tuple(data_dir / "val_tuple.pkl", input_dim=args.input_dim)
    x_test, y_test = load_tuple(data_dir / "test_tuple.pkl", input_dim=args.input_dim)
    if args.num_classes is None:
        args.num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)

    rows = []
    if not args.skip_trtr:
        for seed in args.seeds:
            metrics = train_eval(x_train, y_train, x_val, y_val, x_test, y_test, args, seed)
            rows.append({"method": "real_only", "protocol": "TRTR", "alpha": "", "seed": seed, "classifier": "TimesNet", **metrics})

    for item in args.synthetic:
        if "=" not in item:
            raise ValueError("--synthetic must be method_name=path")
        method, path = item.split("=", 1)
        x_syn, y_syn = load_tuple(path, input_dim=args.input_dim)
        for seed in args.seeds:
            metrics = train_eval(x_syn, y_syn, x_val, y_val, x_test, y_test, args, seed)
            rows.append({"method": method, "protocol": "TSTR", "alpha": "", "seed": seed, "classifier": "TimesNet", **metrics})
            if args.skip_tsrtr:
                continue
            for alpha in args.alphas:
                n_syn = int(round(alpha * len(x_train)))
                x_aug, y_aug = sample_synthetic(x_syn, y_syn, n_syn, seed)
                x_mix = np.concatenate([x_train, x_aug], axis=0)
                y_mix = np.concatenate([y_train, y_aug], axis=0)
                metrics = train_eval(x_mix, y_mix, x_val, y_val, x_test, y_test, args, seed)
                rows.append({"method": method, "protocol": "TSRTR", "alpha": alpha, "seed": seed, "classifier": "TimesNet", **metrics})

    write_rows(args.out, rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
