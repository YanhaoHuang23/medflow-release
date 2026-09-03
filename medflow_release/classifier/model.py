# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from __future__ import annotations
import torch
from torch import nn


class RNNClassifier(nn.Module):
    """Bidirectional LSTM/GRU classifier for fixed‑length sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type.lower()]
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F)
        rnn_out, _ = self.rnn(x)  # (B, T, 2*H)
        last_hidden = rnn_out[:, -1, :]  # final time‑step representation
        logits = self.fc(last_hidden)  # (B, C) or (B, 1)
        return logits.squeeze(-1)  # binary → (B,) ; multi‑class stays (B, C)
