"""Sequence taggers: a bidirectional LSTM and a Transformer encoder.

Both take the integer feature matrix from :mod:`piano_hand_assignment.features`
and emit two logits per note (right hand / left hand).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .features import FEAT_DIMS

EMB_DIMS = [32, 16, 16, 16, 8, 8, 16, 8]
IN_DIM = sum(EMB_DIMS)


class InputEncoder(nn.Module):
    """Embedding tables for every discrete feature, followed by a projection."""

    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, e) for n, e in zip(FEAT_DIMS, EMB_DIMS)])
        self.proj = nn.Sequential(nn.Linear(IN_DIM, d), nn.LayerNorm(d), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [emb(x[..., i]) for i, emb in enumerate(self.embs)]
        return self.proj(torch.cat(parts, dim=-1))


class BiLSTMTagger(nn.Module):
    def __init__(self, d: int = 192, layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.enc = InputEncoder(d)
        self.rnn = nn.LSTM(d, d, num_layers=layers, batch_first=True,
                           bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(d, 2))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        lengths = mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=h.size(1))
        return self.head(out)


class TransformerTagger(nn.Module):
    def __init__(self, d: int = 192, layers: int = 4, heads: int = 4,
                 dropout: float = 0.15, max_len: int = 1024):
        super().__init__()
        self.enc = InputEncoder(d)
        self.pos = nn.Embedding(max_len, d)
        nn.init.normal_(self.pos.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=3 * d,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.trf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(d, 2))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        h = h + self.pos.weight[:h.size(1)].unsqueeze(0)
        out = self.trf(h, src_key_padding_mask=~mask)
        return self.head(out)


def build(arch: str, **kwargs) -> nn.Module:
    if arch == "lstm":
        return BiLSTMTagger(**kwargs)
    if arch == "transformer":
        return TransformerTagger(**kwargs)
    raise ValueError(f"unknown architecture: {arch}")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
