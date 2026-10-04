"""Neural network architectures trained in this study (pure PyTorch, no training logic).

This file is the single reference for the architectures described in the paper, so that text,
figures and code can be checked against each other in one place.

Input convention for every model:
  x         (batch, window, n_features)  the LAST feature is the (scaled) consumption; the
                                          other columns are exogenous inputs observed up to t-1
  x_future  (batch, horizon, n_future)   only for LightTCN-F: inputs known for the target hours
  output    (batch, horizon)             direct multi-step forecast of the scaled consumption

Proposed TCN family (all causal: an output at time t never sees inputs after t):
  LightTCN      single dilated causal TCN over all input channels, linear read-out of the last step
  ConcatTCN     LightTCN + a small MLP on the exogenous values of the last input step (late fusion)
  DualTCN       separate TCNs for consumption and exogenous channels, gated fusion
  DualTCN_Attn  DualTCN whose exogenous read-out is a softmax-weighted pooling over all input steps,
                queried by the last consumption state. The attention has NO learnable projections,
                so DualTCN and DualTCN_Attn have identical parameter counts and differ only in how
                the exogenous path is read out.
  LightTCN_F    LightTCN encoder + a per-step decoder that also receives inputs known for the target
                hours (calendar, temperature forecasts). Used only in the known-future experiments;
                its control run feeds zeros instead of the future inputs (same network, no information).
Reference recurrent models: LSTM, BiLSTM, CNN-LSTM.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Size presets. "large" doubles the channels and uses more dropout.
MODEL_SIZES = {
    "medium": {"base_channels": 64, "kernel_size": 3, "dropout": 0.15},
    "large": {"base_channels": 128, "kernel_size": 3, "dropout": 0.20},
}
RNN_CONFIG = {"hidden_size": 64, "num_layers": 2, "dropout": 0.15}
CNN_LSTM_CONFIG = {"cnn_filters": 64, "lstm_hidden": 64, "dropout": 0.15}
STEP_EMBEDDING_DIM = 16


# ------------------------------------------------------------------ TCN building blocks
class LightCausalBlock(nn.Module):
    """Causal dilated Conv1d -> LayerNorm -> GELU -> Dropout, with a residual connection.

    Only the left side is padded, so the output at step t depends on inputs <= t.
    LayerNorm (instead of BatchNorm) keeps behaviour identical for any batch size.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (batch, channels, time)
        res = self.residual(x)
        x = self.conv(F.pad(x, (self.pad, 0)))
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.drop(self.act(x)) + res


def n_blocks_for(window: int) -> int:
    """Blocks with dilations 1, 2, 4, ... so that the receptive field 2^(n+1)-1 covers the window.
    Note: depth therefore grows with the window; window comparisons include this capacity change."""
    return max(3, math.ceil(math.log2(max(window, 2))))


def tcn_stack(in_ch: int, channels: int, kernel_size: int, dropout: float, window: int) -> nn.Sequential:
    return nn.Sequential(*[LightCausalBlock(in_ch if i == 0 else channels, channels, kernel_size, 2 ** i, dropout)
                           for i in range(n_blocks_for(window))])


# ------------------------------------------------------------------ proposed TCN family
class LightTCN(nn.Module):
    def __init__(self, n_features, horizon, window, base_channels=64, kernel_size=3, dropout=0.15):
        super().__init__()
        self.tcn = tcn_stack(n_features, base_channels, kernel_size, dropout, window)
        self.fc = nn.Linear(base_channels, horizon)

    def forward(self, x):
        return self.fc(self.tcn(x.transpose(1, 2))[:, :, -1])


class ConcatTCN(nn.Module):
    """The TCN sees all channels (as LightTCN); an MLP adds the exogenous values of the last input
    step before the read-out. It tests whether a late, explicit exogenous branch helps."""

    def __init__(self, n_features, horizon, window, base_channels=64, kernel_size=3, dropout=0.15):
        super().__init__()
        n_exo, exo_dim = n_features - 1, base_channels // 2
        self.tcn = tcn_stack(n_features, base_channels, kernel_size, dropout, window)
        self.exo_mlp = nn.Sequential(nn.Linear(n_exo, exo_dim * 2), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(exo_dim * 2, exo_dim), nn.LayerNorm(exo_dim))
        self.norm = nn.LayerNorm(base_channels + exo_dim)
        self.fc = nn.Linear(base_channels + exo_dim, horizon)

    def forward(self, x):
        last = self.tcn(x.transpose(1, 2))[:, :, -1]
        exo = self.exo_mlp(x[:, -1, :-1])
        return self.fc(self.norm(torch.cat([last, exo], dim=-1)))


class DualTCN(nn.Module):
    """Consumption and exogenous channels get separate TCNs; a sigmoid gate mixes the two last states."""

    def __init__(self, n_features, horizon, window, base_channels=64, kernel_size=3, dropout=0.15):
        super().__init__()
        self.cons_tcn = tcn_stack(1, base_channels, kernel_size, dropout, window)
        self.exo_tcn = tcn_stack(n_features - 1, base_channels, kernel_size, dropout, window)
        self.gate = nn.Sequential(nn.Linear(2 * base_channels, base_channels), nn.Sigmoid())
        self.norm = nn.LayerNorm(base_channels)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels, horizon)

    def exo_readout(self, cons_seq, exo_seq):
        return exo_seq[:, -1, :]

    def forward(self, x):
        cons_seq = self.cons_tcn(x[:, :, -1:].transpose(1, 2)).transpose(1, 2)   # (batch, time, C)
        exo_seq = self.exo_tcn(x[:, :, :-1].transpose(1, 2)).transpose(1, 2)
        t_last = cons_seq[:, -1, :]
        e = self.exo_readout(cons_seq, exo_seq)
        g = self.gate(torch.cat([t_last, e], dim=-1))
        return self.fc(self.norm(self.drop(g * t_last + (1 - g) * e)))


class DualTCNAttn(DualTCN):
    """Same network as DualTCN; only the exogenous read-out differs: instead of the last exogenous
    state, a softmax-weighted average of all exogenous states, with weights from the dot product
    between the last consumption state and each exogenous state (scale 1/sqrt(C), no learned
    projections). Set `keep_attention = True` to store the weights for analysis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = self.fc.in_features ** -0.5
        self.keep_attention = False
        self.last_attention = None

    def exo_readout(self, cons_seq, exo_seq):
        q = cons_seq[:, -1:, :]                                                 # (batch, 1, C)
        w = torch.softmax(torch.bmm(q, exo_seq.transpose(1, 2)) * self.scale, dim=-1)   # (batch, 1, time)
        if self.keep_attention:
            self.last_attention = w.squeeze(1).detach()
        return torch.bmm(w, exo_seq).squeeze(1)


class LightTCNF(nn.Module):
    """LightTCN encoder over the past + a decoder for each target step that also sees the inputs known
    for that step. Output = linear read-out of the encoder (as LightTCN) + per-step correction:
        y_k = W_k h + b_k + g([h, x_future_k, e_k]),   k = 1..H
    where e_k is a learned step embedding. The control run (F0) passes zeros as x_future, so the
    comparison with F1 measures the information in the future inputs, not a change of architecture."""

    def __init__(self, n_features, horizon, window, n_future, base_channels=64, kernel_size=3, dropout=0.15):
        super().__init__()
        self.tcn = tcn_stack(n_features, base_channels, kernel_size, dropout, window)
        self.fc = nn.Linear(base_channels, horizon)
        self.step_embedding = nn.Embedding(horizon, STEP_EMBEDDING_DIM)
        self.decoder = nn.Sequential(nn.Linear(base_channels + n_future + STEP_EMBEDDING_DIM, base_channels),
                                     nn.GELU(), nn.Dropout(dropout), nn.Linear(base_channels, 1))

    def forward(self, x, x_future):
        h = self.tcn(x.transpose(1, 2))[:, :, -1]                                # (batch, C)
        b, horizon = h.shape[0], self.fc.out_features
        steps = self.step_embedding.weight.unsqueeze(0).expand(b, horizon, -1)
        z = torch.cat([h.unsqueeze(1).expand(b, horizon, -1), x_future, steps], dim=-1)
        return self.fc(h) + self.decoder(z).squeeze(-1)


# ------------------------------------------------------------------ recurrent references
class LSTMModel(nn.Module):
    def __init__(self, n_features, horizon, hidden_size=64, num_layers=2, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1, :])


class BiLSTMModel(nn.Module):
    """Read-out = final forward state (after the last input hour) + final backward state (after reading
    the window backwards, i.e. at the first input hour). The backward output at the last time step
    would have seen only one input hour."""

    def __init__(self, n_features, horizon, hidden_size=64, num_layers=2, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(2 * hidden_size, horizon)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(torch.cat([h_n[-2], h_n[-1]], dim=-1))


class CNNLSTMModel(nn.Module):
    """Two Conv1d layers for local patterns, then an LSTM. The convolutions are not causal, but they
    only mix hours inside the input window, which lies entirely before the forecast origin."""

    def __init__(self, n_features, horizon, cnn_filters=64, lstm_hidden=64, dropout=0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, cnn_filters, 3, padding=1), nn.BatchNorm1d(cnn_filters), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_filters, cnn_filters, 3, padding=1), nn.BatchNorm1d(cnn_filters), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, horizon)

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.fc(self.lstm(z)[0][:, -1, :])


# ------------------------------------------------------------------ factory
TCN_CLASSES = {"LightTCN": LightTCN, "ConcatTCN": ConcatTCN, "DualTCN": DualTCN, "DualTCN_Attn": DualTCNAttn}
NEEDS_EXOGENOUS = {"ConcatTCN", "DualTCN", "DualTCN_Attn"}


def create_model(name: str, n_features: int, horizon: int, window: int, n_future: int = 0) -> nn.Module:
    """name examples: 'LightTCN_medium', 'DualTCN_Attn_large', 'LightTCN_F_medium', 'LSTM', 'CNN_LSTM'."""
    if name.startswith("LightTCN_F_"):
        return LightTCNF(n_features, horizon, window, n_future, **MODEL_SIZES[name.rsplit("_", 1)[1]])
    base, _, size = name.rpartition("_")
    if base in TCN_CLASSES:
        if base in NEEDS_EXOGENOUS and n_features < 2:
            raise ValueError(f"{base} needs exogenous input channels")
        return TCN_CLASSES[base](n_features, horizon, window, **MODEL_SIZES[size])
    if name == "LSTM":
        return LSTMModel(n_features, horizon, **RNN_CONFIG)
    if name == "BiLSTM":
        return BiLSTMModel(n_features, horizon, **RNN_CONFIG)
    if name == "CNN_LSTM":
        return CNNLSTMModel(n_features, horizon, **CNN_LSTM_CONFIG)
    raise ValueError(f"unknown model {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
