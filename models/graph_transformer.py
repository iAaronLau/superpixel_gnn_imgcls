from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_add_pool, global_max_pool, global_mean_pool


class GraphTransformerClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        edge_dim: int | None = None,
        num_layers: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
        pooling: str = "meanmax",
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.dropout = dropout
        self.pooling = pooling

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        layer_dims = [in_channels] + [hidden_channels] * (num_layers - 1)
        for in_dim in layer_dims:
            self.convs.append(
                TransformerConv(
                    in_dim,
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))
            self.residuals.append(nn.Linear(in_dim, hidden_channels) if in_dim != hidden_channels else nn.Identity())

        pooled_dim = hidden_channels * 2 if pooling == "meanmax" else hidden_channels
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "add":
            return global_add_pool(x, batch)
        if self.pooling == "meanmax":
            return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        return global_mean_pool(x, batch)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)

        hidden_states = []
        for conv, norm, residual in zip(self.convs, self.norms, self.residuals):
            h = conv(x, edge_index, edge_attr=edge_attr)
            h = norm(h)
            h = F.relu(h + residual(x))
            h = F.dropout(h, p=self.dropout, training=self.training)
            hidden_states.append(h)
            x = h

        x = torch.stack(hidden_states, dim=0).mean(dim=0)
        g = self._pool(x, batch)
        return self.head(g)
