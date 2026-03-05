from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool


class GATClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        pooling: str = "mean",
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.dropout = dropout
        self.pooling = pooling

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                concat=True,
                dropout=dropout,
            )
        )

        in_dim = hidden_channels * heads
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    in_dim,
                    hidden_channels,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            )
            in_dim = hidden_channels * heads

        self.proj = GATConv(
            in_dim,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.proj(x, edge_index)
        x = F.elu(x)

        if self.pooling == "add":
            g = global_add_pool(x, batch)
        else:
            g = global_mean_pool(x, batch)

        return self.head(g)
