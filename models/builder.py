from __future__ import annotations

from .gnn_gat import GATClassifier
from .gnn_gcn import GCNClassifier
from .graph_transformer import GraphTransformerClassifier
from .resnet import ResNetBaseline


def build_model(args, num_classes: int, node_feature_dim: int | None = None):
    model_name = args.model.lower()

    if model_name == "resnet":
        cifar_stem = args.dataset.lower() == "cifar10" or args.image_size <= 96
        return ResNetBaseline(
            num_classes=num_classes,
            cifar_stem=cifar_stem,
            dropout=args.dropout,
        )

    if node_feature_dim is None:
        raise ValueError("node_feature_dim is required for graph models")

    if model_name == "gcn":
        return GCNClassifier(
            in_channels=node_feature_dim,
            hidden_channels=args.hidden_dim,
            num_classes=num_classes,
            num_layers=args.gnn_layers,
            dropout=args.dropout,
            pooling=args.pooling,
        )
    if model_name == "gat":
        return GATClassifier(
            in_channels=node_feature_dim,
            hidden_channels=args.hidden_dim,
            num_classes=num_classes,
            num_layers=args.gnn_layers,
            heads=args.gat_heads,
            dropout=args.dropout,
            pooling=args.pooling,
        )
    if model_name == "graph_transformer":
        return GraphTransformerClassifier(
            in_channels=node_feature_dim,
            hidden_channels=args.hidden_dim,
            num_classes=num_classes,
            num_layers=args.gnn_layers,
            heads=args.gat_heads,
            dropout=args.dropout,
            pooling=args.pooling,
        )

    raise ValueError(f"Unsupported model: {args.model}")
