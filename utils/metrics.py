from __future__ import annotations

import torch


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batch_stats(logits: torch.Tensor, labels: torch.Tensor):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().float()
    total = torch.tensor(float(labels.size(0)), device=labels.device)
    return correct, total
