from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val_acc: float,
    config: dict,
    accelerator=None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
    state = {
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "model_state": unwrapped.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
    }
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, accelerator=None):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
    unwrapped.load_state_dict(state["model_state"])

    if optimizer is not None and state.get("optimizer_state") is not None:
        optimizer.load_state_dict(state["optimizer_state"])

    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])

    return state


def update_latest_symlink(output_dir: str, target_path: str) -> None:
    latest_path = Path(output_dir) / "latest"
    target = Path(target_path)
    if not target.exists():
        return
    if latest_path.is_symlink() or latest_path.is_file():
        latest_path.unlink()
    elif latest_path.is_dir():
        return
    latest_path.symlink_to(target.resolve())
