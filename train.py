from __future__ import annotations

import argparse
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

from data_utils.datasets import DatasetBundle, load_dataset_bundle
from data_utils.graph_dataset import GraphBuildConfig, HFGraphDataset
from data_utils.image_dataset import HFImageDataset, build_image_transform
from models import build_model
from utils.checkpoint import load_checkpoint, save_checkpoint, update_latest_symlink
from utils.classification_metrics import (
    compute_classification_metrics,
    log_wandb_classification_artifacts,
    logits_to_probs,
    prefix_metrics,
)
from utils.io import append_result_row, dump_config, ensure_dir
from utils.metrics import batch_stats, count_parameters
from utils.seed import set_global_seed
from utils.trainer_backend import run_transformers_training


GRAPH_MODELS = {"gcn", "gat", "graph_transformer"}


def is_rank0_process() -> bool:
    rank = os.environ.get("RANK")
    if rank is None:
        return True
    return int(rank) == 0


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Superpixel GNN Image Classification")

    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagewoof"])
    parser.add_argument(
        "--model",
        type=str,
        default="resnet",
        choices=["resnet", "gcn", "gat", "graph_transformer"],
    )
    parser.add_argument("--train_backend", type=str, default="transformers", choices=["accelerate", "transformers"])

    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--n_segments", type=int, default=100)
    parser.add_argument("--use_xy", type=int, default=1)
    parser.add_argument("--slic_compactness", type=float, default=10.0)
    parser.add_argument("--slic_sigma", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "step"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--resnet_name", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "add"])
    parser.add_argument("--grad_clip", type=float, default=0.0)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--persistent_workers", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", type=int, default=1)
    parser.add_argument("--cudnn_benchmark", type=int, default=1)
    parser.add_argument("--channels_last", type=int, default=1)
    parser.add_argument("--torch_compile", type=int, default=0)
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddp_find_unused_parameters", type=int, default=0)

    parser.add_argument("--use_cache", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default="graph_cache")

    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    parser.add_argument("--save_strategy", type=str, default="auto", choices=["auto", "epoch", "steps"])
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--save_safetensors", type=int, default=1)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="superpixel-gnn-imgcls")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--push_to_hub", type=int, default=0)
    parser.add_argument("--hub_model_id", type=str, default="")
    parser.add_argument("--hub_private_repo", type=int, default=0)
    parser.add_argument(
        "--hub_strategy",
        type=str,
        default="every_save",
        choices=["end", "every_save", "checkpoint", "all_checkpoints"],
    )
    parser.add_argument("--hub_token", type=str, default="")

    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_val_samples", type=int, default=-1)
    parser.add_argument("--max_test_samples", type=int, default=-1)

    return parser.parse_args()


def _default_image_size(args) -> int:
    if args.image_size is not None:
        return args.image_size
    return 64 if args.dataset == "cifar10" else 224


def configure_torch_runtime(args):
    if not torch.cuda.is_available():
        return

    allow_tf32 = bool(args.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    except Exception:  # noqa: PERF203
        pass

    try:
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    except Exception:  # noqa: PERF203
        pass

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def _subset_split(split: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples is None or max_samples < 0 or max_samples >= len(split):
        return split
    split = split.shuffle(seed=seed)
    return split.select(range(max_samples))


def maybe_subset_bundle(bundle: DatasetBundle, args) -> DatasetBundle:
    train_split = _subset_split(bundle.splits["train"], args.max_train_samples, args.seed)
    val_split = _subset_split(bundle.splits["val"], args.max_val_samples, args.seed + 1)
    test_split = _subset_split(bundle.splits["test"], args.max_test_samples, args.seed + 2)
    bundle.splits = {"train": train_split, "val": val_split, "test": test_split}
    return bundle


def _build_loader_common_kwargs(args, pin_memory: bool):
    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "worker_init_fn": _seed_worker,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def build_dataloaders(args, bundle: DatasetBundle):
    is_graph = args.model in GRAPH_MODELS
    pin_memory = torch.cuda.is_available()
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    eval_generator = torch.Generator()
    eval_generator.manual_seed(args.seed + 1)

    common_kwargs = _build_loader_common_kwargs(args, pin_memory=pin_memory)

    if not is_graph:
        train_t = build_image_transform(args.dataset, args.image_size, train=True)
        eval_t = build_image_transform(args.dataset, args.image_size, train=False)

        train_ds = HFImageDataset(bundle.splits["train"], bundle.image_key, bundle.label_key, train_t)
        val_ds = HFImageDataset(bundle.splits["val"], bundle.image_key, bundle.label_key, eval_t)
        test_ds = HFImageDataset(bundle.splits["test"], bundle.image_key, bundle.label_key, eval_t)

        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            generator=train_generator,
            **common_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            generator=eval_generator,
            **common_kwargs,
        )
        test_loader = DataLoader(
            test_ds,
            shuffle=False,
            generator=eval_generator,
            **common_kwargs,
        )
        return train_loader, val_loader, test_loader, None

    graph_cfg = GraphBuildConfig(
        image_size=args.image_size,
        n_segments=args.n_segments,
        use_xy=bool(args.use_xy),
        compactness=args.slic_compactness,
        sigma=args.slic_sigma,
    )

    train_ds = HFGraphDataset(
        bundle.splits["train"],
        image_key=bundle.image_key,
        label_key=bundle.label_key,
        split_name="train",
        dataset_name=args.dataset,
        graph_cfg=graph_cfg,
        use_cache=bool(args.use_cache),
        cache_dir=args.cache_dir,
    )
    val_ds = HFGraphDataset(
        bundle.splits["val"],
        image_key=bundle.image_key,
        label_key=bundle.label_key,
        split_name="val",
        dataset_name=args.dataset,
        graph_cfg=graph_cfg,
        use_cache=bool(args.use_cache),
        cache_dir=args.cache_dir,
    )
    test_ds = HFGraphDataset(
        bundle.splits["test"],
        image_key=bundle.image_key,
        label_key=bundle.label_key,
        split_name="test",
        dataset_name=args.dataset,
        graph_cfg=graph_cfg,
        use_cache=bool(args.use_cache),
        cache_dir=args.cache_dir,
    )

    train_loader = PyGDataLoader(
        train_ds,
        shuffle=True,
        generator=train_generator,
        **common_kwargs,
    )
    val_loader = PyGDataLoader(
        val_ds,
        shuffle=False,
        generator=eval_generator,
        **common_kwargs,
    )
    test_loader = PyGDataLoader(
        test_ds,
        shuffle=False,
        generator=eval_generator,
        **common_kwargs,
    )

    node_feature_dim = 5 if args.use_xy else 3
    return train_loader, val_loader, test_loader, node_feature_dim


def create_scheduler(args, optimizer):
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    if args.scheduler == "step":
        step_size = max(1, args.epochs // 3)
        return StepLR(optimizer, step_size=step_size, gamma=0.1)
    return None


def step_epoch(
    model,
    loader,
    criterion,
    accelerator: Accelerator,
    optimizer=None,
    is_graph=False,
    grad_clip=0.0,
    use_channels_last=False,
):
    is_train = optimizer is not None
    if is_train:
        model.train()
        context = nullcontext()
    else:
        model.eval()
        context = torch.no_grad()

    loss_total = 0.0
    correct_total = 0.0
    sample_total = 0.0

    with context:
        for batch in loader:
            if is_train:
                with accelerator.accumulate(model):
                    if is_graph:
                        labels = batch.y.view(-1)
                        logits = model(batch)
                    else:
                        images, labels = batch
                        if use_channels_last and images.ndim == 4:
                            images = images.contiguous(memory_format=torch.channels_last)
                        logits = model(images)

                    loss = criterion(logits, labels)
                    accelerator.backward(loss)

                    if grad_clip > 0 and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                if is_graph:
                    labels = batch.y.view(-1)
                    logits = model(batch)
                else:
                    images, labels = batch
                    if use_channels_last and images.ndim == 4:
                        images = images.contiguous(memory_format=torch.channels_last)
                    logits = model(images)
                loss = criterion(logits, labels)

            correct, total = batch_stats(logits, labels)
            loss_sum = loss.detach().float() * float(labels.size(0))

            gathered_loss = accelerator.gather_for_metrics(loss_sum.unsqueeze(0)).sum().item()
            gathered_correct = accelerator.gather_for_metrics(correct.unsqueeze(0)).sum().item()
            gathered_total = accelerator.gather_for_metrics(total.unsqueeze(0)).sum().item()

            loss_total += gathered_loss
            correct_total += gathered_correct
            sample_total += gathered_total

    mean_loss = loss_total / max(sample_total, 1.0)
    mean_acc = correct_total / max(sample_total, 1.0)
    return mean_loss, mean_acc


def evaluate_with_predictions(model, loader, criterion, accelerator: Accelerator, is_graph=False, use_channels_last=False):
    model.eval()
    loss_total = 0.0
    correct_total = 0.0
    sample_total = 0.0

    all_logits = [] if accelerator.is_main_process else None
    all_labels = [] if accelerator.is_main_process else None

    with torch.no_grad():
        for batch in loader:
            if is_graph:
                labels = batch.y.view(-1)
                logits = model(batch)
            else:
                images, labels = batch
                if use_channels_last and images.ndim == 4:
                    images = images.contiguous(memory_format=torch.channels_last)
                logits = model(images)

            loss = criterion(logits, labels)

            correct, total = batch_stats(logits, labels)
            loss_sum = loss.detach().float() * float(labels.size(0))

            gathered_loss = accelerator.gather_for_metrics(loss_sum.unsqueeze(0)).sum().item()
            gathered_correct = accelerator.gather_for_metrics(correct.unsqueeze(0)).sum().item()
            gathered_total = accelerator.gather_for_metrics(total.unsqueeze(0)).sum().item()

            loss_total += gathered_loss
            correct_total += gathered_correct
            sample_total += gathered_total

            gathered_logits = accelerator.gather_for_metrics(logits.detach())
            gathered_labels = accelerator.gather_for_metrics(labels.detach())
            if accelerator.is_main_process:
                all_logits.append(gathered_logits.cpu())
                all_labels.append(gathered_labels.cpu())

    mean_loss = loss_total / max(sample_total, 1.0)
    mean_acc = correct_total / max(sample_total, 1.0)

    labels_np = None
    probs_np = None
    if accelerator.is_main_process and all_logits and all_labels:
        logits_np = torch.cat(all_logits, dim=0).numpy()
        labels_np = torch.cat(all_labels, dim=0).numpy()
        probs_np = logits_to_probs(logits_np)

    return mean_loss, mean_acc, labels_np, probs_np


def _resolve_resume_for_accelerate(resume_arg: str, run_dir: str):
    if not resume_arg:
        return None

    if resume_arg == "latest":
        latest = Path(run_dir) / "latest"
        if latest.exists():
            target = latest.resolve()
            if target.is_dir():
                candidate = target / "best.pt"
                if candidate.exists():
                    return str(candidate)
                return None
            return str(target)
        return None

    path = Path(resume_arg)
    if path.is_file():
        return str(path.resolve())
    if path.is_dir():
        candidate = path / "best.pt"
        if candidate.exists():
            return str(candidate.resolve())
    return None


def run_accelerate_training(args, bundle: DatasetBundle, run_dir: str):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    wandb_run = None
    if args.use_wandb and args.wandb_mode != "disabled" and accelerator.is_main_process:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
                dir=run_dir,
                mode=args.wandb_mode,
            )
        except Exception as exc:  # noqa: PERF203
            print(f"[WARN] wandb init failed, disable logging: {exc}")
            wandb_run = None

    train_loader, val_loader, test_loader, node_feature_dim = build_dataloaders(args, bundle)

    model = build_model(args, num_classes=bundle.num_classes, node_feature_dim=node_feature_dim)
    use_channels_last = bool(args.channels_last) and args.model == "resnet"
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    if bool(args.torch_compile):
        try:
            model = torch.compile(model, mode=args.torch_compile_mode)
            if accelerator.is_main_process:
                print(f"[Perf] torch.compile enabled (mode={args.torch_compile_mode})")
        except Exception as exc:  # noqa: PERF203
            if accelerator.is_main_process:
                print(f"[WARN] torch.compile disabled due to error: {exc}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    start_epoch = 0
    best_val_acc = -1.0
    best_ckpt_path = os.path.join(run_dir, "best.pt")

    resume_path = _resolve_resume_for_accelerate(args.resume, run_dir)
    if args.resume and resume_path is None and accelerator.is_main_process:
        print(f"[WARN] resume file not found or unsupported: {args.resume}")

    if resume_path:
        state = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
        )
        start_epoch = int(state.get("epoch", -1)) + 1
        best_val_acc = float(state.get("best_val_acc", -1.0))
        if accelerator.is_main_process:
            print(f"[Resume] loaded '{resume_path}' from epoch={start_epoch}")

    num_params = count_parameters(accelerator.unwrap_model(model))
    if accelerator.is_main_process:
        print(f"[Model] backend=accelerate model={args.model} trainable_params={num_params}")

    epoch_times = []
    is_graph = args.model in GRAPH_MODELS
    class_names = bundle.class_names
    best_val_labels = None
    best_val_probs = None
    best_val_per_class = None

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()

        train_loss, train_acc = step_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            accelerator=accelerator,
            optimizer=optimizer,
            is_graph=is_graph,
            grad_clip=args.grad_clip,
            use_channels_last=use_channels_last,
        )

        val_loss, val_acc, val_labels, val_probs = evaluate_with_predictions(
            model=model,
            loader=val_loader,
            criterion=criterion,
            accelerator=accelerator,
            is_graph=is_graph,
            use_channels_last=use_channels_last,
        )

        val_cls_metrics = {}
        val_per_class = []
        if accelerator.is_main_process and val_labels is not None and val_probs is not None:
            val_cls_metrics, val_per_class = compute_classification_metrics(val_labels, val_probs)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.perf_counter() - t0
        epoch_times.append(elapsed)

        if accelerator.is_main_process:
            print(
                f"[Epoch {epoch + 1}/{args.epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"time={elapsed:.1f}s"
            )
            if wandb_run is not None:
                payload = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "time/epoch_sec": elapsed,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                payload.update(prefix_metrics(val_cls_metrics, "val_cls"))
                wandb_run.log(payload, step=epoch + 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_checkpoint(
                    best_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_acc=best_val_acc,
                    config=vars(args),
                    accelerator=accelerator,
                )
                update_latest_symlink(run_dir, best_ckpt_path)
                if val_labels is not None and val_probs is not None:
                    best_val_labels = val_labels.copy()
                    best_val_probs = val_probs.copy()
                    best_val_per_class = list(val_per_class)

        accelerator.wait_for_everyone()

    if os.path.exists(best_ckpt_path):
        load_checkpoint(best_ckpt_path, model=model, optimizer=None, scheduler=None, accelerator=accelerator)

    test_loss, test_acc, test_labels, test_probs = evaluate_with_predictions(
        model=model,
        loader=test_loader,
        criterion=criterion,
        accelerator=accelerator,
        is_graph=is_graph,
        use_channels_last=use_channels_last,
    )

    test_cls_metrics = {}
    test_per_class = []
    if accelerator.is_main_process and test_labels is not None and test_probs is not None:
        test_cls_metrics, test_per_class = compute_classification_metrics(test_labels, test_probs)

    avg_epoch_time = sum(epoch_times) / max(1, len(epoch_times))

    if accelerator.is_main_process and wandb_run is not None:
        payload = {
            "best/val_acc": best_val_acc,
            "test/loss": test_loss,
            "test/acc": test_acc,
            "model/params": num_params,
            "time/avg_epoch_sec": avg_epoch_time,
        }
        payload.update(prefix_metrics(test_cls_metrics, "test_cls"))
        wandb_run.log(payload)

        if test_labels is not None and test_probs is not None:
            log_wandb_classification_artifacts(
                wandb_run,
                prefix="test_cls",
                labels=test_labels,
                probs=test_probs,
                class_names=class_names,
                per_class_rows=test_per_class,
                step=args.epochs,
            )
        if best_val_labels is not None and best_val_probs is not None:
            log_wandb_classification_artifacts(
                wandb_run,
                prefix="best_val_cls",
                labels=best_val_labels,
                probs=best_val_probs,
                class_names=class_names,
                per_class_rows=best_val_per_class,
                step=args.epochs,
            )

        wandb_run.summary["best_checkpoint"] = best_ckpt_path
        wandb_run.finish()

    return {
        "is_main_process": accelerator.is_main_process,
        "num_params": num_params,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "avg_epoch_time": float(avg_epoch_time),
        "best_checkpoint": best_ckpt_path if os.path.exists(best_ckpt_path) else "",
    }


def main():
    args = parse_args()
    args.image_size = _default_image_size(args)

    set_global_seed(args.seed)
    configure_torch_runtime(args)

    run_tag = args.run_name.strip()
    if not run_tag:
        run_tag = f"{args.dataset}_{args.model}_{args.train_backend}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    args.run_name = run_tag

    run_dir = os.path.join(args.output_dir, run_tag)
    ensure_dir(run_dir)
    if is_rank0_process():
        dump_config(os.path.join(run_dir, "config.json"), vars(args))

    bundle = load_dataset_bundle(dataset_name=args.dataset, seed=args.seed)
    bundle = maybe_subset_bundle(bundle, args)

    if is_rank0_process():
        print(f"[Data] source={bundle.source_name} image_key={bundle.image_key} label_key={bundle.label_key}")
        print(
            f"[Data] train={len(bundle.splits['train'])} val={len(bundle.splits['val'])} "
            f"test={len(bundle.splits['test'])} num_classes={bundle.num_classes}"
        )

    if args.train_backend == "transformers":
        summary = run_transformers_training(args, bundle, run_dir)
    else:
        summary = run_accelerate_training(args, bundle, run_dir)

    if summary.get("is_main_process", True):
        print(
            f"[Final] dataset={args.dataset} model={args.model} backend={args.train_backend} "
            f"n_segments={args.n_segments} use_xy={int(bool(args.use_xy))} "
            f"best_val_acc={summary['best_val_acc']:.4f} test_acc={summary['test_acc']:.4f}"
        )

        append_result_row(
            os.path.join(args.output_dir, "results.csv"),
            {
                "dataset": args.dataset,
                "model": args.model,
                "train_backend": args.train_backend,
                "n_segments": args.n_segments if args.model in GRAPH_MODELS else "-",
                "use_xy": int(bool(args.use_xy)) if args.model in GRAPH_MODELS else "-",
                "params": summary["num_params"],
                "val_acc": round(summary["best_val_acc"], 6),
                "test_acc": round(summary["test_acc"], 6),
                "time_per_epoch": round(summary["avg_epoch_time"], 4),
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "epochs": args.epochs,
                "seed": args.seed,
            },
        )


if __name__ == "__main__":
    main()
