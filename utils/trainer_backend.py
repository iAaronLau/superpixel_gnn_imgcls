from __future__ import annotations

import math
import os
import time
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch as PyGBatch

from data_utils.graph_dataset import GraphAugmentConfig, GraphBuildConfig, HFGraphDataset
from data_utils.image_dataset import HFImageDictDataset, build_image_transform
from models import build_model
from utils.checkpoint import update_latest_symlink
from utils.classification_metrics import (
    compute_classification_metrics,
    log_wandb_classification_artifacts,
    logits_to_probs,
    prefix_metrics,
)
from utils.metrics import count_parameters

try:
    from transformers import PreTrainedModel, PretrainedConfig, Trainer, TrainerCallback, TrainingArguments
    from transformers.modeling_outputs import SequenceClassifierOutput

    try:
        from transformers.trainer_utils import get_last_checkpoint
    except Exception:  # noqa: PERF203
        get_last_checkpoint = None

    _TRANSFORMERS_AVAILABLE = True
    _TRANSFORMERS_IMPORT_ERROR = None
except Exception as exc:  # noqa: PERF203
    PreTrainedModel = object
    PretrainedConfig = object
    Trainer = object
    TrainerCallback = object
    TrainingArguments = None
    SequenceClassifierOutput = None
    get_last_checkpoint = None
    _TRANSFORMERS_AVAILABLE = False
    _TRANSFORMERS_IMPORT_ERROR = exc


GRAPH_MODELS = {"gcn", "gat", "graph_transformer"}


class SuperpixelConfig(PretrainedConfig):
    model_type = "superpixel_gnn_classifier"

    def __init__(
        self,
        model_name: str = "resnet",
        dataset_name: str = "cifar10",
        image_size: int = 64,
        num_labels: int = 10,
        node_feature_dim: int = 5,
        edge_feature_dim: int = 0,
        resnet_name: str = "resnet18",
        channels_last: bool = True,
        hidden_dim: int = 256,
        gnn_layers: int = 4,
        gat_heads: int = 8,
        dropout: float = 0.2,
        pooling: str = "meanmax",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.num_labels = num_labels
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.resnet_name = resnet_name
        self.channels_last = channels_last
        self.hidden_dim = hidden_dim
        self.gnn_layers = gnn_layers
        self.gat_heads = gat_heads
        self.dropout = dropout
        self.pooling = pooling


class SuperpixelForImageClassification(PreTrainedModel):
    config_class = SuperpixelConfig
    base_model_prefix = "backbone"

    def __init__(self, config: SuperpixelConfig):
        super().__init__(config)
        args = SimpleNamespace(
            dataset=config.dataset_name,
            model=config.model_name,
            image_size=config.image_size,
            resnet_name=config.resnet_name,
            hidden_dim=config.hidden_dim,
            gnn_layers=config.gnn_layers,
            gat_heads=config.gat_heads,
            dropout=config.dropout,
            pooling=config.pooling,
        )
        node_feature_dim = None if config.model_name == "resnet" else config.node_feature_dim
        edge_feature_dim = None if config.model_name == "resnet" else config.edge_feature_dim
        self.backbone = build_model(
            args=args,
            num_classes=config.num_labels,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
        )
        self.post_init()

    def forward(self, pixel_values=None, graph_data=None, labels=None, **kwargs):
        if self.config.model_name == "resnet":
            if pixel_values is None:
                raise ValueError("`pixel_values` is required for resnet model.")
            if self.config.channels_last and pixel_values.ndim == 4:
                pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
            logits = self.backbone(pixel_values)
        else:
            if graph_data is None:
                raise ValueError("`graph_data` is required for graph models.")
            logits = self.backbone(graph_data)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)


def _scheduler_for_transformers(name: str) -> str:
    if name == "cosine":
        return "cosine"
    if name == "step":
        return "linear"
    return "constant"


def _find_latest_checkpoint_dir(output_dir: str) -> str | None:
    path = Path(output_dir)
    candidates = []
    for entry in path.glob("checkpoint-*"):
        suffix = entry.name.split("-")[-1]
        if suffix.isdigit() and entry.is_dir():
            candidates.append((int(suffix), entry))
    if not candidates:
        return None
    return str(max(candidates, key=lambda x: x[0])[1].resolve())


def resolve_resume_checkpoint(resume_arg: str, run_dir: str):
    if not resume_arg:
        return None

    resume_arg = resume_arg.strip().lower()
    if resume_arg in {"latest", "last"}:
        latest = Path(run_dir) / "latest"
        if latest.exists():
            return str(latest.resolve())
        return _find_latest_checkpoint_dir(run_dir)

    if resume_arg == "auto":
        if get_last_checkpoint is not None:
            try:
                return get_last_checkpoint(run_dir)
            except Exception:  # noqa: PERF203
                return _find_latest_checkpoint_dir(run_dir)
        return _find_latest_checkpoint_dir(run_dir)

    path = Path(resume_arg)
    if path.exists():
        return str(path.resolve())
    return None


class LatestCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            fallback = _find_latest_checkpoint_dir(args.output_dir)
            if fallback is None:
                return control
            checkpoint_dir = Path(fallback)

        update_latest_symlink(args.output_dir, str(checkpoint_dir))
        return control


class EpochTimerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self._epoch_start = None
        self.epoch_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start = time.perf_counter()
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._epoch_start is not None:
            elapsed = time.perf_counter() - self._epoch_start
            self.epoch_times.append(elapsed)
        self._epoch_start = None
        return control


class WandbBestCheckpointCallback(TrainerCallback):
    def __init__(self, run_name: str):
        super().__init__()
        self.run_name = run_name

    def on_train_end(self, args, state, control, **kwargs):
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return control

        if state.best_model_checkpoint is None:
            return control

        try:
            import wandb
        except Exception:  # noqa: PERF203
            return control

        if wandb.run is None:
            return control

        artifact = wandb.Artifact(f"{self.run_name}-best-checkpoint", type="model")
        artifact.add_dir(state.best_model_checkpoint)
        wandb.log_artifact(artifact)
        return control


def image_data_collator(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features], dim=0)
    labels = torch.stack([f["labels"] for f in features], dim=0)
    return {"pixel_values": pixel_values, "labels": labels}


def graph_data_collator(features):
    graph_data = PyGBatch.from_data_list(features)
    labels = graph_data.y.view(-1)
    return {"graph_data": graph_data, "labels": labels}


def build_compute_metrics_fn():
    def _metrics(eval_pred):
        logits, labels = eval_pred
        probs = logits_to_probs(np.asarray(logits))
        metrics, _ = compute_classification_metrics(np.asarray(labels), probs)
        return metrics

    return _metrics


class HFGraphTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        prepared = super()._prepare_inputs(inputs)
        if isinstance(prepared, dict):
            graph_data = prepared.get("graph_data")
            if graph_data is not None and hasattr(graph_data, "to"):
                prepared["graph_data"] = graph_data.to(self.args.device)
        return prepared


def build_trainer_datasets(args, bundle):
    is_graph = args.model in GRAPH_MODELS

    if not is_graph:
        train_t = build_image_transform(args.dataset, args.image_size, train=True)
        eval_t = build_image_transform(args.dataset, args.image_size, train=False)

        train_ds = HFImageDictDataset(bundle.splits["train"], bundle.image_key, bundle.label_key, train_t)
        val_ds = HFImageDictDataset(bundle.splits["val"], bundle.image_key, bundle.label_key, eval_t)
        test_ds = HFImageDictDataset(bundle.splits["test"], bundle.image_key, bundle.label_key, eval_t)
        return train_ds, val_ds, test_ds, None, image_data_collator

    graph_cfg = GraphBuildConfig(
        image_size=args.image_size,
        n_segments=args.n_segments,
        use_xy=bool(args.use_xy),
        compactness=args.slic_compactness,
        sigma=args.slic_sigma,
        include_rgb_stats=bool(args.graph_include_rgb_stats),
        include_hsv_stats=bool(args.graph_include_hsv_stats),
        include_lab_stats=bool(args.graph_include_lab_stats),
        include_shape_stats=bool(args.graph_include_shape_stats),
        include_patch_features=bool(args.graph_include_patch_features),
        patch_embed_size=args.graph_patch_embed_size,
        mask_patch_background=bool(args.graph_mask_patch_background),
        cache_version=args.graph_cache_version,
    )
    augment_cfg = GraphAugmentConfig(
        node_drop_prob=args.graph_node_drop_prob,
        edge_drop_prob=args.graph_edge_drop_prob,
        feature_mask_prob=args.graph_feature_mask_prob,
        feature_noise_std=args.graph_feature_noise_std,
        edge_noise_std=args.graph_edge_noise_std,
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
        cache_version=args.graph_cache_version,
        augment_cfg=augment_cfg,
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
        cache_version=args.graph_cache_version,
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
        cache_version=args.graph_cache_version,
    )

    node_feature_dim, edge_feature_dim = train_ds.feature_dims()
    return train_ds, val_ds, test_ds, node_feature_dim, edge_feature_dim, graph_data_collator


def _build_hf_config(args, bundle, node_feature_dim: int | None, edge_feature_dim: int | None) -> SuperpixelConfig:
    if args.model == "resnet":
        node_dim = 0
        edge_dim = 0
    else:
        node_dim = node_feature_dim if node_feature_dim is not None else 0
        edge_dim = edge_feature_dim if edge_feature_dim is not None else 0

    return SuperpixelConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        image_size=args.image_size,
        num_labels=bundle.num_classes,
        node_feature_dim=node_dim,
        edge_feature_dim=edge_dim,
        resnet_name=getattr(args, "resnet_name", "resnet18"),
        channels_last=bool(getattr(args, "channels_last", 1)),
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        pooling=args.pooling,
    )


def run_transformers_training(args, bundle, run_dir: str):
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers is required for --train_backend transformers. "
            f"Original import error: {_TRANSFORMERS_IMPORT_ERROR}"
        )

    if args.use_wandb and args.wandb_mode != "disabled":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_NAME", args.run_name)
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    train_ds, val_ds, test_ds, node_feature_dim, edge_feature_dim, data_collator = build_trainer_datasets(args, bundle)

    config = _build_hf_config(args, bundle, node_feature_dim, edge_feature_dim)
    model = SuperpixelForImageClassification(config)
    if args.model == "resnet" and bool(getattr(args, "channels_last", 1)):
        model = model.to(memory_format=torch.channels_last)

    eval_strategy = args.eval_strategy
    save_strategy = args.save_strategy
    if save_strategy == "auto":
        save_strategy = eval_strategy

    eval_steps = None
    if eval_strategy == "steps":
        eval_steps = args.eval_steps if args.eval_steps > 0 else max(1, args.logging_steps)

    save_steps = None
    if save_strategy == "steps":
        if args.save_steps > 0:
            save_steps = args.save_steps
        elif eval_steps is not None:
            save_steps = eval_steps
        else:
            save_steps = max(1, args.logging_steps)

    report_to = ["wandb"] if args.use_wandb and args.wandb_mode != "disabled" else []
    prefetch_factor = args.prefetch_factor if args.num_workers > 0 else None
    load_best_at_end = save_strategy == eval_strategy
    if not load_best_at_end:
        print(
            f"[WARN] load_best_model_at_end disabled because eval_strategy={eval_strategy} "
            f"and save_strategy={save_strategy} differ."
        )

    hub_model_id = args.hub_model_id.strip() or None
    hub_token = args.hub_token.strip() or None

    training_kwargs = dict(
        output_dir=run_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type=_scheduler_for_transformers(args.scheduler),
        num_train_epochs=args.epochs,
        logging_strategy="steps",
        logging_steps=max(1, args.logging_steps),
        evaluation_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=args.checkpoints_total_limit,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=bool(args.persistent_workers and args.num_workers > 0),
        dataloader_prefetch_factor=prefetch_factor,
        bf16=args.mixed_precision == "bf16",
        fp16=args.mixed_precision == "fp16",
        remove_unused_columns=False,
        report_to=report_to,
        run_name=args.run_name,
        seed=args.seed,
        load_best_model_at_end=load_best_at_end,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        ddp_find_unused_parameters=bool(args.ddp_find_unused_parameters),
        max_grad_norm=args.grad_clip if args.grad_clip > 0 else 0.0,
        save_safetensors=bool(args.save_safetensors),
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=hub_model_id,
        hub_private_repo=bool(args.hub_private_repo),
        hub_strategy=args.hub_strategy,
        hub_token=hub_token,
        torch_compile=bool(getattr(args, "torch_compile", 0)),
        torch_compile_backend="inductor",
        torch_compile_mode=getattr(args, "torch_compile_mode", "reduce-overhead"),
    )

    def _build_training_args_compatible(kwargs: dict):
        sig = inspect.signature(TrainingArguments.__init__)
        supported = set(sig.parameters.keys())
        adapted = dict(kwargs)

        # Transformers versions may use either `evaluation_strategy` or `eval_strategy`.
        if "evaluation_strategy" in adapted and "evaluation_strategy" not in supported and "eval_strategy" in supported:
            adapted["eval_strategy"] = adapted.pop("evaluation_strategy")
        if "eval_strategy" in adapted and "eval_strategy" not in supported and "evaluation_strategy" in supported:
            adapted["evaluation_strategy"] = adapted.pop("eval_strategy")

        removed = []
        for key in list(adapted.keys()):
            if key not in supported:
                removed.append(key)
                adapted.pop(key)

        if removed:
            print(f"[WARN] Dropping unsupported TrainingArguments keys: {sorted(removed)}")

        return TrainingArguments(**adapted)

    try:
        training_args = _build_training_args_compatible(training_kwargs)
    except TypeError as exc:
        # Final defensive pass for corner cases in very old/new versions.
        print(f"[WARN] TrainingArguments fallback due to version mismatch: {exc}")
        for key in [
            "dataloader_persistent_workers",
            "dataloader_prefetch_factor",
            "ddp_find_unused_parameters",
            "save_safetensors",
            "hub_private_repo",
            "hub_strategy",
            "hub_token",
            "evaluation_strategy",
            "eval_strategy",
        ]:
            training_kwargs.pop(key, None)
        training_args = _build_training_args_compatible(training_kwargs)

    if bool(getattr(args, "torch_compile", 0)) and not bool(getattr(training_args, "torch_compile", False)):
        try:
            model = torch.compile(model, mode=getattr(args, "torch_compile_mode", "reduce-overhead"))
            print(
                "[Perf] training_args has no torch_compile support; "
                f"fallback to manual torch.compile(mode={getattr(args, 'torch_compile_mode', 'reduce-overhead')})"
            )
        except Exception as exc:  # noqa: PERF203
            print(f"[WARN] manual torch.compile fallback disabled: {exc}")

    timer_cb = EpochTimerCallback()
    callbacks = [LatestCheckpointCallback(), timer_cb]
    if args.use_wandb and args.wandb_mode != "disabled":
        callbacks.append(WandbBestCheckpointCallback(args.run_name))

    trainer = HFGraphTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics_fn(),
        callbacks=callbacks,
    )

    resume_checkpoint = resolve_resume_checkpoint(args.resume, run_dir)
    if args.resume and resume_checkpoint is None:
        print(f"[WARN] resume checkpoint not found: {args.resume}")

    train_start = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_elapsed = time.perf_counter() - train_start

    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    val_output = trainer.predict(val_ds, metric_key_prefix="val_final")
    test_output = trainer.predict(test_ds, metric_key_prefix="test")
    test_metrics = test_output.metrics

    val_labels = np.asarray(val_output.label_ids)
    val_probs = logits_to_probs(np.asarray(val_output.predictions))
    val_cls_metrics, val_per_class = compute_classification_metrics(val_labels, val_probs)

    test_labels = np.asarray(test_output.label_ids)
    test_probs = logits_to_probs(np.asarray(test_output.predictions))
    test_cls_metrics, test_per_class = compute_classification_metrics(test_labels, test_probs)

    num_params = count_parameters(trainer.model)
    best_ckpt = trainer.state.best_model_checkpoint
    best_val_acc = trainer.state.best_metric
    if best_val_acc is None:
        best_val_acc = val_metrics.get("val_accuracy", float("nan"))

    if best_ckpt is None:
        best_ckpt = _find_latest_checkpoint_dir(run_dir)
    if best_ckpt:
        update_latest_symlink(run_dir, best_ckpt)

    if bool(args.push_to_hub):
        try:
            trainer.push_to_hub(commit_message=f"Training complete: {args.run_name}")
        except Exception as exc:  # noqa: PERF203
            print(f"[WARN] push_to_hub failed: {exc}")

    avg_epoch_time = (
        sum(timer_cb.epoch_times) / len(timer_cb.epoch_times)
        if timer_cb.epoch_times
        else train_elapsed / max(1, math.ceil(args.epochs))
    )

    test_acc = test_metrics.get("test_accuracy", float("nan"))
    if isinstance(test_acc, float) and np.isnan(test_acc):
        test_acc = test_cls_metrics.get("accuracy", float("nan"))
    test_loss = test_metrics.get("test_loss", float("nan"))

    if args.use_wandb and args.wandb_mode != "disabled" and trainer.is_world_process_zero():
        try:
            import wandb

            wandb_run = wandb.run
        except Exception:  # noqa: PERF203
            wandb_run = None

        if wandb_run is not None:
            payload = {}
            payload.update(prefix_metrics(val_cls_metrics, "val_final_cls"))
            payload.update(prefix_metrics(test_cls_metrics, "test_cls"))
            if payload:
                wandb_run.log(payload, step=trainer.state.global_step)

            log_wandb_classification_artifacts(
                wandb_run,
                prefix="val_final_cls",
                labels=val_labels,
                probs=val_probs,
                class_names=bundle.class_names,
                per_class_rows=val_per_class,
                step=trainer.state.global_step,
            )
            log_wandb_classification_artifacts(
                wandb_run,
                prefix="test_cls",
                labels=test_labels,
                probs=test_probs,
                class_names=bundle.class_names,
                per_class_rows=test_per_class,
                step=trainer.state.global_step,
            )

    return {
        "is_main_process": trainer.is_world_process_zero(),
        "num_params": num_params,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "avg_epoch_time": float(avg_epoch_time),
        "best_checkpoint": best_ckpt,
    }
