from __future__ import annotations

import math

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        log_loss,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    _SKLEARN_AVAILABLE = True
except Exception:  # noqa: PERF203
    _SKLEARN_AVAILABLE = False


def normalize_class_names(class_names: list[str] | None, num_classes: int) -> list[str]:
    if class_names is None or len(class_names) != num_classes:
        return [str(i) for i in range(num_classes)]
    return [str(x) for x in class_names]


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom = np.clip(denom, a_min=1e-12, a_max=None)
    return exp / denom


def compute_classification_metrics(labels: np.ndarray, probs: np.ndarray) -> tuple[dict, list[dict]]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    preds = np.argmax(probs, axis=1)

    num_classes = probs.shape[1]
    one_hot = np.eye(num_classes, dtype=np.float64)[labels]

    metrics = {
        "accuracy": float((preds == labels).mean()),
    }

    if not _SKLEARN_AVAILABLE:
        return metrics, []

    metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, preds))
    for avg in ["macro", "micro", "weighted"]:
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average=avg, zero_division=0)
        metrics[f"precision_{avg}"] = float(p)
        metrics[f"recall_{avg}"] = float(r)
        metrics[f"f1_{avg}"] = float(f1)

    try:
        metrics["mcc"] = float(matthews_corrcoef(labels, preds))
    except Exception:  # noqa: PERF203
        metrics["mcc"] = float("nan")

    try:
        metrics["cohen_kappa"] = float(cohen_kappa_score(labels, preds))
    except Exception:  # noqa: PERF203
        metrics["cohen_kappa"] = float("nan")

    try:
        metrics["log_loss"] = float(log_loss(labels, probs, labels=list(range(num_classes))))
    except Exception:  # noqa: PERF203
        metrics["log_loss"] = float("nan")

    if num_classes == 2:
        pos_prob = probs[:, 1]
        try:
            metrics["roc_auc"] = float(roc_auc_score(labels, pos_prob))
        except Exception:  # noqa: PERF203
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(labels, pos_prob))
        except Exception:  # noqa: PERF203
            metrics["pr_auc"] = float("nan")
    else:
        try:
            metrics["roc_auc_ovr_macro"] = float(roc_auc_score(one_hot, probs, multi_class="ovr", average="macro"))
        except Exception:  # noqa: PERF203
            metrics["roc_auc_ovr_macro"] = float("nan")
        try:
            metrics["roc_auc_ovr_weighted"] = float(
                roc_auc_score(one_hot, probs, multi_class="ovr", average="weighted")
            )
        except Exception:  # noqa: PERF203
            metrics["roc_auc_ovr_weighted"] = float("nan")
        try:
            metrics["pr_auc_macro"] = float(average_precision_score(one_hot, probs, average="macro"))
        except Exception:  # noqa: PERF203
            metrics["pr_auc_macro"] = float("nan")
        try:
            metrics["pr_auc_weighted"] = float(average_precision_score(one_hot, probs, average="weighted"))
        except Exception:  # noqa: PERF203
            metrics["pr_auc_weighted"] = float("nan")

    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )
    per_class_rows = []
    for idx in range(num_classes):
        per_class_rows.append(
            {
                "class_id": idx,
                "precision": float(per_class_precision[idx]),
                "recall": float(per_class_recall[idx]),
                "f1": float(per_class_f1[idx]),
                "support": int(per_class_support[idx]),
            }
        )

    return metrics, per_class_rows


def prefix_metrics(metrics: dict, prefix: str) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (float, int)) and not (isinstance(value, float) and math.isnan(value)):
            out[f"{prefix}/{key}"] = float(value)
    return out


def log_wandb_classification_artifacts(
    wandb_run,
    prefix: str,
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: list[str] | None,
    per_class_rows: list[dict] | None = None,
    step: int | None = None,
):
    if wandb_run is None:
        return

    try:
        import wandb
    except Exception:  # noqa: PERF203
        return

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    preds = np.argmax(probs, axis=1)
    names = normalize_class_names(class_names, probs.shape[1])

    payload = {}

    try:
        payload[f"{prefix}/confusion_matrix"] = wandb.plot.confusion_matrix(
            y_true=labels.tolist(),
            preds=preds.tolist(),
            class_names=names,
        )
    except Exception:  # noqa: PERF203
        pass

    try:
        payload[f"{prefix}/roc_curve"] = wandb.plot.roc_curve(labels, probs, labels=names)
    except Exception:  # noqa: PERF203
        pass

    try:
        payload[f"{prefix}/pr_curve"] = wandb.plot.pr_curve(labels, probs, labels=names)
    except Exception:  # noqa: PERF203
        pass

    if per_class_rows:
        table = wandb.Table(columns=["class", "precision", "recall", "f1", "support"])
        for row in per_class_rows:
            class_name = names[row["class_id"]] if row["class_id"] < len(names) else str(row["class_id"])
            table.add_data(class_name, row["precision"], row["recall"], row["f1"], row["support"])
        payload[f"{prefix}/per_class_metrics"] = table

    if payload:
        wandb_run.log(payload, step=step)
