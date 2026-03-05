from __future__ import annotations

import csv
import json
import os


RESULT_FIELDS = [
    "dataset",
    "model",
    "train_backend",
    "n_segments",
    "use_xy",
    "params",
    "val_acc",
    "test_acc",
    "time_per_epoch",
    "image_size",
    "batch_size",
    "lr",
    "epochs",
    "seed",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dump_config(path: str, config: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def append_result_row(csv_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})
