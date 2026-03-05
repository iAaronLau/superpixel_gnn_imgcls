from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from datasets import ClassLabel, Dataset, DatasetDict, Image as HFImage, load_dataset


@dataclass
class DatasetBundle:
    splits: Dict[str, Dataset]
    image_key: str
    label_key: str
    num_classes: int
    class_names: list[str] | None
    source_name: str


DATASET_CANDIDATES = {
    "cifar10": [
        ("cifar10", None),
    ],
    "imagewoof": [
        ("frgfm/imagewoof", None),
        ("frgfm/imagewoof", "160px"),
        ("frgfm/imagewoof", "320px"),
        ("imagewoof", None),
    ],
}


def _load_dataset_with_fallback(dataset_name: str) -> Tuple[DatasetDict, str]:
    candidates = DATASET_CANDIDATES.get(dataset_name, [(dataset_name, None)])
    errors = []
    for repo_id, config_name in candidates:
        try:
            if config_name is None:
                ds = load_dataset(repo_id)
            else:
                ds = load_dataset(repo_id, config_name)
            return ds, f"{repo_id}:{config_name or 'default'}"
        except Exception as exc:  # noqa: PERF203
            errors.append(f"{repo_id}:{config_name or 'default'} -> {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to load dataset '{dataset_name}'. Tried:\n{joined}")


def _infer_keys(split: Dataset) -> Tuple[str, str]:
    image_key = None
    label_key = None

    for key, feature in split.features.items():
        if isinstance(feature, HFImage):
            image_key = key
        if isinstance(feature, ClassLabel):
            label_key = key

    if image_key is None:
        for candidate in ["image", "img", "pixel_values"]:
            if candidate in split.column_names:
                image_key = candidate
                break

    if label_key is None:
        for candidate in ["label", "labels", "fine_label", "category", "target"]:
            if candidate in split.column_names:
                label_key = candidate
                break

    if image_key is None or label_key is None:
        raise ValueError(
            f"Cannot infer image/label columns from dataset columns: {split.column_names}"
        )

    return image_key, label_key


def _num_classes_from_feature(split: Dataset, label_key: str) -> Tuple[int, list[str] | None]:
    feature = split.features[label_key]
    if isinstance(feature, ClassLabel):
        return feature.num_classes, list(feature.names)

    labels = set(split[label_key])
    return len(labels), None


def _split_train_val(train_split: Dataset, seed: int, val_size: int | None = None, val_ratio: float | None = None) -> Tuple[Dataset, Dataset]:
    if val_size is None and val_ratio is None:
        raise ValueError("Either val_size or val_ratio must be provided.")

    test_size = val_size if val_size is not None else val_ratio
    split = train_split.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    return split["train"], split["test"]


def load_dataset_bundle(dataset_name: str, seed: int, cifar_val_size: int = 5000, imagewoof_val_ratio: float = 0.1) -> DatasetBundle:
    dataset_name = dataset_name.lower()
    ds, source_name = _load_dataset_with_fallback(dataset_name)

    split_names = list(ds.keys())
    if not split_names:
        raise RuntimeError(f"Dataset '{dataset_name}' has no available splits.")

    train_key = "train" if "train" in ds else split_names[0]
    train_split = ds[train_key]

    image_key, label_key = _infer_keys(train_split)

    if dataset_name == "cifar10":
        if "test" not in ds:
            raise RuntimeError("CIFAR-10 requires an official test split, but none was found.")
        train_split, val_split = _split_train_val(train_split, seed=seed, val_size=cifar_val_size)
        test_split = ds["test"]
    elif dataset_name == "imagewoof":
        val_like = ds.get("validation", ds.get("val"))
        test_like = ds.get("test")

        if test_like is not None and val_like is not None:
            val_split = val_like
            test_split = test_like
        elif test_like is not None and val_like is None:
            train_split, val_split = _split_train_val(train_split, seed=seed, val_ratio=imagewoof_val_ratio)
            test_split = test_like
        elif test_like is None and val_like is not None:
            train_split, val_split = _split_train_val(train_split, seed=seed, val_ratio=imagewoof_val_ratio)
            test_split = val_like
        else:
            first = train_split.train_test_split(test_size=0.2, seed=seed, shuffle=True)
            second = first["test"].train_test_split(test_size=0.5, seed=seed, shuffle=True)
            train_split = first["train"]
            val_split = second["train"]
            test_split = second["test"]
    else:
        test_like = ds.get("test")
        val_like = ds.get("validation", ds.get("val"))

        if test_like is None:
            first = train_split.train_test_split(test_size=0.2, seed=seed, shuffle=True)
            second = first["test"].train_test_split(test_size=0.5, seed=seed, shuffle=True)
            train_split = first["train"]
            val_split = second["train"]
            test_split = second["test"]
        elif val_like is None:
            train_split, val_split = _split_train_val(train_split, seed=seed, val_ratio=0.1)
            test_split = test_like
        else:
            val_split = val_like
            test_split = test_like

    num_classes, class_names = _num_classes_from_feature(train_split, label_key)

    return DatasetBundle(
        splits={"train": train_split, "val": val_split, "test": test_split},
        image_key=image_key,
        label_key=label_key,
        num_classes=num_classes,
        class_names=class_names,
        source_name=source_name,
    )
