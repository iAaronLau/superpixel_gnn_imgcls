from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from datasets import ClassLabel, Dataset, DatasetDict, Features, Image as HFImage, load_dataset


@dataclass
class DatasetBundle:
    splits: Dict[str, Dataset]
    image_key: str
    label_key: str
    num_classes: int
    class_names: list[str] | None
    source_name: str


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
IMAGENET_STYLE_DATASETS = {"imagewoof", "imagenette2"}

REMOTE_DATASET_CANDIDATES = {
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


LOCAL_DATASET_DIR_CANDIDATES = {
    "imagewoof": ["imagewoof2", "imagewoof"],
    "imagenette2": ["imagenette2", "imagenette2-320", "imagenette2-160"],
}


def _local_data_roots() -> list[Path]:
    roots = []
    for root in [PROJECT_ROOT / "data" / "hf_cache", PROJECT_ROOT / "data"]:
        if root.exists():
            roots.append(root)
    return roots


def _resolve_local_dataset_root(dataset_name: str) -> Path | None:
    for base_root in _local_data_roots():
        for candidate_name in LOCAL_DATASET_DIR_CANDIDATES.get(dataset_name, []):
            candidate_root = base_root / candidate_name
            if candidate_root.is_dir() and (candidate_root / "train").is_dir():
                return candidate_root
    return None


def _iter_image_paths(class_dir: Path) -> list[str]:
    image_paths = []
    for path in sorted(class_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(str(path))
    return image_paths


def _build_local_imagefolder_split(split_dir: Path, class_names: list[str]) -> Dataset:
    image_paths: list[str] = []
    labels: list[int] = []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        class_image_paths = _iter_image_paths(class_dir)
        image_paths.extend(class_image_paths)
        labels.extend([class_to_idx[class_name]] * len(class_image_paths))

    if not image_paths:
        raise RuntimeError(f"No image files found under local dataset split: {split_dir}")

    features = Features(
        {
            "image": HFImage(),
            "label": ClassLabel(names=class_names),
        }
    )
    return Dataset.from_dict({"image": image_paths, "label": labels}, features=features)


def _load_local_imagefolder_dataset(dataset_name: str) -> Tuple[DatasetDict, str] | None:
    dataset_root = _resolve_local_dataset_root(dataset_name)
    if dataset_root is None:
        return None

    split_dirs = {path.name: path for path in dataset_root.iterdir() if path.is_dir()}
    train_dir = split_dirs.get("train")
    if train_dir is None:
        raise RuntimeError(f"Local dataset root is missing a train split: {dataset_root}")

    class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    if not class_names:
        raise RuntimeError(f"No class directories found under: {train_dir}")

    dataset_splits = {"train": _build_local_imagefolder_split(train_dir, class_names)}

    for split_name in ("val", "validation", "test"):
        split_dir = split_dirs.get(split_name)
        if split_dir is not None:
            dataset_splits[split_name] = _build_local_imagefolder_split(split_dir, class_names)

    return DatasetDict(dataset_splits), f"local_imagefolder:{dataset_root}"


def _load_dataset_with_fallback(dataset_name: str) -> Tuple[DatasetDict, str]:
    errors = []

    try:
        local_dataset = _load_local_imagefolder_dataset(dataset_name)
        if local_dataset is not None:
            return local_dataset
    except Exception as exc:  # noqa: PERF203
        errors.append(f"local:{dataset_name} -> {exc}")

    candidates = REMOTE_DATASET_CANDIDATES.get(dataset_name, [(dataset_name, None)])
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
    elif dataset_name in IMAGENET_STYLE_DATASETS:
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
