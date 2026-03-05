from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm.auto import tqdm


@dataclass
class GraphBuildConfig:
    image_size: int
    n_segments: int
    use_xy: bool
    compactness: float = 10.0
    sigma: float = 1.0


try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _BILINEAR = Image.BILINEAR


def _to_pil_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def _compute_edges(segment_map: np.ndarray) -> np.ndarray:
    pairs = set()

    left = segment_map[:, :-1]
    right = segment_map[:, 1:]
    mask = left != right
    if np.any(mask):
        for src, dst in zip(left[mask].tolist(), right[mask].tolist()):
            pairs.add((src, dst) if src < dst else (dst, src))

    top = segment_map[:-1, :]
    bottom = segment_map[1:, :]
    mask = top != bottom
    if np.any(mask):
        for src, dst in zip(top[mask].tolist(), bottom[mask].tolist()):
            pairs.add((src, dst) if src < dst else (dst, src))

    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)

    undirected = np.array(list(pairs), dtype=np.int64)
    src = np.concatenate([undirected[:, 0], undirected[:, 1]], axis=0)
    dst = np.concatenate([undirected[:, 1], undirected[:, 0]], axis=0)
    return np.stack([src, dst], axis=0)


def build_superpixel_graph(image, label: int, config: GraphBuildConfig) -> Data:
    pil_image = _to_pil_rgb(image)
    pil_image = pil_image.resize((config.image_size, config.image_size), resample=_BILINEAR)

    arr = np.asarray(pil_image, dtype=np.float32) / 255.0
    h, w, _ = arr.shape

    segments = slic(
        arr,
        n_segments=config.n_segments,
        compactness=config.compactness,
        sigma=config.sigma,
        start_label=0,
        channel_axis=-1,
    )

    flat_seg = segments.reshape(-1)
    _, inverse = np.unique(flat_seg, return_inverse=True)
    seg_map = inverse.reshape(h, w)

    num_nodes = int(seg_map.max()) + 1
    pixels = arr.reshape(-1, 3)
    node_idx = seg_map.reshape(-1)

    counts = np.bincount(node_idx, minlength=num_nodes).astype(np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)

    rgb_means = []
    for ch in range(3):
        sums = np.bincount(node_idx, weights=pixels[:, ch], minlength=num_nodes).astype(np.float32)
        rgb_means.append((sums / counts)[:, None])
    node_features = np.concatenate(rgb_means, axis=1)

    if config.use_xy:
        yy, xx = np.indices((h, w), dtype=np.float32)
        x_flat = xx.reshape(-1) / max(1.0, float(w - 1))
        y_flat = yy.reshape(-1) / max(1.0, float(h - 1))

        cx = np.bincount(node_idx, weights=x_flat, minlength=num_nodes).astype(np.float32) / counts
        cy = np.bincount(node_idx, weights=y_flat, minlength=num_nodes).astype(np.float32) / counts
        node_features = np.concatenate([node_features, cx[:, None], cy[:, None]], axis=1)

    edge_index = _compute_edges(seg_map)

    data = Data(
        x=torch.from_numpy(node_features).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        y=torch.tensor([int(label)], dtype=torch.long),
    )
    return data


class HFGraphDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        image_key: str,
        label_key: str,
        split_name: str,
        dataset_name: str,
        graph_cfg: GraphBuildConfig,
        use_cache: bool = True,
        cache_dir: str = "graph_cache",
        cache_version: str = "v1",
    ):
        self.dataset = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
        self.split_name = split_name
        self.dataset_name = dataset_name
        self.graph_cfg = graph_cfg
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        self.cache_path = os.path.join(
            cache_dir,
            (
                f"{dataset_name}_{split_name}_img{graph_cfg.image_size}_"
                f"seg{graph_cfg.n_segments}_xy{int(graph_cfg.use_xy)}_{cache_version}.pt"
            ),
        )

        self.graphs = None
        self._prepare_graphs()

    def _prepare_graphs(self) -> None:
        def _load_cache():
            try:
                return torch.load(self.cache_path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(self.cache_path, map_location="cpu")

        if self.use_cache and os.path.exists(self.cache_path):
            self.graphs = _load_cache()
            return

        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            lock_path = f"{self.cache_path}.lock"
            lock_acquired = False
            lock_wait_sec = 0

            while not lock_acquired:
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    lock_acquired = True
                except FileExistsError:
                    if os.path.exists(self.cache_path):
                        self.graphs = _load_cache()
                        return
                    time.sleep(1.0)
                    lock_wait_sec += 1
                    if lock_wait_sec > 7200:
                        raise TimeoutError(f"Waited too long for graph cache lock: {lock_path}")

            try:
                if os.path.exists(self.cache_path):
                    self.graphs = _load_cache()
                    return

                graphs = []
                iterator = tqdm(self.dataset, desc=f"Build graphs [{self.split_name}]", leave=False)
                for sample in iterator:
                    graph = build_superpixel_graph(
                        image=sample[self.image_key],
                        label=int(sample[self.label_key]),
                        config=self.graph_cfg,
                    )
                    graphs.append(graph)

                self.graphs = graphs
                torch.save(self.graphs, self.cache_path)
            finally:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
            return

        graphs = []
        iterator = tqdm(self.dataset, desc=f"Build graphs [{self.split_name}]", leave=False)
        for sample in iterator:
            graph = build_superpixel_graph(
                image=sample[self.image_key],
                label=int(sample[self.label_key]),
                config=self.graph_cfg,
            )
            graphs.append(graph)
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]
