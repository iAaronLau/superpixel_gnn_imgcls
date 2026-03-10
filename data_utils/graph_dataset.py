from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2hsv, rgb2lab
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
    include_rgb_stats: bool = True
    include_hsv_stats: bool = True
    include_lab_stats: bool = True
    include_shape_stats: bool = True
    include_patch_features: bool = True
    patch_embed_size: int = 4
    mask_patch_background: bool = True
    cache_version: str = "v3"

    def node_feature_dim(self) -> int:
        dim = 0
        if self.include_rgb_stats:
            dim += 6
        if self.include_hsv_stats:
            dim += 6
        if self.include_lab_stats:
            dim += 6
        if self.include_shape_stats:
            dim += 5
        if self.use_xy:
            dim += 2
        if self.include_patch_features:
            dim += 3 * self.patch_embed_size * self.patch_embed_size
        return dim

    @staticmethod
    def edge_feature_dim() -> int:
        return 8


@dataclass
class GraphAugmentConfig:
    node_drop_prob: float = 0.0
    edge_drop_prob: float = 0.0
    feature_mask_prob: float = 0.0
    feature_noise_std: float = 0.0
    edge_noise_std: float = 0.0

    def enabled(self) -> bool:
        return any(
            value > 0
            for value in (
                self.node_drop_prob,
                self.edge_drop_prob,
                self.feature_mask_prob,
                self.feature_noise_std,
                self.edge_noise_std,
            )
        )


try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _BILINEAR = Image.BILINEAR


def _to_pil_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def _accumulate_boundary_counts(lhs: np.ndarray, rhs: np.ndarray, boundary_counts: dict[tuple[int, int], int]) -> None:
    mask = lhs != rhs
    if not np.any(mask):
        return

    src_vals = lhs[mask].astype(np.int64, copy=False)
    dst_vals = rhs[mask].astype(np.int64, copy=False)
    for src, dst in zip(src_vals.tolist(), dst_vals.tolist()):
        if src > dst:
            src, dst = dst, src
        boundary_counts[(src, dst)] = boundary_counts.get((src, dst), 0) + 1


def _normalize_lab(arr: np.ndarray) -> np.ndarray:
    lab = rgb2lab(arr).astype(np.float32, copy=False)
    lab[..., 0] = lab[..., 0] / 100.0
    lab[..., 1] = (lab[..., 1] + 128.0) / 255.0
    lab[..., 2] = (lab[..., 2] + 128.0) / 255.0
    return np.clip(lab, 0.0, 1.0)


def _channel_mean_std(values: np.ndarray, node_idx: np.ndarray, counts: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for ch in range(values.shape[1]):
        channel = values[:, ch]
        sums = np.bincount(node_idx, weights=channel, minlength=num_nodes).astype(np.float32)
        sq_sums = np.bincount(node_idx, weights=channel * channel, minlength=num_nodes).astype(np.float32)
        mean = sums / counts
        var = np.clip((sq_sums / counts) - (mean * mean), a_min=0.0, a_max=None)
        means.append(mean[:, None])
        stds.append(np.sqrt(var)[:, None])
    return np.concatenate(means, axis=1), np.concatenate(stds, axis=1)


def _segment_boxes(seg_map: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = seg_map.shape
    yy, xx = np.indices((h, w), dtype=np.int32)
    node_idx = seg_map.reshape(-1)
    x_flat = xx.reshape(-1)
    y_flat = yy.reshape(-1)

    min_x = np.full(num_nodes, w - 1, dtype=np.int32)
    min_y = np.full(num_nodes, h - 1, dtype=np.int32)
    max_x = np.zeros(num_nodes, dtype=np.int32)
    max_y = np.zeros(num_nodes, dtype=np.int32)

    np.minimum.at(min_x, node_idx, x_flat)
    np.minimum.at(min_y, node_idx, y_flat)
    np.maximum.at(max_x, node_idx, x_flat)
    np.maximum.at(max_y, node_idx, y_flat)
    return min_x, min_y, max_x, max_y


def _extract_patch_features(
    arr: np.ndarray,
    seg_map: np.ndarray,
    min_x: np.ndarray,
    min_y: np.ndarray,
    max_x: np.ndarray,
    max_y: np.ndarray,
    config: GraphBuildConfig,
) -> np.ndarray:
    if not config.include_patch_features:
        return np.zeros((len(min_x), 0), dtype=np.float32)

    patch_feats = []
    patch_size = config.patch_embed_size
    for node_id in range(len(min_x)):
        x0 = int(min_x[node_id])
        y0 = int(min_y[node_id])
        x1 = int(max_x[node_id]) + 1
        y1 = int(max_y[node_id]) + 1

        patch = arr[y0:y1, x0:x1].copy()
        if config.mask_patch_background:
            node_mask = seg_map[y0:y1, x0:x1] == node_id
            patch[~node_mask] = 0.0

        patch_uint8 = np.clip(patch * 255.0, 0.0, 255.0).astype(np.uint8, copy=False)
        patch_img = Image.fromarray(patch_uint8)
        patch_img = patch_img.resize((patch_size, patch_size), resample=_BILINEAR)
        patch_arr = np.asarray(patch_img, dtype=np.float32) / 255.0
        patch_feats.append(patch_arr.reshape(-1))

    return np.stack(patch_feats, axis=0).astype(np.float32, copy=False)


def _build_edge_index_and_attr(
    boundary_counts: dict[tuple[int, int], int],
    rgb_mean: np.ndarray,
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    area_ratio: np.ndarray,
    pixel_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edge_dim = GraphBuildConfig.edge_feature_dim()
    if not boundary_counts:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, edge_dim), dtype=np.float32)

    undirected_edges = np.array(list(boundary_counts.keys()), dtype=np.int64)
    shared_boundaries = np.array(list(boundary_counts.values()), dtype=np.float32)

    src = undirected_edges[:, 0]
    dst = undirected_edges[:, 1]

    color_diff = np.abs(rgb_mean[dst] - rgb_mean[src])
    dx = centroid_x[dst] - centroid_x[src]
    dy = centroid_y[dst] - centroid_y[src]
    dist = np.sqrt((dx * dx) + (dy * dy))
    boundary_ratio = shared_boundaries / np.maximum(np.sqrt(pixel_counts[src] * pixel_counts[dst]), 1.0)
    area_diff = np.abs(area_ratio[dst] - area_ratio[src])

    undirected_attr = np.concatenate(
        [
            color_diff.astype(np.float32, copy=False),
            dx[:, None].astype(np.float32, copy=False),
            dy[:, None].astype(np.float32, copy=False),
            dist[:, None].astype(np.float32, copy=False),
            boundary_ratio[:, None].astype(np.float32, copy=False),
            area_diff[:, None].astype(np.float32, copy=False),
        ],
        axis=1,
    )

    edge_index = np.stack(
        [
            np.concatenate([src, dst], axis=0),
            np.concatenate([dst, src], axis=0),
        ],
        axis=0,
    )
    edge_attr = np.concatenate([undirected_attr, undirected_attr], axis=0)
    return edge_index.astype(np.int64, copy=False), edge_attr.astype(np.float32, copy=False)


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
    node_idx = seg_map.reshape(-1)
    pixel_count = np.bincount(node_idx, minlength=num_nodes).astype(np.float32)
    counts = np.clip(pixel_count, a_min=1.0, a_max=None)

    pixels_rgb = arr.reshape(-1, 3)
    rgb_mean, rgb_std = _channel_mean_std(pixels_rgb, node_idx, counts, num_nodes)

    feature_blocks = []
    if config.include_rgb_stats:
        feature_blocks.extend([rgb_mean, rgb_std])

    if config.include_hsv_stats:
        hsv = rgb2hsv(arr).astype(np.float32, copy=False).reshape(-1, 3)
        hsv_mean, hsv_std = _channel_mean_std(hsv, node_idx, counts, num_nodes)
        feature_blocks.extend([hsv_mean, hsv_std])

    if config.include_lab_stats:
        lab = _normalize_lab(arr).reshape(-1, 3)
        lab_mean, lab_std = _channel_mean_std(lab, node_idx, counts, num_nodes)
        feature_blocks.extend([lab_mean, lab_std])

    yy, xx = np.indices((h, w), dtype=np.float32)
    x_flat = xx.reshape(-1) / max(1.0, float(w - 1))
    y_flat = yy.reshape(-1) / max(1.0, float(h - 1))
    centroid_x = np.bincount(node_idx, weights=x_flat, minlength=num_nodes).astype(np.float32) / counts
    centroid_y = np.bincount(node_idx, weights=y_flat, minlength=num_nodes).astype(np.float32) / counts

    min_x, min_y, max_x, max_y = _segment_boxes(seg_map, num_nodes)
    width_ratio = ((max_x - min_x + 1).astype(np.float32) / max(1.0, float(w)))[:, None]
    height_ratio = ((max_y - min_y + 1).astype(np.float32) / max(1.0, float(h)))[:, None]
    area_ratio = (counts / float(h * w)).astype(np.float32)
    bbox_area_ratio = width_ratio[:, 0] * height_ratio[:, 0]
    fill_ratio = (area_ratio / np.clip(bbox_area_ratio, a_min=1e-6, a_max=None)).astype(np.float32)

    boundary_counts: dict[tuple[int, int], int] = {}
    _accumulate_boundary_counts(seg_map[:, :-1], seg_map[:, 1:], boundary_counts)
    _accumulate_boundary_counts(seg_map[:-1, :], seg_map[1:, :], boundary_counts)

    boundary_total = np.zeros(num_nodes, dtype=np.float32)
    for (src, dst), count in boundary_counts.items():
        boundary_total[src] += count
        boundary_total[dst] += count
    perimeter_ratio = (boundary_total / max(1.0, float(2 * (h + w))))[:, None]

    if config.include_shape_stats:
        feature_blocks.extend(
            [
                area_ratio[:, None],
                width_ratio,
                height_ratio,
                fill_ratio[:, None],
                perimeter_ratio,
            ]
        )

    if config.use_xy:
        feature_blocks.extend([centroid_x[:, None], centroid_y[:, None]])

    if config.include_patch_features:
        patch_feat = _extract_patch_features(arr, seg_map, min_x, min_y, max_x, max_y, config)
        feature_blocks.append(patch_feat)

    node_features = np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)
    edge_index, edge_attr = _build_edge_index_and_attr(
        boundary_counts=boundary_counts,
        rgb_mean=rgb_mean,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        area_ratio=area_ratio,
        pixel_counts=counts,
    )

    return Data(
        x=torch.from_numpy(node_features).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr).float(),
        y=torch.tensor([int(label)], dtype=torch.long),
    )


def _drop_nodes(data: Data, drop_prob: float) -> Data:
    if drop_prob <= 0 or data.x.size(0) <= 2:
        return data

    num_nodes = int(data.x.size(0))
    keep_mask = torch.rand(num_nodes, device=data.x.device) >= drop_prob
    if int(keep_mask.sum().item()) < 2:
        topk = min(2, num_nodes)
        keep_mask[:] = False
        keep_idx = torch.randperm(num_nodes, device=data.x.device)[:topk]
        keep_mask[keep_idx] = True

    if bool(keep_mask.all()):
        return data

    old_to_new = torch.full((num_nodes,), -1, dtype=torch.long, device=data.x.device)
    old_to_new[keep_mask] = torch.arange(int(keep_mask.sum().item()), device=data.x.device)

    edge_keep = keep_mask[data.edge_index[0]] & keep_mask[data.edge_index[1]]
    data.x = data.x[keep_mask]
    data.edge_index = old_to_new[data.edge_index[:, edge_keep]]
    if getattr(data, "edge_attr", None) is not None:
        data.edge_attr = data.edge_attr[edge_keep]
    return data


def _drop_undirected_edges(data: Data, drop_prob: float) -> Data:
    if drop_prob <= 0 or data.edge_index.numel() == 0:
        return data

    keep_mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=data.edge_index.device)
    pair_to_indices: dict[tuple[int, int], list[int]] = {}
    for idx in range(data.edge_index.size(1)):
        src = int(data.edge_index[0, idx].item())
        dst = int(data.edge_index[1, idx].item())
        key = (src, dst) if src < dst else (dst, src)
        pair_to_indices.setdefault(key, []).append(idx)

    if not pair_to_indices:
        return data

    all_groups = list(pair_to_indices.values())
    dropped_any = False
    for indices in all_groups:
        if torch.rand(1).item() < drop_prob:
            keep_mask[indices] = False
            dropped_any = True

    if dropped_any and not bool(keep_mask.any()):
        keep_mask[all_groups[0]] = True

    data.edge_index = data.edge_index[:, keep_mask]
    if getattr(data, "edge_attr", None) is not None:
        data.edge_attr = data.edge_attr[keep_mask]
    return data


def apply_graph_augmentations(data: Data, cfg: GraphAugmentConfig) -> Data:
    if not cfg.enabled():
        return data

    data = _drop_nodes(data, cfg.node_drop_prob)
    data = _drop_undirected_edges(data, cfg.edge_drop_prob)

    if cfg.feature_mask_prob > 0:
        mask = torch.rand_like(data.x) < cfg.feature_mask_prob
        data.x = data.x.masked_fill(mask, 0.0)

    if cfg.feature_noise_std > 0:
        data.x = data.x + torch.randn_like(data.x) * cfg.feature_noise_std

    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is not None and cfg.edge_noise_std > 0 and edge_attr.numel() > 0:
        data.edge_attr = edge_attr + torch.randn_like(edge_attr) * cfg.edge_noise_std

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
        cache_version: str | None = None,
        augment_cfg: GraphAugmentConfig | None = None,
    ):
        self.dataset = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
        self.split_name = split_name
        self.dataset_name = dataset_name
        self.graph_cfg = graph_cfg
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.cache_version = cache_version or graph_cfg.cache_version
        self.augment_cfg = augment_cfg if augment_cfg is not None and augment_cfg.enabled() else None

        self.cache_path = os.path.join(
            cache_dir,
            (
                f"{dataset_name}_{split_name}_img{graph_cfg.image_size}_seg{graph_cfg.n_segments}_"
                f"xy{int(graph_cfg.use_xy)}_compact{str(graph_cfg.compactness).replace('.', 'p')}_"
                f"patch{graph_cfg.patch_embed_size}_{self.cache_version}.pt"
            ),
        )

        self.graphs = None
        self._prepare_graphs()

    def _build_graphs(self) -> list[Data]:
        graphs = []
        iterator = tqdm(self.dataset, desc=f"Build graphs [{self.split_name}]", leave=False)
        for sample in iterator:
            graph = build_superpixel_graph(
                image=sample[self.image_key],
                label=int(sample[self.label_key]),
                config=self.graph_cfg,
            )
            graphs.append(graph)
        return graphs

    def _prepare_graphs(self) -> None:
        def _load_cache():
            try:
                return torch.load(self.cache_path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(self.cache_path, map_location="cpu")

        def _load_cache_or_none():
            if not os.path.exists(self.cache_path):
                return None
            try:
                graphs = _load_cache()
            except Exception as exc:  # noqa: PERF203
                print(f"[Cache] invalid graph cache, rebuilding: {self.cache_path} ({exc})")
                try:
                    os.remove(self.cache_path)
                except OSError:
                    pass
                return None
            return graphs

        if self.use_cache and os.path.exists(self.cache_path):
            cached = _load_cache_or_none()
            if cached is not None:
                self.graphs = cached
                return

        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            lock_path = f"{self.cache_path}.lock"
            lock_acquired = False
            lock_wait_sec = 0.0
            poll_interval_sec = max(0.1, float(os.getenv("GRAPH_CACHE_LOCK_POLL_SEC", "1.0")))
            log_wait_every_sec = max(1, int(os.getenv("GRAPH_CACHE_WAIT_LOG_EVERY_SEC", "60")))
            next_log_wait_sec = float(log_wait_every_sec)
            stale_lock_sec = max(60, int(os.getenv("GRAPH_CACHE_STALE_LOCK_SEC", "7200")))
            wait_timeout_sec = max(0, int(os.getenv("GRAPH_CACHE_LOCK_TIMEOUT_SEC", "180")))
            fallback_to_local = os.getenv("GRAPH_CACHE_FALLBACK_TO_LOCAL", "1").strip().lower() not in {
                "0",
                "false",
                "no",
            }
            build_local_only = False

            while not lock_acquired:
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    lock_acquired = True
                except FileExistsError:
                    cached = _load_cache_or_none()
                    if cached is not None:
                        self.graphs = cached
                        return

                    try:
                        lock_age = time.time() - os.path.getmtime(lock_path)
                    except OSError:
                        lock_age = 0.0
                    if lock_age > stale_lock_sec:
                        print(f"[Cache] removing stale lock: {lock_path}")
                        try:
                            os.remove(lock_path)
                        except OSError:
                            pass
                        continue

                    time.sleep(poll_interval_sec)
                    lock_wait_sec += poll_interval_sec
                    if lock_wait_sec >= next_log_wait_sec:
                        print(
                            f"[Cache] waiting on lock for {self.split_name}: "
                            f"{self.cache_path} waited={int(lock_wait_sec)}s"
                        )
                        next_log_wait_sec += float(log_wait_every_sec)
                    if fallback_to_local and wait_timeout_sec > 0 and lock_wait_sec >= wait_timeout_sec:
                        print(
                            f"[Cache] lock wait exceeded for {self.split_name}: "
                            f"{self.cache_path} waited={int(lock_wait_sec)}s; "
                            "building graphs locally without writing shared cache"
                        )
                        build_local_only = True
                        break

            if build_local_only:
                self.graphs = self._build_graphs()
                return

            try:
                cached = _load_cache_or_none()
                if cached is not None:
                    self.graphs = cached
                    return

                self.graphs = self._build_graphs()
                tmp_cache_path = f"{self.cache_path}.tmp.{os.getpid()}.{time.time_ns()}"
                try:
                    torch.save(self.graphs, tmp_cache_path)
                    os.replace(tmp_cache_path, self.cache_path)
                finally:
                    if os.path.exists(tmp_cache_path):
                        try:
                            os.remove(tmp_cache_path)
                        except OSError:
                            pass
            finally:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
            return

        self.graphs = self._build_graphs()

    def feature_dims(self) -> tuple[int, int]:
        if self.graphs:
            sample = self.graphs[0]
            edge_attr = getattr(sample, "edge_attr", None)
            edge_dim = int(edge_attr.size(-1)) if edge_attr is not None else self.graph_cfg.edge_feature_dim()
            return int(sample.x.size(-1)), edge_dim
        return self.graph_cfg.node_feature_dim(), self.graph_cfg.edge_feature_dim()

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        graph = self.graphs[idx]
        if self.augment_cfg is None or self.split_name != "train":
            return graph
        return apply_graph_augmentations(graph.clone(), self.augment_cfg)
