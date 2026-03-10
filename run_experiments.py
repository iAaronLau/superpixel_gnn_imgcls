#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GRAPH_MODELS = {"gcn", "gat", "graph_transformer"}
IMAGENET_STYLE_DATASETS = {"imagewoof", "imagenette2"}
DOWNSTREAM_GROUP_IDS = {
    "imagewoof": ("9a", "9b"),
    "imagenette2": ("10a", "10b"),
}
FINAL_RE = re.compile(
    r"best_val_acc=([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\s+test_acc=([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"
)
OOM_PATTERNS = (
    "out of memory",
    "cuda error: out of memory",
    "cudnn_status_alloc_failed",
)


@dataclass
class ExperimentSpec:
    group_id: str
    dataset: str
    model: str
    image_size: int
    epochs: int
    n_segments: int | None = None
    use_xy: int | None = None
    resnet_name: str = "resnet18"
    note: str = ""


@dataclass
class RunRecord:
    group_id: str
    run_name: str
    status: str
    attempts: int
    lr: float
    batch_size: int
    weight_decay: float
    dataset: str
    model: str
    n_segments: str
    use_xy: str
    val_acc: str
    test_acc: str
    elapsed_sec: float
    log_file: str
    note: str


def parse_float_grid(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Empty float grid")
    return values


def parse_int_grid(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Empty int grid")
    return values


def parse_str_grid(raw: str) -> list[str]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(token)
    if not values:
        raise ValueError("Empty str grid")
    return values


def append_sweep_row(csv_path: Path, row: RunRecord) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fields = [
        "group_id",
        "run_name",
        "status",
        "attempts",
        "lr",
        "batch_size",
        "weight_decay",
        "dataset",
        "model",
        "n_segments",
        "use_xy",
        "val_acc",
        "test_acc",
        "elapsed_sec",
        "log_file",
        "note",
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: getattr(row, k) for k in fields})


def read_last_results_row(results_csv: Path) -> dict | None:
    if not results_csv.exists():
        return None
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[-1]


def has_completed_results(results_csv: Path) -> bool:
    if not results_csv.exists():
        return False
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            val = str(row.get("val_acc", "")).strip()
            test = str(row.get("test_acc", "")).strip()
            if val and test:
                return True
    return False


def has_success_in_sweep_log(sweep_log_csv: Path, run_name: str) -> bool:
    if not sweep_log_csv.exists():
        return False
    with sweep_log_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("run_name") == run_name and row.get("status") == "success":
                return True
    return False


def should_skip_batch(spec: ExperimentSpec, batch_size: int, args) -> tuple[bool, str]:
    if spec.model not in GRAPH_MODELS:
        return False, ""

    if spec.dataset == "cifar10" and batch_size > args.max_graph_batch_cifar:
        return True, f"graph model on CIFAR-10: batch_size>{args.max_graph_batch_cifar}"

    if spec.dataset in IMAGENET_STYLE_DATASETS:
        limit = max_graph_batch_for_dataset(args, spec.dataset)
        if batch_size > limit:
            return True, f"graph model on {spec.dataset}: batch_size>{limit}"

    return False, ""


def make_run_name(args, spec: ExperimentSpec, lr: float, batch_size: int, weight_decay: float) -> str:
    def _s(x):
        return str(x).replace(".", "p")

    suffix = []
    if spec.model in GRAPH_MODELS:
        suffix.append(f"seg{spec.n_segments}")
        suffix.append(f"xy{spec.use_xy}")
    if spec.model == "resnet":
        suffix.append(spec.resnet_name)

    suffix_str = "_".join(suffix)
    if suffix_str:
        suffix_str = f"_{suffix_str}"

    return (
        f"{args.sweep_name}_g{spec.group_id}_{spec.dataset}_{spec.model}{suffix_str}"
        f"_lr{_s(lr)}_bs{batch_size}_wd{_s(weight_decay)}_seed{args.seed}"
    )


def build_command(args, spec: ExperimentSpec, lr: float, batch_size: int, weight_decay: float, run_name: str) -> list[str]:
    cmd = [
        args.python,
        str(Path(args.project_root) / "train.py"),
        "--dataset",
        spec.dataset,
        "--model",
        spec.model,
        "--train_backend",
        args.train_backend,
        "--image_size",
        str(spec.image_size),
        "--batch_size",
        str(batch_size),
        "--epochs",
        str(spec.epochs),
        "--lr",
        str(lr),
        "--weight_decay",
        str(weight_decay),
        "--seed",
        str(args.seed),
        "--mixed_precision",
        args.mixed_precision,
        "--num_workers",
        str(args.num_workers),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--scheduler",
        args.scheduler,
        "--output_dir",
        args.output_dir,
        "--run_name",
        run_name,
        "--use_wandb",
        str(args.use_wandb),
        "--wandb_project",
        args.wandb_project,
        "--wandb_mode",
        args.wandb_mode,
        "--eval_strategy",
        args.eval_strategy,
        "--save_strategy",
        args.save_strategy,
        "--checkpoints_total_limit",
        str(args.checkpoints_total_limit),
    ]

    if args.eval_steps > 0:
        cmd.extend(["--eval_steps", str(args.eval_steps)])
    if args.save_steps > 0:
        cmd.extend(["--save_steps", str(args.save_steps)])

    if spec.model in GRAPH_MODELS:
        cmd.extend(
            [
                "--n_segments",
                str(spec.n_segments),
                "--use_xy",
                str(spec.use_xy),
                "--use_cache",
                str(args.use_cache),
                "--cache_dir",
                args.cache_dir,
            ]
        )
    else:
        cmd.extend(["--resnet_name", spec.resnet_name])

    if args.extra_train_args.strip():
        cmd.extend(shlex.split(args.extra_train_args.strip()))

    return cmd


def run_command(cmd: list[str], log_path: Path, dry_run: bool) -> tuple[int, float, str, str, bool]:
    if dry_run:
        return 0, 0.0, "", "", False

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    tail = deque(maxlen=400)
    final_line = ""
    oom = False

    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=Path(__file__).resolve().parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            tail.append(line.rstrip("\n"))
            if "[Final]" in line:
                final_line = line.strip()
            lower = line.lower()
            if any(p in lower for p in OOM_PATTERNS):
                oom = True
        return_code = proc.wait()

    elapsed = time.time() - start
    tail_text = "\n".join(tail)
    return return_code, elapsed, tail_text, final_line, oom


def parse_final_metrics(final_line: str) -> tuple[str, str]:
    if not final_line:
        return "", ""
    match = FINAL_RE.search(final_line)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def build_cifar_specs(args) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(group_id="1", dataset="cifar10", model="resnet", image_size=64, epochs=args.cifar_epochs, resnet_name="resnet18", note="CIFAR baseline"),
        ExperimentSpec(group_id="2", dataset="cifar10", model="gcn", image_size=64, epochs=args.cifar_epochs, n_segments=100, use_xy=0),
        ExperimentSpec(group_id="3", dataset="cifar10", model="gat", image_size=64, epochs=args.cifar_epochs, n_segments=100, use_xy=0),
        ExperimentSpec(group_id="4", dataset="cifar10", model="gcn", image_size=64, epochs=args.cifar_epochs, n_segments=50, use_xy=0),
        ExperimentSpec(group_id="5", dataset="cifar10", model="gcn", image_size=64, epochs=args.cifar_epochs, n_segments=200, use_xy=0),
        ExperimentSpec(group_id="6", dataset="cifar10", model="gcn", image_size=64, epochs=args.cifar_epochs, n_segments=100, use_xy=1),
        ExperimentSpec(group_id="7", dataset="cifar10", model="gat", image_size=64, epochs=args.cifar_epochs, n_segments=100, use_xy=1),
        ExperimentSpec(group_id="8", dataset="cifar10", model="resnet", image_size=64, epochs=args.cifar_epochs, resnet_name="resnet34", note="CIFAR baseline variant"),
    ]


def epochs_for_dataset(args, dataset_name: str) -> int:
    if dataset_name == "cifar10":
        return args.cifar_epochs
    if dataset_name == "imagewoof":
        return args.imagewoof_epochs
    if dataset_name == "imagenette2":
        return args.imagenette2_epochs
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def max_graph_batch_for_dataset(args, dataset_name: str) -> int:
    if dataset_name == "cifar10":
        return args.max_graph_batch_cifar
    if dataset_name == "imagewoof":
        return args.max_graph_batch_imagewoof
    if dataset_name == "imagenette2":
        return args.max_graph_batch_imagenette2
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_downstream_specs(args, dataset_name: str, best_graph_model: str, best_graph_use_xy: int) -> list[ExperimentSpec]:
    if dataset_name not in DOWNSTREAM_GROUP_IDS:
        raise ValueError(f"Unsupported downstream dataset: {dataset_name}")

    resnet_group_id, graph_group_id = DOWNSTREAM_GROUP_IDS[dataset_name]
    dataset_note = dataset_name.capitalize()
    return [
        ExperimentSpec(
            group_id=resnet_group_id,
            dataset=dataset_name,
            model="resnet",
            image_size=224,
            epochs=epochs_for_dataset(args, dataset_name),
            resnet_name="resnet18",
            note=f"{dataset_note} baseline",
        ),
        ExperimentSpec(
            group_id=graph_group_id,
            dataset=dataset_name,
            model=best_graph_model,
            image_size=224,
            epochs=epochs_for_dataset(args, dataset_name),
            n_segments=100,
            use_xy=best_graph_use_xy,
            note=f"{dataset_note} with best CIFAR graph model",
        ),
    ]


def run_grid_for_spec(
    args,
    spec: ExperimentSpec,
    sweep_log_csv: Path,
    results_csv: Path,
    graph_candidates: list[dict],
    counters: dict,
) -> bool:
    stop_early = False

    for lr in args.lr_values:
        for batch_size in args.batch_sizes:
            for weight_decay in args.weight_decays:
                if args.max_runs > 0 and counters["launched"] >= args.max_runs:
                    print(f"[Sweep] Reached --max_runs={args.max_runs}. Stop scheduling.")
                    return True

                skip, reason = should_skip_batch(spec, batch_size, args)
                run_name = make_run_name(args, spec, lr, batch_size, weight_decay)
                run_dir = Path(args.output_dir) / run_name

                if skip:
                    rec = RunRecord(
                        group_id=spec.group_id,
                        run_name=run_name,
                        status="skipped",
                        attempts=0,
                        lr=lr,
                        batch_size=batch_size,
                        weight_decay=weight_decay,
                        dataset=spec.dataset,
                        model=spec.model,
                        n_segments="-" if spec.n_segments is None else str(spec.n_segments),
                        use_xy="-" if spec.use_xy is None else str(spec.use_xy),
                        val_acc="",
                        test_acc="",
                        elapsed_sec=0.0,
                        log_file="",
                        note=reason,
                    )
                    append_sweep_row(sweep_log_csv, rec)
                    print(f"[Skip] {run_name}: {reason}")
                    continue

                if args.skip_completed and (
                    has_success_in_sweep_log(sweep_log_csv, run_name)
                    or has_completed_results(Path(args.output_dir) / run_name / "results.csv")
                ):
                    rec = RunRecord(
                        group_id=spec.group_id,
                        run_name=run_name,
                        status="skipped_completed",
                        attempts=0,
                        lr=lr,
                        batch_size=batch_size,
                        weight_decay=weight_decay,
                        dataset=spec.dataset,
                        model=spec.model,
                        n_segments="-" if spec.n_segments is None else str(spec.n_segments),
                        use_xy="-" if spec.use_xy is None else str(spec.use_xy),
                        val_acc="",
                        test_acc="",
                        elapsed_sec=0.0,
                        log_file="",
                        note="results.csv already present",
                    )
                    append_sweep_row(sweep_log_csv, rec)
                    print(f"[Skip] completed run: {run_name}")
                    continue

                if args.skip_existing and run_dir.exists() and (run_dir / "config.json").exists():
                    rec = RunRecord(
                        group_id=spec.group_id,
                        run_name=run_name,
                        status="skipped_existing",
                        attempts=0,
                        lr=lr,
                        batch_size=batch_size,
                        weight_decay=weight_decay,
                        dataset=spec.dataset,
                        model=spec.model,
                        n_segments="-" if spec.n_segments is None else str(spec.n_segments),
                        use_xy="-" if spec.use_xy is None else str(spec.use_xy),
                        val_acc="",
                        test_acc="",
                        elapsed_sec=0.0,
                        log_file="",
                        note="run directory already exists",
                    )
                    append_sweep_row(sweep_log_csv, rec)
                    print(f"[Skip] existing run_dir: {run_dir}")
                    continue

                cmd = build_command(args, spec, lr, batch_size, weight_decay, run_name)
                cmd_str = shlex.join(cmd)

                if args.dry_run:
                    print(f"[DryRun] {spec.group_id} :: {cmd_str}")
                    rec = RunRecord(
                        group_id=spec.group_id,
                        run_name=run_name,
                        status="dry_run",
                        attempts=0,
                        lr=lr,
                        batch_size=batch_size,
                        weight_decay=weight_decay,
                        dataset=spec.dataset,
                        model=spec.model,
                        n_segments="-" if spec.n_segments is None else str(spec.n_segments),
                        use_xy="-" if spec.use_xy is None else str(spec.use_xy),
                        val_acc="",
                        test_acc="",
                        elapsed_sec=0.0,
                        log_file="",
                        note=spec.note,
                    )
                    append_sweep_row(sweep_log_csv, rec)
                    counters["launched"] += 1
                    continue

                print(f"\n[Run] {spec.group_id} :: {run_name}")
                print(f"[Cmd] {cmd_str}")

                attempts = 0
                success = False
                val_acc = ""
                test_acc = ""
                elapsed = 0.0
                final_note = spec.note
                log_file = str(Path(args.output_dir) / "sweep_logs" / f"{run_name}.log")

                while attempts < max(1, args.retries + 1):
                    attempts += 1
                    log_path = Path(log_file)
                    rc, elapsed, tail_text, final_line, oom = run_command(cmd, log_path, dry_run=False)

                    val_acc, test_acc = parse_final_metrics(final_line)
                    if not val_acc or not test_acc:
                        last_row = read_last_results_row(results_csv)
                        if last_row is not None:
                            val_acc = val_acc or str(last_row.get("val_acc", ""))
                            test_acc = test_acc or str(last_row.get("test_acc", ""))

                    if rc == 0:
                        success = True
                        break

                    final_note = f"attempt={attempts} failed rc={rc}"
                    if oom:
                        final_note += " (OOM)"
                    print(f"[Fail] {run_name} attempt={attempts} rc={rc}")

                    if attempts < max(1, args.retries + 1):
                        print(f"[Retry] sleep {args.retry_sleep_sec}s")
                        time.sleep(args.retry_sleep_sec)
                    else:
                        tail_preview = "\n".join(tail_text.splitlines()[-20:])
                        print(f"[Tail]\n{tail_preview}")

                counters["launched"] += 1
                if success:
                    counters["success"] += 1
                else:
                    counters["failed"] += 1

                rec = RunRecord(
                    group_id=spec.group_id,
                    run_name=run_name,
                    status="success" if success else "failed",
                    attempts=attempts,
                    lr=lr,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    dataset=spec.dataset,
                    model=spec.model,
                    n_segments="-" if spec.n_segments is None else str(spec.n_segments),
                    use_xy="-" if spec.use_xy is None else str(spec.use_xy),
                    val_acc=val_acc,
                    test_acc=test_acc,
                    elapsed_sec=elapsed,
                    log_file=log_file,
                    note=final_note,
                )
                append_sweep_row(sweep_log_csv, rec)

                if success and spec.dataset == "cifar10" and spec.model in {"gcn", "gat"} and spec.n_segments == 100:
                    try:
                        va = float(val_acc)
                    except Exception:  # noqa: PERF203
                        va = float("nan")
                    if va == va:  # not nan
                        graph_candidates.append(
                            {
                                "model": spec.model,
                                "use_xy": int(spec.use_xy) if spec.use_xy is not None else 0,
                                "val_acc": va,
                                "run_name": run_name,
                            }
                        )

    return stop_early


def choose_best_graph_candidate(args, graph_candidates: list[dict]) -> tuple[str, int]:
    if not graph_candidates:
        print(
            "[Sweep] No completed CIFAR graph candidates found; "
            f"fallback to model={args.default_graph_model}, use_xy={args.default_graph_use_xy}"
        )
        return args.default_graph_model, args.default_graph_use_xy

    best = max(graph_candidates, key=lambda x: x["val_acc"])
    print(
        "[Sweep] Best CIFAR graph candidate: "
        f"model={best['model']} use_xy={best['use_xy']} val_acc={best['val_acc']:.6f} run={best['run_name']}"
    )
    return best["model"], int(best["use_xy"])


def plan_total_runs(specs: Iterable[ExperimentSpec], args) -> int:
    total = 0
    for spec in specs:
        for _ in args.lr_values:
            for bs in args.batch_sizes:
                for _ in args.weight_decays:
                    skip, _ = should_skip_batch(spec, bs, args)
                    if not skip:
                        total += 1
    return total


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch experiment/sweep runner for superpixel_gnn_imgcls")

    p.add_argument("--project_root", type=str, default=str(Path(__file__).resolve().parent))
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--results_csv", type=str, default="outputs/results.csv")
    p.add_argument("--sweep_log_csv", type=str, default="outputs/sweep_runs.csv")
    p.add_argument("--sweep_name", type=str, default="batch9")

    p.add_argument("--train_backend", type=str, default="transformers", choices=["transformers", "accelerate"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step"])

    p.add_argument("--cifar_epochs", type=int, default=50)
    p.add_argument("--imagewoof_epochs", type=int, default=20)
    p.add_argument("--imagenette2_epochs", type=int, default=-1)

    p.add_argument("--lr_grid", type=str, default="3e-4,1e-3")
    p.add_argument("--batch_size_grid", type=str, default="16,32,64")
    p.add_argument("--weight_decay_grid", type=str, default="1e-4")

    p.add_argument("--max_graph_batch_cifar", type=int, default=128)
    p.add_argument("--max_graph_batch_imagewoof", type=int, default=128)
    p.add_argument("--max_graph_batch_imagenette2", type=int, default=-1)
    p.add_argument("--downstream_datasets", type=str, default="imagewoof")

    p.add_argument("--use_cache", type=int, default=1)
    p.add_argument("--cache_dir", type=str, default="graph_cache")

    p.add_argument("--use_wandb", type=int, default=1)
    p.add_argument("--wandb_project", type=str, default="superpixel-gnn-imgcls")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    p.add_argument("--eval_strategy", type=str, default="epoch", choices=["epoch", "steps"])
    p.add_argument("--save_strategy", type=str, default="auto", choices=["auto", "epoch", "steps"])
    p.add_argument("--eval_steps", type=int, default=0)
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--checkpoints_total_limit", type=int, default=3)

    p.add_argument("--skip_completed", type=int, default=1)
    p.add_argument("--skip_existing", type=int, default=0)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--retry_sleep_sec", type=int, default=15)
    p.add_argument("--max_runs", type=int, default=-1)
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--default_graph_model", type=str, default="gcn", choices=["gcn", "gat"])
    p.add_argument("--default_graph_use_xy", type=int, default=1, choices=[0, 1])

    p.add_argument("--extra_train_args", type=str, default="")

    return p


def main() -> None:
    args = build_parser().parse_args()

    args.project_root = str(Path(args.project_root).resolve())
    args.lr_values = parse_float_grid(args.lr_grid)
    args.batch_sizes = parse_int_grid(args.batch_size_grid)
    args.weight_decays = parse_float_grid(args.weight_decay_grid)
    args.downstream_datasets_list = parse_str_grid(args.downstream_datasets)

    if args.imagenette2_epochs <= 0:
        args.imagenette2_epochs = args.imagewoof_epochs
    if args.max_graph_batch_imagenette2 <= 0:
        args.max_graph_batch_imagenette2 = args.max_graph_batch_imagewoof

    unsupported = [name for name in args.downstream_datasets_list if name not in IMAGENET_STYLE_DATASETS]
    if unsupported:
        raise ValueError(
            f"Unsupported downstream dataset(s): {unsupported}. "
            f"Choices: {sorted(IMAGENET_STYLE_DATASETS)}"
        )

    results_csv = Path(args.results_csv)
    sweep_log_csv = Path(args.sweep_log_csv)

    cifar_specs = build_cifar_specs(args)
    planned_cifar = plan_total_runs(cifar_specs, args)
    print(f"[Sweep] Planned CIFAR runs (before max_runs): {planned_cifar}")

    counters = {"launched": 0, "success": 0, "failed": 0}
    graph_candidates: list[dict] = []

    for spec in cifar_specs:
        stop = run_grid_for_spec(args, spec, sweep_log_csv, results_csv, graph_candidates, counters)
        if stop:
            break

    best_graph_model, best_graph_use_xy = choose_best_graph_candidate(args, graph_candidates)

    for dataset_name in args.downstream_datasets_list:
        downstream_specs = build_downstream_specs(args, dataset_name, best_graph_model, best_graph_use_xy)
        planned_downstream = plan_total_runs(downstream_specs, args)
        print(f"[Sweep] Planned {dataset_name} runs (before max_runs): {planned_downstream}")

        for spec in downstream_specs:
            stop = run_grid_for_spec(args, spec, sweep_log_csv, results_csv, graph_candidates, counters)
            if stop:
                break
        if stop:
            break

    print(
        f"[Sweep Done] launched={counters['launched']} success={counters['success']} "
        f"failed={counters['failed']} dry_run={args.dry_run}"
    )
    print(f"[Sweep Done] logs: {sweep_log_csv}")
    print(f"[Sweep Done] training results: {results_csv}")


if __name__ == "__main__":
    main()
