import os
import subprocess
from pathlib import Path, PurePosixPath
from datetime import datetime

import modal

APP_NAME = "bacon-train"
PROJECT_DIR = "/root/BaCon"
STORAGE_DIR = "/bacon-storage"
DATA_DIR = f"{STORAGE_DIR}/data"
OUTPUT_DIR = f"{STORAGE_DIR}/outputs"
TORCH_CACHE_DIR = f"{STORAGE_DIR}/torch-cache"
IMAGENET_DIR = f"{DATA_DIR}/imagenet"

EXPERIMENTS_DIR = f"{OUTPUT_DIR}/experiments"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "loguru",
        "numpy<2",
        "pandas",
        "scikit-learn",
        "scipy",
        "tqdm",
        "pillow",
        "opencv-python-headless",
        "tensorboard",
        "matplotlib",
        "torch",
        "torchvision",
    )
    .add_local_dir(
        ".", 
        remote_path=PROJECT_DIR, 
        ignore=["cub200_*", "cifar10_*", "cifar100_*", "imagenet100_*", "dev_outputs", "checkpoints", "tensorboard", "__pycache__", "venv", ".venv", ".git", "env"]
    )
)

storage_volume = modal.Volume.from_name("bacon-storage", create_if_missing=True)


def _run(cmd: list[str], cwd: Path = PROJECT_DIR) -> None:
    log_path = Path(OUTPUT_DIR) / "last_train.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUB_ROOT"] = f"{DATA_DIR}/cub/CUB_200_2011"
    env["CARS_ROOT"] = f"{DATA_DIR}/cars"
    env["AIRCRAFT_ROOT"] = f"{DATA_DIR}/aircraft"
    print("$", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
            lines.append(line)
            if len(lines) > 200:
                lines.pop(0)
        return_code = process.wait()
    if return_code != 0:
        print("\nLast 200 training log lines:\n", flush=True)
        print("".join(lines), flush=True)
        raise subprocess.CalledProcessError(return_code, cmd)


def _ensure_link(path: Path, target: Path) -> None:
    if path.exists() or path.is_symlink():
        return
    target.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=True)


def _create_experiment_dirs(experiment_name: str) -> dict:
    """Create experiment directory structure."""
    exp_root = Path(EXPERIMENTS_DIR) / experiment_name
    
    dirs = {
        'root': exp_root,
        'checkpoints': exp_root / "checkpoints",
        'tensorboard': exp_root / "tensorboard",
        'visualizations': exp_root / "visualizations",
        'tsne': exp_root / "visualizations" / "tsne",
        'pca': exp_root / "visualizations" / "pca",
        'confusion_matrix': exp_root / "visualizations" / "confusion_matrix",
        'logs': exp_root / "logs",
        'config': exp_root / "config",
    }
    
    # Create all directories
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def _generate_experiment_name(
    dataset_name: str,
    variant: str = "baseline",
    imb_ratio: int = 100,
    pseudo_mode: int = 0,
    confidence_threshold: float = 0.9,
    extra_suffix: str = "",
    backbone: str = "dinov2_vitb14",
    novel_pseudo: bool = False,
) -> str:
    """Generate a descriptive experiment name."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    if variant == "baseline":
        variant_str = "baseline"
    elif variant == "pseudo":
        if pseudo_mode == 1:
            variant_str = "pseudo_mode1_trainacc"
        elif pseudo_mode == 2:
            variant_str = f"pseudo_mode2_th{confidence_threshold:.2f}"
        elif pseudo_mode == 3:
            variant_str = "pseudo_mode3_bestconf"
        else:
            variant_str = f"pseudo_modeX"
    else:
        variant_str = variant
    
    backbone_tag = {"dino_vitb16": "dinov1b16",
                      "dinov2_vitb14": "dinov2b14",
                      "dinov2_vitb14_reg": "dinov2b14reg"}.get(backbone, backbone)
    if novel_pseudo:
        variant_str = f"{variant_str}+novel"
    parts = [dataset_name, backbone_tag, variant_str, f"imb{imb_ratio}", timestamp]
    
    if extra_suffix:
        parts.insert(-1, extra_suffix)
    
    return "_".join(parts)


@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=60 * 60 * 24,
    volumes={STORAGE_DIR: storage_volume},
)
def train(
    dataset_name: str = "cifar100",
    imb_ratio: int = 100,
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float | None = None,
    gpu_count_workers: int = 8,
    extra_args: list[str] | None = None,
    # Pseudo labeling flags
    enable_pseudo_labeling: bool = False,
    pseudo_mode: int = 1,
    confidence_threshold: float = 0.9,
    pseudo_top_ratio: float = 0.8,
    max_samples_per_class: int = 500,
    pseudo_update_freq: int = 10,
    max_pseudo_iterations: int = 20,
    pseudo_warmup_epoch: int = 30,
    pseudo_conf_bar: float = 0.5,
    pseudo_bar_k: float = 2.0,
    pseudo_min_hi: int = 10,
    # Novel-door pseudo (D2): 1 cong tac duy nhat --enable-novel-pseudo tren CLI
    # (Modal tu doi _ thanh -). Tat = known-only nhu cu.
    enable_novel_pseudo: bool = False,
    novel_warmup_epoch: int = 50,
    novel_update_freq: int = 10,
    max_novel_iterations: int = 2,
    novel_max_samples: int = 100,
    novel_jaccard_th: float = 0.6,
    novel_agree_th: float = 0.7,
    novel_min_size: int = 10,
    # Visualization frequency (0 = off)
    vis_freq: int = 10,
    # AdaPart arguments
    use_parts: bool = False,
    num_slots: int = 3,
    part_lambda: float = 0.5,
    ablate_fused_ce: bool = False,
    ablate_spatial_loss: bool = False,
    ablate_confidence: bool = False,
    ablate_adaptive_capacity: bool = False,
    ablate_concat_eval: bool = False,
    # Backbone: dino_vitb16 (DINOv1) or dinov2_vitb14[_reg] (DINOv2, default)
    backbone: str = "dinov2_vitb14",
    # Train seed: -1 = legacy deterministic (20364); >=0 = explicit repeat seed
    seed: int = -1,
    # A4/A1 (viec 6): opt-in, default tat de khop run cu
    use_logit_adjust: bool = False,
    use_momentum_teacher: bool = False,
    teacher_m0: float = 0.996,
    # Early stopping: dung sau N test-rounds khong best moi (0 = tat)
    early_stop_patience: int = 50,
    # Resume: duong dan .pt tren volume de train tiep ('' = train moi)
    resume_ckpt: str = "",
    # Gioi han gio train (0 = tat) — dung nhe + final report khi het gio
    time_budget_hours: float = 0,
    # Custom experiment name
    exp_name_suffix: str = "",
) -> dict:
    """Train BaCon model with optional pseudo labeling."""

    # Modal chi convert _ -> - o TEN co, VALUE giu nguyen -> user go
    # 'dinov2-vitb14' van chay duoc (chuan hoa ve underscore cua bacon.py).
    backbone = (backbone or "").replace("-", "_")

    project_dir = Path(PROJECT_DIR)
    data_dir = Path(DATA_DIR)
    output_dir = Path(OUTPUT_DIR)
    torch_cache_dir = Path(TORCH_CACHE_DIR)
    imagenet_dir = Path(IMAGENET_DIR)

    os.chdir(project_dir)
    
    # Setup data links
    _ensure_link(project_dir / "cifar10", data_dir / "cifar10")
    _ensure_link(project_dir / "cifar100", data_dir / "cifar100")
    _ensure_link(Path("/root/.cache/torch"), torch_cache_dir)
    _ensure_link(Path("/ImageNet"), imagenet_dir)
    
    # Create experiment directory structure
    variant = "pseudo" if enable_pseudo_labeling else "baseline"
    experiment_name = _generate_experiment_name(
        dataset_name=dataset_name,
        variant=variant,
        imb_ratio=imb_ratio,
        pseudo_mode=pseudo_mode if enable_pseudo_labeling else 0,
        confidence_threshold=confidence_threshold,
        extra_suffix=exp_name_suffix,
        backbone=backbone,
        novel_pseudo=enable_novel_pseudo,
    )
    
    exp_dirs = _create_experiment_dirs(experiment_name)
    
    # Create symlinks for easier access
    _ensure_link(project_dir / "checkpoints", exp_dirs['checkpoints'])
    
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Root: {exp_dirs['root']}")
    print(f"Checkpoints: {exp_dirs['checkpoints']}")
    print(f"TensorBoard: {exp_dirs['tensorboard']}")
    print(f"{'='*60}\n")
    
    # Create symlink for tensorboard
    _ensure_link(project_dir / "tensorboard", exp_dirs['tensorboard'])

    # Determine labeled classes
    if dataset_name == "cifar10":
        labeled_classes = 5
    elif dataset_name == "cub200":
        labeled_classes = 100
        # Ensure CUB directory exists
        cub_dir = Path(f"{DATA_DIR}/cub")
        cub_dir.mkdir(parents=True, exist_ok=True)
    elif dataset_name == "stanford_cars":
        labeled_classes = 98
        Path(f"{DATA_DIR}/cars").mkdir(parents=True, exist_ok=True)
    elif dataset_name == "fgvc_aircraft":
        labeled_classes = 80
        Path(f"{DATA_DIR}/aircraft").mkdir(parents=True, exist_ok=True)
    elif dataset_name == "imagenet100":
        labeled_classes = 50
    else:
        labeled_classes = 80

    # Build command
    command = [
        "python",
        "-m",
        "model.bacon",
        "--dataset-name",
        dataset_name,
        "--labeled-classes",
        str(labeled_classes),
        "--imb-ratio",
        str(imb_ratio),
        "--epochs",
        str(epochs),
        "--stop-epoch",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(gpu_count_workers),
        "--exp-name",
        dataset_name,
        "--warmup-teacher-temp-epochs",
        str(min(50, epochs)),
        "--est-freq",
        "10",
        "--ce-warmup",
        "1",
        "--alpha",
        "0",
        "--beta",
        "0.5",
        "--use-exact-exp-root",
        "--exp-root",
        str(exp_dirs['root']),
    ]

    # Add pseudo labeling arguments
    if enable_pseudo_labeling:
        command.append("--enable-pseudo-labeling")
        command.extend(["--pseudo-mode", str(pseudo_mode)])
        command.extend(["--confidence-threshold", str(confidence_threshold)])
        command.extend(["--pseudo-top-ratio", str(pseudo_top_ratio)])
        command.extend(["--max-samples-per-class", str(max_samples_per_class)])
        command.extend(["--pseudo-update-freq", str(pseudo_update_freq)])
        command.extend(["--max-pseudo-iterations", str(max_pseudo_iterations)])
        command.extend(["--pseudo-warmup-epoch", str(pseudo_warmup_epoch)])
        command.extend(["--pseudo-conf-bar", str(pseudo_conf_bar)])
        command.extend(["--pseudo-bar-k", str(pseudo_bar_k)])
        command.extend(["--pseudo-min-hi", str(pseudo_min_hi)])
        # Novel door (D2): chi truyen khi bat --enable-novel-pseudo
        if enable_novel_pseudo:
            command.append("--enable-novel-pseudo")
            command.extend(["--novel-warmup-epoch", str(novel_warmup_epoch)])
            command.extend(["--novel-update-freq", str(novel_update_freq)])
            command.extend(["--max-novel-iterations", str(max_novel_iterations)])
            command.extend(["--novel-max-samples", str(novel_max_samples)])
            command.extend(["--novel-jaccard-th", str(novel_jaccard_th)])
            command.extend(["--novel-agree-th", str(novel_agree_th)])
            command.extend(["--novel-min-size", str(novel_min_size)])

    # Visualization frequency (PCA / t-SNE / confusion matrix)
    command.extend(["--vis-freq", str(vis_freq)])

    # Backbone (must match model/bacon.py --backbone choices)
    command.extend(["--backbone", backbone])

    # Train seed: -1 = legacy (20364, khop run cu); >=0 = repeat co chu dich
    if seed is not None and int(seed) >= 0:
        command.extend(["--seed", str(int(seed))])

    # Early stopping (default 50 test rounds, 0 = tat)
    command.extend(["--early-stop-patience", str(int(early_stop_patience))])

    # Resume + time budget (chi truyen khi dat)
    if resume_ckpt:
        command.extend(["--resume", str(resume_ckpt)])
    if time_budget_hours and float(time_budget_hours) > 0:
        command.extend(["--time-budget-hours", str(float(time_budget_hours))])

    # A4/A1: chi truyen khi bat (bacon.py default tat)
    if use_logit_adjust:
        command.append("--use-logit-adjust")
    if use_momentum_teacher:
        command.append("--use-momentum-teacher")
        command.extend(["--teacher-m0", str(teacher_m0)])

    # AdaPart arguments
    if use_parts:
        command.append("--use-parts")
        command.extend(["--num-slots", str(num_slots)])
        command.extend(["--part-lambda", str(part_lambda)])
        if ablate_fused_ce:
            command.append("--ablate-fused-ce")
        if ablate_spatial_loss:
            command.append("--ablate-spatial-loss")
        if ablate_confidence:
            command.append("--ablate-confidence")
        if ablate_adaptive_capacity:
            command.append("--ablate-adaptive-capacity")
        if ablate_concat_eval:
            command.append("--ablate-concat-eval")

    # Learning rate
    if lr is not None:
        command += ["--lr", str(lr)]
    elif dataset_name == "cifar10":
        command += ["--lr", "0.01"]

    # Extra arguments
    if extra_args:
        command += extra_args

    _run(command)
    storage_volume.commit()
    
    # Return experiment info
    return {
        'experiment_name': experiment_name,
        'experiment_dir': str(exp_dirs['root']),
        'checkpoints_dir': str(exp_dirs['checkpoints']),
        'tensorboard_dir': str(exp_dirs['tensorboard']),
        'logs_dir': str(exp_dirs['logs']),
        'visualizations_dir': str(exp_dirs['visualizations']),
    }


def _download_visualizations(experiment_name: str) -> None:
    """Download the experiment's visualizations + final report from the Modal volume to ./dev_outputs."""
    import asyncio

    remote_exp_dir = f"{EXPERIMENTS_DIR}/{experiment_name}"
    remote_vis_dir = f"{remote_exp_dir}/visualizations"
    local_dir = Path("dev_outputs") / experiment_name

    async def _fetch() -> int:
        from modal.volume import FileEntryType

        count = 0
        entries = [e async for e in storage_volume.listdir(remote_exp_dir, recursive=True)]
        wanted = [e for e in entries if e.type == FileEntryType.FILE
                  and (str(e.path).endswith(".png")
                       or PurePosixPath(str(e.path)).name.startswith("final_report."))]
        if not wanted:
            print(f"[download] No .png / final_report files found under {remote_exp_dir}.")
            return 0

        for e in wanted:
            entry_path = PurePosixPath(str(e.path))
            # entry.path may be absolute ("/bacon-storage/.../pca/x.png") or
            # volume-relative ("/pca/x.png") — handle both.
            try:
                rel = entry_path.relative_to(remote_exp_dir.lstrip("/"))
            except ValueError:
                rel = entry_path.relative_to("/") if entry_path.is_absolute() else entry_path
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(dest, "wb") as fp:
                    async for chunk in storage_volume.read_file(str(e.path)):
                        fp.write(chunk)
                print(f"[download] {rel}")
                count += 1
            except Exception as err:
                print(f"[download] FAILED {rel}: {err}")
        return count

    try:
        downloaded = asyncio.run(_fetch())
    except Exception as e:
        print(f"[download] Could not access volume ({e}).")
        print("[download] Manual fetch:")
        print(f"  modal volume get bacon-storage \"{remote_exp_dir}\" dev_outputs/")
        return

    if downloaded:
        print(f"\nResults saved to: {local_dir}\n")


def _short_exp_name(experiment_name: str) -> str:
    """Strip trailing _YYYYMMDD_HHMMSS so local exp/ matches repo convention."""
    import re
    return re.sub(r"_\d{8}_\d{6}$", "", experiment_name)


def _download_results(experiment_name: str) -> None:
    """Download logs/final_report/test_history/pseudo_events to local exp/<short>/.

    Works from ANY machine/workspace sharing the Modal account: results live on
    the shared volume, not in the launching terminal. Checkpoints (*.pt, ~800MB)
    are SKIPPED — fetch manually:
      modal volume get bacon-storage "outputs/experiments/<exp>/checkpoints/model_epoch<N>.pt" ckpts/
    """
    import asyncio

    remote_exp_dir = f"{EXPERIMENTS_DIR}/{experiment_name}"
    local_dir = Path("exp") / _short_exp_name(experiment_name)
    keep_names = {"final_report.csv", "log.txt"}

    async def _fetch() -> int:
        from modal.volume import FileEntryType

        count = 0
        entries = [e async for e in storage_volume.listdir(remote_exp_dir, recursive=True)]
        wanted = [e for e in entries if e.type == FileEntryType.FILE
                  and PurePosixPath(str(e.path)).name in keep_names
                  and "checkpoints" not in str(e.path)]
        if not wanted:
            print(f"[results] Nothing found under {remote_exp_dir}.")
            return 0
        for e in wanted:
            entry_path = PurePosixPath(str(e.path))
            try:
                rel = entry_path.relative_to(remote_exp_dir.lstrip("/"))
            except ValueError:
                rel = entry_path.relative_to("/") if entry_path.is_absolute() else entry_path
            # Flatten logs/ into exp root (repo convention: exp/<run>/log.txt).
            name = entry_path.name
            dest = local_dir / name
            if dest.exists():
                print(f"[results] skip (exists): {name}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(dest, "wb") as fp:
                    async for chunk in storage_volume.read_file(str(e.path)):
                        fp.write(chunk)
                print(f"[results] {name}")
                count += 1
            except Exception as err:
                print(f"[results] FAILED {name}: {err}")
        return count

    try:
        got = asyncio.run(_fetch())
    except Exception as e:
        print(f"[results] Could not access volume ({e}). Manual fetch:")
        print(f"  modal volume get bacon-storage \"{remote_exp_dir}\" \"exp/{_short_exp_name(experiment_name)}\"")
        return

    if got:
        print(f"\nResults saved to: {local_dir}\n")


@app.local_entrypoint()
def main(
    dataset_name: str = "cifar100",
    imb_ratio: int = 100,
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float | None = None,
    num_workers: int = 8,
    extra_args: str = "",
    # Pseudo labeling flags
    enable_pseudo_labeling: bool = False,
    pseudo_mode: int = 1,
    confidence_threshold: float = 0.9,
    pseudo_top_ratio: float = 0.8,
    max_samples_per_class: int = 500,
    pseudo_update_freq: int = 10,
    max_pseudo_iterations: int = 20,
    pseudo_warmup_epoch: int = 30,
    pseudo_conf_bar: float = 0.5,
    pseudo_bar_k: float = 2.0,
    pseudo_min_hi: int = 10,
    # Novel-door pseudo (D2): 1 cong tac duy nhat
    enable_novel_pseudo: bool = False,
    novel_warmup_epoch: int = 50,
    novel_update_freq: int = 10,
    max_novel_iterations: int = 2,
    novel_max_samples: int = 100,
    novel_jaccard_th: float = 0.6,
    novel_agree_th: float = 0.7,
    novel_min_size: int = 10,
    # AdaPart flags
    use_parts: bool = False,
    num_slots: int = 3,
    part_lambda: float = 0.5,
    ablate_fused_ce: bool = False,
    ablate_spatial_loss: bool = False,
    ablate_confidence: bool = False,
    ablate_adaptive_capacity: bool = False,
    ablate_concat_eval: bool = False,
    backbone: str = "dinov2_vitb14",
    seed: int = -1,
    use_logit_adjust: bool = False,
    use_momentum_teacher: bool = False,
    teacher_m0: float = 0.996,
    early_stop_patience: int = 50,
    # Custom experiment name
    exp_name_suffix: str = "",
    # Visualization frequency (0 = off)
    vis_freq: int = 10,
    # Download visualizations to local machine after training
    download_vis: bool = True,
    # Download results (logs/final_report/test_history) after training
    download_results: bool = True,
) -> None:
    """Local entry point for training."""
    parsed_extra_args = extra_args.split() if extra_args else []

    result = train.remote(
        dataset_name=dataset_name,
        imb_ratio=imb_ratio,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        gpu_count_workers=num_workers,
        extra_args=parsed_extra_args,
        # Pseudo labeling flags
        enable_pseudo_labeling=enable_pseudo_labeling,
        pseudo_mode=pseudo_mode,
        confidence_threshold=confidence_threshold,
        pseudo_top_ratio=pseudo_top_ratio,
        max_samples_per_class=max_samples_per_class,
        pseudo_update_freq=pseudo_update_freq,
        max_pseudo_iterations=max_pseudo_iterations,
        pseudo_warmup_epoch=pseudo_warmup_epoch,
        pseudo_conf_bar=pseudo_conf_bar,
        pseudo_bar_k=pseudo_bar_k,
        pseudo_min_hi=pseudo_min_hi,
        enable_novel_pseudo=enable_novel_pseudo,
        novel_warmup_epoch=novel_warmup_epoch,
        novel_update_freq=novel_update_freq,
        max_novel_iterations=max_novel_iterations,
        novel_max_samples=novel_max_samples,
        novel_jaccard_th=novel_jaccard_th,
        novel_agree_th=novel_agree_th,
        novel_min_size=novel_min_size,
        # AdaPart flags
        use_parts=use_parts,
        num_slots=num_slots,
        part_lambda=part_lambda,
        ablate_fused_ce=ablate_fused_ce,
        ablate_confidence=ablate_confidence,
        ablate_adaptive_capacity=ablate_adaptive_capacity,
        ablate_concat_eval=ablate_concat_eval,
        backbone=backbone,
        seed=seed,
        use_logit_adjust=use_logit_adjust,
        use_momentum_teacher=use_momentum_teacher,
        teacher_m0=teacher_m0,
        early_stop_patience=early_stop_patience,
        resume_ckpt=resume_ckpt,
        time_budget_hours=time_budget_hours,
        # Visualization frequency (PCA / t-SNE / confusion matrix)
        vis_freq=vis_freq,
        # Custom experiment name
        exp_name_suffix=exp_name_suffix,
    )

    if download_results:
        _download_results(result['experiment_name'])

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Experiment: {result['experiment_name']}")
    print(f"Dir: {result['experiment_dir']}")
    print(f"Checkpoints: {result['checkpoints_dir']}")
    print(f"TensorBoard: {result['tensorboard_dir']}")
    print(f"{'='*60}")

    if download_vis:
        _download_visualizations(result['experiment_name'])



@app.local_entrypoint()
def launch(
    dataset_name: str = "cifar100",
    imb_ratio: int = 100,
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float | None = None,
    num_workers: int = 8,
    extra_args: str = "",
    # Pseudo labeling flags
    enable_pseudo_labeling: bool = False,
    pseudo_mode: int = 1,
    confidence_threshold: float = 0.9,
    pseudo_top_ratio: float = 0.8,
    max_samples_per_class: int = 500,
    pseudo_update_freq: int = 10,
    max_pseudo_iterations: int = 20,
    pseudo_warmup_epoch: int = 30,
    pseudo_conf_bar: float = 0.5,
    pseudo_bar_k: float = 2.0,
    pseudo_min_hi: int = 10,
    # Novel-door pseudo (D2): 1 cong tac duy nhat
    enable_novel_pseudo: bool = False,
    novel_warmup_epoch: int = 50,
    novel_update_freq: int = 10,
    max_novel_iterations: int = 2,
    novel_max_samples: int = 100,
    novel_jaccard_th: float = 0.6,
    novel_agree_th: float = 0.7,
    novel_min_size: int = 10,
    # AdaPart flags
    use_parts: bool = False,
    num_slots: int = 3,
    part_lambda: float = 0.5,
    ablate_fused_ce: bool = False,
    ablate_spatial_loss: bool = False,
    ablate_confidence: bool = False,
    ablate_adaptive_capacity: bool = False,
    ablate_concat_eval: bool = False,
    backbone: str = "dinov2_vitb14",
    seed: int = -1,
    use_logit_adjust: bool = False,
    use_momentum_teacher: bool = False,
    teacher_m0: float = 0.996,
    early_stop_patience: int = 50,
    resume_ckpt: str = "",
    time_budget_hours: float = 0,
    # Custom experiment name
    exp_name_suffix: str = "",
    # Visualization frequency (0 = off)
    vis_freq: int = 10,
) -> None:
    """Fire-and-forget launch: spawns the run detached on Modal and exits.

    IMPORTANT: invoke with the --detach flag so Modal keeps the app running
    after this local process exits (without it the app is torn down as soon
    as the entrypoint returns):

        modal run -d modal_train.py::launch ...

    Follow progress with:
        modal app ls (to see the running app ID)
        modal logs <app_id>
    """
    parsed_extra_args = extra_args.split() if extra_args else []

    call = train.spawn(
        dataset_name=dataset_name,
        imb_ratio=imb_ratio,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        gpu_count_workers=num_workers,
        extra_args=parsed_extra_args,
        enable_pseudo_labeling=enable_pseudo_labeling,
        pseudo_mode=pseudo_mode,
        confidence_threshold=confidence_threshold,
        pseudo_top_ratio=pseudo_top_ratio,
        max_samples_per_class=max_samples_per_class,
        pseudo_update_freq=pseudo_update_freq,
        max_pseudo_iterations=max_pseudo_iterations,
        pseudo_warmup_epoch=pseudo_warmup_epoch,
        pseudo_conf_bar=pseudo_conf_bar,
        pseudo_bar_k=pseudo_bar_k,
        pseudo_min_hi=pseudo_min_hi,
        enable_novel_pseudo=enable_novel_pseudo,
        novel_warmup_epoch=novel_warmup_epoch,
        novel_update_freq=novel_update_freq,
        max_novel_iterations=max_novel_iterations,
        novel_max_samples=novel_max_samples,
        novel_jaccard_th=novel_jaccard_th,
        novel_agree_th=novel_agree_th,
        novel_min_size=novel_min_size,
        use_parts=use_parts,
        num_slots=num_slots,
        part_lambda=part_lambda,
        ablate_fused_ce=ablate_fused_ce,
        ablate_confidence=ablate_confidence,
        ablate_adaptive_capacity=ablate_adaptive_capacity,
        ablate_concat_eval=ablate_concat_eval,
        backbone=backbone,
        seed=seed,
        use_logit_adjust=use_logit_adjust,
        use_momentum_teacher=use_momentum_teacher,
        teacher_m0=teacher_m0,
        early_stop_patience=early_stop_patience,
        resume_ckpt=resume_ckpt,
        time_budget_hours=time_budget_hours,
        vis_freq=vis_freq,
        exp_name_suffix=exp_name_suffix,
    )
    print("\n" + "=" * 60)
    print(f"Spawned detached Modal run (call id: {call.object_id})")
    print("It keeps running when you close this terminal / shut down.")
    print("=" * 60)
    print("Follow logs:   modal app logs bacon-train")
    print("List apps:     modal app list")
    print("Stop manually: modal app stop bacon-train")
    print()
    print("FETCH RESULTS (from THIS or ANY other machine/ws on your account,")
    print("after the run finishes — volume is shared, terminal is not):")
    print("  modal volume ls bacon-storage outputs/experiments | tail -5")
    print("  modal run modal_train.py::fetch --exp-name <full_exp_dir_name>")
    print("  (downloads final_report/log/test_history into local exp/<short>/)")
    print()
    print("NOTE: if you did not pass --detach (-d) to 'modal run', this app")
    print("was already torn down when the entrypoint returned. Re-run with:")
    print("  modal run -d modal_train.py::launch ...")


@app.local_entrypoint()
def fetch(exp_name: str = "") -> None:
    """Download results of a FINISHED run by full experiment dir name.

    Works from any workspace/machine on the same Modal account (ws1 -> ws2 OK):
    results live on the shared volume, not in the launching terminal.

        modal volume ls bacon-storage outputs/experiments | tail -5
        modal run modal_train.py::fetch --exp-name <full_name>
    """
    if not exp_name:
        print("Usage: modal run modal_train.py::fetch --exp-name <full_exp_dir_name>")
        print("Find names with: modal volume ls bacon-storage outputs/experiments")
        return
    _download_results(exp_name)
    _download_visualizations(exp_name)


if __name__ == "__main__":
    main()
