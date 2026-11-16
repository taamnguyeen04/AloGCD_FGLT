"""Kaggle training runner for ProtoTail (SimGCD & BaCon baselines) on 1x NVIDIA T4.

Optimized for Kaggle Notebooks and Scripts:
- Automatically loads dataset from: https://www.kaggle.com/datasets/nguyntrnminhtm/gcd-data
  (Works whether mounted at /kaggle/input/gcd-data or auto-downloaded via kagglehub).
- Automatically unpacks any archives (.tgz, .tar.gz, .zip) if present.
- Discovers CUB-200 data and precomputed long-tail splits directly from gcd-data.
- Uses 1x NVIDIA T4 GPU (default: 'cuda:0' or 'cuda').
- Enables FP16 Automatic Mixed Precision (AMP) by default (~3x speedup, 50% less VRAM).
- Sets eval_interval=5 by default (saves ~2 hours of KMeans clustering overhead, epoch 200 always evaluated).
- Sets num_workers=2 by default (ideal for Kaggle's 2-4 vCPUs).
- Saves checkpoints and metrics persistently to /kaggle/working/outputs/.

Usage in Kaggle Notebook cells:
--------------------------------
# 1. Train BaCon Baseline (1x T4, ~3 hours for 200 epochs):
!python kaggle_train.py --baseline bacon --epochs 200

# 2. Train BaCon + ProtoTail (1x T4, ~3.2 hours for 200 epochs):
!python kaggle_train.py --baseline bacon --enable-prototail --epochs 200

# 3. Train SimGCD Baseline (1x T4, ~2.5 hours for 200 epochs):
!python kaggle_train.py --baseline simgcd --epochs 200
"""

import argparse
import os
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path


def _setup_kmeans_backend(requested_backend: str) -> str:
    """Ensure requested KMeans backend is functional on Kaggle GPU.
    
    If faiss_gpu is requested, attempts to install faiss-gpu-cu12 or faiss-gpu if missing.
    If unavailable, falls back to sklearn_cpu to guarantee zero-crash execution.
    """
    if requested_backend != "faiss_gpu":
        return requested_backend

    try:
        import faiss
        if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
            print(f"[Kaggle Setup] faiss-gpu is ready ({faiss.get_num_gpus()} GPU(s) active).")
            return "faiss_gpu"
    except Exception:
        pass

    print("[Kaggle Setup] faiss-gpu not found. Installing faiss-gpu for accelerated clustering...")
    for pkg in ("faiss-gpu-cu12", "faiss-gpu"):
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                check=False,
            )
            if res.returncode == 0:
                import faiss
                if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
                    print(f"[Kaggle Setup] Successfully installed {pkg} with GPU support!")
                    return "faiss_gpu"
        except Exception as exc:
            print(f"[Kaggle Setup] Attempt {pkg} failed: {exc}")

    print("[Kaggle Setup] Warning: faiss-gpu could not be installed. Falling back to sklearn_cpu to guarantee completion.")
    return "sklearn_cpu"


def _extract_archive_if_needed(archive_path: Path, target_dir: Path) -> Path:
    """Extract .tar.gz, .tgz, or .zip archive to target_dir if not already extracted."""
    target_dir.mkdir(parents=True, exist_ok=True)
    marker = target_dir / f".extracted_{archive_path.stem}"
    if marker.is_file():
        return target_dir

    print(f"[Dataset] Extracting archive: {archive_path.name} -> {target_dir}...")
    if archive_path.suffix in (".tgz", ".gz") or archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=target_dir)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)

    marker.touch()
    return target_dir


def _prepare_gcd_dataset(
    dataset_slug: str = "nguyntrnminhtm/gcd-data",
    repo_root: Path = Path("."),
    dataset_name: str = "cub",
    imb_ratio: int = 20,
) -> tuple[str, str]:
    """
    Locate or download dataset from https://www.kaggle.com/datasets/nguyntrnminhtm/gcd-data.
    Discovers data_root and splits_root automatically based on dataset_name.
    Returns (data_root, splits_root).
    """
    norm = dataset_name.lower().strip()
    search_dirs: list[Path] = []

    # 1. Check mounted dataset directories in Kaggle
    for m in [Path("/kaggle/input/gcd-data"), Path("/kaggle/input")]:
        if m.is_dir() and m not in search_dirs:
            search_dirs.append(m)

    # 2. Local fallback directories
    for d in [Path("D:/data"), repo_root / "data", Path("./data"), Path("../data")]:
        if d.is_dir() and d not in search_dirs:
            search_dirs.append(d)

    cand_names = {
        "cub": ["cub", "CUB_200_2011", "cub200"],
        "cars": ["car", "cars", "stanford_cars", "scars"],
        "car": ["car", "cars", "stanford_cars", "scars"],
        "aircraft": ["aircraft", "fgvc_aircraft", "fgvc-aircraft-2013b"],
        "air": ["aircraft", "fgvc_aircraft", "fgvc-aircraft-2013b"],
        "herbarium": ["Herbarium 19", "herbarium", "herbarium19", "herb19"],
        "herbarium19": ["Herbarium 19", "herbarium", "herbarium19", "herb19"],
        "inat2018": ["iNaturalist 2018", "inat2018", "inat", "inaturalist"],
        "inat": ["iNaturalist 2018", "inat2018", "inat", "inaturalist"],
        "inaturalist": ["iNaturalist 2018", "inat2018", "inat", "inaturalist"],
        "cifar100": ["cifar100", "cifar-100-python", "cifar-100", "cifar100-gcd"],
        "cifar10": ["cifar10", "cifar-10-batches-py", "cifar-10"],
        "imagenet100": ["imagenet100", "imagenet100-cmc", "imagenet100_cmc", "imagenet-100", "ImageNet"],
        "imagenet": ["imagenet100", "imagenet100-cmc", "imagenet100_cmc", "imagenet-100", "ImageNet"],
    }.get(norm, [norm])

    if norm in ("imagenet100", "imagenet-100", "imagenet") and dataset_slug == "nguyntrnminhtm/gcd-data":
        dataset_slug = "minhtmnguyntrn/imagenet100-cmc"

    def _resolve_from_dirs(dirs: list[Path]) -> Path | None:
        for sdir in dirs:
            if not sdir.is_dir():
                continue
            for name in cand_names:
                if (sdir / name).is_dir():
                    return sdir / name
            for child in sdir.iterdir():
                if child.is_dir():
                    for name in cand_names:
                        if child.name.lower() == name.lower():
                            return child
                        if (child / name).is_dir():
                            return child / name
        return None

    found_dir = _resolve_from_dirs(search_dirs)

    # 3. If not found locally, download via kagglehub (standard in Kaggle)
    if found_dir is None:
        try:
            import kagglehub
        except ImportError:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "kagglehub", "-q"],
                    check=False,
                )
                import kagglehub
            except Exception:
                kagglehub = None

        if kagglehub is not None:
            try:
                print(f"[Dataset] Fetching '{dataset_slug}' via kagglehub...")
                download_path = Path(kagglehub.dataset_download(dataset_slug))
                print(f"[Dataset] Dataset ready at: {download_path}")
                search_dirs.insert(0, download_path)
                found_dir = _resolve_from_dirs([download_path])
            except Exception as exc:
                print(f"[Dataset] kagglehub notice: {exc}")

    # 4. Check for any compressed archives that need unpacking
    extract_base = (
        Path("/kaggle/working/data")
        if Path("/kaggle/working").is_dir()
        else repo_root / "data"
    )
    if found_dir is None:
        for sdir in list(search_dirs):
            if not sdir.is_dir():
                continue
            for ext in ("*.tgz", "*.tar.gz", "*.zip"):
                for archive_path in sdir.glob(ext):
                    extracted_dir = _extract_archive_if_needed(archive_path, extract_base)
                    search_dirs.insert(0, extracted_dir)
                    found_dir = _resolve_from_dirs([extracted_dir])
                    if found_dir:
                        break
                if found_dir:
                    break

    # 4b. Auto-unify ImageNet-100 parts (e.g. train_part01..05, val_part01..05)
    if norm in ("imagenet100", "imagenet-100", "imagenet"):
        unified_dir = (
            Path("/kaggle/working/data/imagenet100")
            if Path("/kaggle/working").is_dir()
            else extract_base / "imagenet100"
        )
        unified_train = unified_dir / "train"
        unified_val = unified_dir / "val"
        parts_found = 0
        for sdir in search_dirs:
            if not sdir.is_dir():
                continue
            candidates_to_check = [sdir]
            try:
                candidates_to_check.extend([c for c in sdir.iterdir() if c.is_dir()])
            except Exception:
                pass
            for cand in candidates_to_check:
                for sub, target_sub in (("train", unified_train), ("val", unified_val)):
                    sub_path = cand / sub
                    if sub_path.is_dir():
                        for syn_dir in sub_path.iterdir():
                            if syn_dir.is_dir():
                                target_sub.mkdir(parents=True, exist_ok=True)
                                dst = target_sub / syn_dir.name
                                if not dst.exists():
                                    try:
                                        os.symlink(syn_dir, dst, target_is_directory=True)
                                        parts_found += 1
                                    except Exception:
                                        try:
                                            shutil.copytree(syn_dir, dst)
                                            parts_found += 1
                                        except Exception:
                                            pass
        if parts_found > 0 or (unified_train.is_dir() and any(unified_train.iterdir())):
            print(f"[Dataset] Unified ImageNet-100 ready at: {unified_dir}")
            found_dir = unified_dir

    # 5. Resolve data_root
    if found_dir is not None:
        data_root = str(found_dir)
    elif Path("/kaggle/input/gcd-data").is_dir():
        data_root = str(Path("/kaggle/input/gcd-data") / norm)
    else:
        data_root = f"./data/{norm}"

    # 6. Resolve splits_root
    if norm in ("cub", "cub200"):
        pattern = f"cub200_k100_imb{imb_ratio}"
    elif norm in ("cars", "stanford_cars", "scars", "car"):
        pattern = f"cars196_k98_imb{imb_ratio}"
    elif norm in ("aircraft", "fgvc_aircraft", "air"):
        pattern = f"aircraft_k80_imb{imb_ratio}"
    elif norm in ("herbarium", "herbarium19", "herb19"):
        pattern = f"herbarium19_k341_imb{imb_ratio}"
    elif norm in ("inat", "inat2018", "inaturalist", "inaturalist2018"):
        pattern = "inat2018_k4071_natural" if str(imb_ratio).lower() == "natural" else f"inat2018_k4071_imb{imb_ratio}"
    elif norm in ("cifar100", "cifar-100"):
        pattern = f"cifar100_k80_imb{imb_ratio}"
    elif norm in ("cifar10", "cifar-10"):
        pattern = f"cifar10_k5_imb{imb_ratio}"
    elif norm in ("imagenet100", "imagenet-100", "imagenet"):
        pattern = "imagenet100"
    else:
        pattern = f"*_imb{imb_ratio}"

    if norm in ("imagenet100", "imagenet-100", "imagenet"):
        if (repo_root / "splits/imagenet100").is_dir():
            splits_root = str(repo_root / "splits/imagenet100")
        elif (repo_root / "splits").is_dir():
            splits_root = str(repo_root / "splits")
        else:
            splits_root = str(repo_root / "data_uq_idxs")
    else:
        splits_root = str(repo_root / "data_uq_idxs")
        if (repo_root / "data_uq_idxs" / pattern).is_dir():
            splits_root = str(repo_root / "data_uq_idxs")
        else:
            for sdir in search_dirs:
                if not sdir.is_dir():
                    continue
                for match in sdir.rglob(pattern):
                    if match.is_dir():
                        splits_root = str(match.parent)
                        break
                if splits_root != str(repo_root / "data_uq_idxs"):
                    break

    return data_root, splits_root


def _get_kaggle_account(cli_override: str | None = None) -> str:
    """Detect active Kaggle account/username from CLI, kaggle API, environment, or kaggle.json."""
    if cli_override:
        return cli_override
    for env_var in ("KAGGLE_USERNAME", "KAGGLE_USER_NAME", "KAGGLE_USER"):
        val = os.environ.get(env_var)
        if val:
            return val
    try:
        import kaggle
        cfg = getattr(kaggle.api, "config_values", {}) or {}
        username = cfg.get("username")
        if username:
            return str(username)
    except Exception:
        pass
    try:
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.is_file():
            import json as _json
            with open(kaggle_json, "r", encoding="utf-8") as f:
                data = _json.load(f)
                if "username" in data and data["username"]:
                    return str(data["username"])
    except Exception:
        pass
    if Path("/kaggle").is_dir():
        return "kaggle_notebook_active"
    return "unknown (local)"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProtoTail Kaggle 1x T4 Training Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="bacon",
        choices=["simgcd", "bacon"],
        help="GCD baseline model to train ('simgcd' or 'bacon')",
    )
    parser.add_argument(
        "--enable-prototail",
        action="store_true",
        help="Enable ProtoTail method plugin on top of the baseline",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default="nguyntrnminhtm/gcd-data",
        help="Kaggle dataset slug (https://www.kaggle.com/datasets/<dataset-slug>)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom experiment YAML config (default: configs/experiment_<baseline>.yaml)",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Total number of epochs to train"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Override base learning rate"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="Evaluation epoch interval (every N epochs). Saves ~2h of KMeans clustering.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Dataloader worker processes (2 is optimal for Kaggle vCPUs)",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP16 Automatic Mixed Precision (AMP) (~3x faster on T4, saves 50%% VRAM)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to train on ('cuda:0', 'cuda', or 'cpu')",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override root path containing CUB-200-2011",
    )
    parser.add_argument(
        "--splits-root",
        type=str,
        default=None,
        help="Override directory containing long-tail split folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints and metrics (default: /kaggle/working/outputs or ./outputs)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom experiment name tag",
    )
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", type=str, default="cub", help="Dataset name: cub, cars, aircraft")
    parser.add_argument("--imb-ratio", "--imb_ratio", dest="imb_ratio", type=int, default=10, help="Imbalance ratio: 10, 20, 50, 100")
    parser.add_argument(
        "--simgcd-eval-head",
        "--simgcd_eval_head",
        dest="simgcd_eval_head",
        type=str,
        choices=["kmeans", "parametric"],
        default="kmeans",
        help="Evaluation head for SimGCD baseline: 'kmeans' (matches ProtoTail/BaCon track) or 'parametric' (logits argmax)",
    )
    parser.add_argument("--use-parts", "--use_parts", dest="use_parts", action="store_true", default=False, help="Enable ProtoTail parts")
    parser.add_argument("--ablate-adaptive-capacity", "--ablate_adaptive_capacity", dest="ablate_adaptive_capacity", action="store_true", default=False, help="Ablate adaptive capacity")
    parser.add_argument("--enable-pseudo-labeling", "--enable_pseudo_labeling", dest="enable_pseudo_labeling", action="store_true", default=False, help="Enable pseudo labeling")
    parser.add_argument("--enable-novel-pseudo", "--enable_novel_pseudo", dest="enable_novel_pseudo", action="store_true", default=False, help="Enable novel pseudo labeling")
    parser.add_argument(
        "--kmeans-backend",
        choices=["sklearn_cpu", "faiss_gpu"],
        default="sklearn_cpu",
        help="KMeans backend for clustering eval and distribution estimation",
    )
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument(
        "--eval-feature-mode",
        choices=["original_parts_concat", "legacy_concat", "balanced_concat"],
        default=None,
        help="Evaluation feature concatenation mode (default: train.py's "
        "original_parts_concat; legacy_concat matches old Kaggle runs)",
    )
    parser.add_argument(
        "--gate-warm-start-from-support",
        action="store_true",
        default=False,
        help="Init gates from estimated training supports at epoch 0",
    )
    parser.add_argument(
        "--known-frequency-mapping",
        choices=["direct_hungarian", "bacon_rank"],
        default="direct_hungarian",
        help="Known frequency mapping mode",
    )
    parser.add_argument("--kaggle-user", "--kaggle_user", dest="kaggle_user", type=str, default=None, help="Kaggle username/account name override")
    parser.add_argument(
        "--dinov2-repo",
        "--dinov2_repo",
        dest="dinov2_repo",
        type=str,
        default=None,
        help="Local dir containing the dinov2 hub repo (hubconf.py) for offline loading; sets PROTOTAIL_DINO_REPO",
    )
    parser.add_argument(
        "--dinov2-weights",
        "--dinov2_weights",
        dest="dinov2_weights",
        type=str,
        default=None,
        help="Path to dinov2_vitb14_pretrain.pth for offline loading; sets PROTOTAIL_DINO_WEIGHTS",
    )
    args, unknown_args = parser.parse_known_args()
    if args.use_parts:
        args.enable_prototail = True

    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "src"))

    # Config path
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = repo_root / f"configs/experiment_{args.baseline}.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Auto-resolve dataset from Kaggle gcd-data
    auto_data_root, auto_splits_root = _prepare_gcd_dataset(
        dataset_slug=args.dataset_slug,
        repo_root=repo_root,
        dataset_name=args.dataset_name,
        imb_ratio=args.imb_ratio,
    )
    data_root = args.data_root or auto_data_root
    splits_root = args.splits_root or auto_splits_root

    # Output directory (with graceful Windows fallback if /kaggle path passed)
    if args.output_dir:
        if str(args.output_dir).startswith("/kaggle") and not Path("/kaggle").is_dir():
            output_dir = repo_root / "outputs"
        else:
            output_dir = Path(args.output_dir)
    elif Path("/kaggle/working").is_dir():
        output_dir = Path("/kaggle/working/outputs")
    else:
        output_dir = repo_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Experiment run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_tag = "prototail" if args.enable_prototail else "baseline"
    run_name = args.run_name or f"{args.baseline}-{method_tag}-t4-{timestamp}"

    kaggle_account = _get_kaggle_account(args.kaggle_user)

    # Determine explicit ablation configuration name
    if not args.use_parts and not args.enable_pseudo_labeling:
        detected_config = "Base"
    elif not args.use_parts and args.enable_pseudo_labeling:
        detected_config = "Pseudo"
    elif args.use_parts and args.ablate_adaptive_capacity and not args.enable_pseudo_labeling:
        detected_config = "Prototype"
    elif args.use_parts and not args.ablate_adaptive_capacity and not args.enable_pseudo_labeling:
        detected_config = "Prototype_Tailaware"
    elif args.use_parts and args.ablate_adaptive_capacity and args.enable_pseudo_labeling:
        detected_config = "Full_wo_Tail"
    elif args.use_parts and not args.ablate_adaptive_capacity and args.enable_pseudo_labeling:
        detected_config = "Full"
    else:
        detected_config = "Custom"

    print("=" * 65)
    print("        PROTOTAIL KAGGLE 1x T4 TRAINING RUNNER")
    active_kmeans_backend = _setup_kmeans_backend(args.kmeans_backend)

    print("=" * 65)
    print(f"Kaggle Account:   {kaggle_account}")
    print(f"Ablation Setting: {detected_config}")
    print(f"Kaggle Dataset:   https://www.kaggle.com/datasets/{args.dataset_slug}")
    print(f"Model Baseline:   {args.baseline.upper()}")
    print(f"Method Plugin:    {'ProtoTail (Enabled)' if args.enable_prototail else 'Pure Baseline'}")
    is_tailaware = args.use_parts and not args.ablate_adaptive_capacity
    tail_str = 'ACTIVE (0.05, warm_start=ON)' if is_tailaware else 'DISABLED (0.0)'
    print(f"Tail-aware Reg:   {tail_str}")
    print(f"KMeans Backend:   {active_kmeans_backend} (standard paper protocol)")
    print(f"Pseudo-labeling:  {'ACTIVE (D1+D2)' if (args.enable_pseudo_labeling and args.enable_novel_pseudo) else ('ACTIVE (D1)' if args.enable_pseudo_labeling else 'DISABLED')}")
    print(f"Device:           {args.device} (1x T4 GPU)")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Mixed Precision:  {'FP16 (Enabled, Tensor Cores Active)' if args.fp16 else 'FP32'}")
    print(f"Eval Interval:    Every {args.eval_interval} epochs (plus final epoch {args.epochs})")
    print(f"Num Workers:      {args.num_workers}")
    print(f"Run Name:         {run_name}")
    print(f"Data Root:        {data_root}")
    print(f"Splits Root:      {splits_root}")
    print(f"Output Directory: {output_dir}")
    print("=" * 65)
    # Build execution command for train.py
    train_script = repo_root / "src/prototail_gcd/train.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--config",
        str(config_path),
        "--run-name",
        run_name,
        "--dataset-name",
        args.dataset_name,
        "--data-root",
        data_root,
        "--splits-root",
        splits_root,
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-interval",
        str(args.eval_interval),
        "--kmeans-backend",
        active_kmeans_backend,
        "--kmeans-n-init",
        str(args.kmeans_n_init),
        "--kmeans-max-iter",
        str(args.kmeans_max_iter),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
    ]

    if args.enable_prototail:
        cmd.append("--enable-prototail")
    if args.imb_ratio is not None:
        cmd.extend(["--imb-ratio", str(args.imb_ratio)])
    if getattr(args, "simgcd_eval_head", None):
        cmd.extend(["--simgcd-eval-head", str(args.simgcd_eval_head)])
    if getattr(args, "use_parts", False):
        cmd.append("--use-parts")
    if getattr(args, "ablate_adaptive_capacity", False):
        cmd.append("--ablate-adaptive-capacity")
    if getattr(args, "enable_pseudo_labeling", False):
        cmd.append("--enable-pseudo-labeling")
    if getattr(args, "enable_novel_pseudo", False):
        cmd.append("--enable-novel-pseudo")
    if getattr(args, "eval_feature_mode", None):
        cmd.extend(["--eval-feature-mode", str(args.eval_feature_mode)])
    is_tailaware = getattr(args, "use_parts", False) and not getattr(args, "ablate_adaptive_capacity", False)
    if getattr(args, "gate_warm_start_from_support", False) or is_tailaware:
        cmd.append("--gate-warm-start-from-support")
    if getattr(args, "known_frequency_mapping", None):
        cmd.extend(["--known-frequency-mapping", str(args.known_frequency_mapping)])
    if args.lr is not None:
        cmd.extend(["--lr", str(args.lr)])
    if args.fp16:
        cmd.append("--fp16")
    else:
        cmd.append("--no-fp16")

    # Pre-flight: verify an offline DINO backbone is available when Internet is off.
    if Path("/kaggle").is_dir():
        try:
            from model.backbone import (
                locate_offline_assets,
                probe_internet,
            )

            offline_repo, offline_weights = locate_offline_assets("dinov2_vitb14")
            if offline_repo is None and not probe_internet():
                print("=" * 65)
                print("[Kaggle Setup] WARNING: Internet is OFF and no offline DINO")
                print("backbone assets were found under /kaggle/input.")
                print("Training will fail at model build. Fix (pick one):")
                print("  1. Notebook Settings -> Internet -> ON, or")
                print("  2. Create + attach the offline dataset:")
                print("       python scripts/prepare_dinov2_offline.py --output dinov2_offline")
                print("     upload `dinov2_offline` as a Kaggle dataset, then attach it.")
                print("  3. Pass --dinov2-repo / --dinov2-weights explicitly.")
                print("=" * 65)
                sys.stdout.flush()
        except Exception:
            pass

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    if args.dinov2_repo:
        env["BACON_DINO_REPO"] = str(args.dinov2_repo)
        env["PROTOTAIL_DINO_REPO"] = str(args.dinov2_repo)
    if args.dinov2_weights:
        env["BACON_DINO_WEIGHTS"] = str(args.dinov2_weights)
        env["PROTOTAIL_DINO_WEIGHTS"] = str(args.dinov2_weights)

    print(f"\n[Starting training]: {' '.join(cmd)}\n")
    sys.stdout.flush()

    process = subprocess.run(cmd, env=env)
    if process.returncode != 0:
        print(f"\n[Error] Training failed with return code {process.returncode}")
        sys.exit(process.returncode)

    print("\n" + "=" * 65)
    print("Training finished successfully!")
    print(f"Artifacts and metrics saved to: {output_dir}")
    summary_file = output_dir / f"{run_name}.summary.json"
    if summary_file.is_file():
        print(f"Training summary saved to:   {summary_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
