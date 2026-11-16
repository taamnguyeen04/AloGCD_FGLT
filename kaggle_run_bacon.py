"""Kaggle-side runner for BaCon (Modal-parity).

Runs the CURRENT repo trainer ``python -m model.bacon`` with EXACTLY the same
command that ``modal_train.py::train`` builds (same flags, same config mapping,
same defaults), so a Kaggle run is protocol-identical to a Modal run.

This replaces the old prototail/kaggle_train.py path. No ``gcd-hosts/prototail``
code is used or required.

Responsibilities:
  1. Locate the dataset under ``/kaggle/input/**`` (per-dataset markers) and
     export the env roots that ``config.py`` reads (CUB_ROOT, CARS_ROOT,
     AIRCRAFT_ROOT, HERBARIUM_ROOT, CIFAR10_ROOT, CIFAR100_ROOT, IMAGENET_ROOT).
  2. Auto-detect an offline DINOv2 dataset (hubconf.py + vitb14 weights) and set
     BACON_DINO_REPO / BACON_DINO_WEIGHTS; otherwise torch.hub downloads online.
  3. Best-effort pip guard (loguru, tensorboard are not always preinstalled).
  4. Build the exact Modal-equivalent command and run it with cwd = bundle root
     (so ``data_uq_idxs/...`` relative split dirs resolve like on Modal).

Kaggle defaults (override via CLI): num-workers 4, time-budget-hours 8.5
(9h GPU session cap), vis-freq 0, save-freq 20 (20GB working-dir quota).
"""

import argparse
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Dataset registry (mirrors modal_train.train())
# ---------------------------------------------------------------------------
_DATASET_ALIASES = {
    "car": "stanford_cars",
    "cars": "stanford_cars",
    "scars": "stanford_cars",
    "stanford_cars": "stanford_cars",
    "cub": "cub200",
    "cub200": "cub200",
    "air": "fgvc_aircraft",
    "aircraft": "fgvc_aircraft",
    "fgvc_aircraft": "fgvc_aircraft",
    "herbarium": "herbarium19",
    "herbarium19": "herbarium19",
    "her19": "herbarium19",
    "herb19": "herbarium19",
    "cifar-10": "cifar10",
    "cifar10": "cifar10",
    "cifar-100": "cifar100",
    "cifar100": "cifar100",
    "imagenet": "imagenet100",
    "imagenet-100": "imagenet100",
    "imagenet100": "imagenet100",
    "inat": "inat2018",
    "inat2018": "inat2018",
    "inaturalist": "inat2018",
    "inaturalist2018": "inat2018",
}

# canonical dataset -> env var read by config.py / data loaders
_ENV_VAR_BY_DATASET = {
    "cub200": "CUB_ROOT",
    "stanford_cars": "CARS_ROOT",
    "fgvc_aircraft": "AIRCRAFT_ROOT",
    "herbarium19": "HERBARIUM_ROOT",
    "cifar10": "CIFAR10_ROOT",
    "cifar100": "CIFAR100_ROOT",
    "imagenet100": "IMAGENET_ROOT",
    "inat2018": "INAT_ROOT",
}

_LABELED_CLASSES = {
    "stanford_cars": 98,
    "cifar10": 5,
    "cub200": 100,
    "fgvc_aircraft": 80,
    "imagenet100": 50,
    "herbarium19": 341,
}

# data_uq_idxs/<name> used by the relative split-file loaders (fail-loud warn)
_SPLIT_DIR_TEMPLATE = {
    "cub200": "cub200_k100_imb{imb}",
    "stanford_cars": "cars196_k98_imb{imb}",
    "fgvc_aircraft": "aircraft_k80_imb{imb}",
    "cifar100": "cifar100_k80_imb{imb}",
    "cifar10": "cifar10_k5_imb{imb}",
}

CONFIG_MAPPING = {
    "Base": dict(use_parts=False, enable_pseudo_labeling=False, enable_novel_pseudo=False,
                 ablate_adaptive_capacity=False),
    "Pseudo": dict(use_parts=False, enable_pseudo_labeling=True, enable_novel_pseudo=True,
                   ablate_adaptive_capacity=False),
    "Prototype": dict(use_parts=True, enable_pseudo_labeling=False, enable_novel_pseudo=False,
                      ablate_adaptive_capacity=True),
    "Prototype_Tailaware": dict(use_parts=True, enable_pseudo_labeling=False, enable_novel_pseudo=False,
                                ablate_adaptive_capacity=False),
    "Full_wo_Tail": dict(use_parts=True, enable_pseudo_labeling=True, enable_novel_pseudo=True,
                         ablate_adaptive_capacity=True),
    "Full": dict(use_parts=True, enable_pseudo_labeling=True, enable_novel_pseudo=True,
                 ablate_adaptive_capacity=False),
}


def map_dataset_name(name: str) -> str:
    return _DATASET_ALIASES.get(name.strip().lower(), name.strip().lower())


def auto_run_name(dataset: str, config: str, imb_ratio, suffix: str = "") -> str:
    name = f"{dataset}_{config}"
    if imb_ratio is not None and str(imb_ratio) not in ("", "None"):
        name += f"_imb{imb_ratio}"
    if suffix:
        name += f"_{suffix}"
    return name


# ---------------------------------------------------------------------------
# /kaggle/input scanning + per-dataset finders
# ---------------------------------------------------------------------------
KAGGLE_INPUT = Path("/kaggle/input")
EXTRACT_DIR = Path("/kaggle/working/extracted")
_SCAN_DEPTH = 6


def scan_roots(extra_root: Path | None = None):
    """Collect (dirs, files, children_map) under /kaggle/input (+ extracted), depth-limited."""
    roots = [KAGGLE_INPUT]
    if extra_root is not None:
        roots.append(extra_root)
    dirs, files, children = [], [], {}
    for root in roots:
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            dirs.append(d)
            subs = []
            try:
                entries = sorted(os.scandir(d), key=lambda e: e.name)
            except OSError:
                continue
            for e in entries:
                if e.is_dir(follow_symlinks=False):
                    subs.append(Path(e.path))
                    if depth < _SCAN_DEPTH:
                        stack.append((Path(e.path), depth + 1))
                elif e.is_file(follow_symlinks=False):
                    files.append(Path(e.path))
            children[d] = subs
    return dirs, files, children


def _shallowest(cands):
    return sorted(cands, key=lambda p: len(p.parts))


def find_herbarium(dirs, children):
    """Root = parent of a dir holding exactly 683 class subdirs (small-train ...)."""
    cands = _shallowest([d for d in dirs if d != KAGGLE_INPUT and len(children.get(d, [])) == 683])
    for d in cands:
        parent = d.parent
        return parent if parent.parent.exists() else d
    return None


def find_cub(dirs, files):
    by_dir = {}
    for f in files:
        if f.name in ("images.txt", "image_class_labels.txt", "train_test_split.txt"):
            by_dir.setdefault(f.parent, set()).add(f.name)
    for d in _shallowest([d for d in by_dir if len(by_dir[d]) >= 2 and "images.txt" in by_dir[d]
                          and "image_class_labels.txt" in by_dir[d]]):
        return d
    for d in _shallowest([d for d in dirs if d.name.upper().startswith("CUB_200_2011")]):
        return d
    return None


def find_cars(dirs, files):
    mats = _shallowest([f.parent for f in files if f.name == "cars_train_annos.mat"])
    mat_dirs = set(mats)
    for d in mats:  # annos at root AND cars_train images subdir present
        if any(c.name == "cars_train" for c in _children_of(d, dirs)):
            return d
    if mats:
        return mats[0]
    for d in _shallowest([d for d in dirs
                          if d.name.lower().replace("_", "").replace("-", "") in
                          ("carsdataset", "stanfordcars", "cars")]):
        names = {c.name for c in _children_of(d, dirs)}
        if {"cars_train", "cars_test"} <= names or {"cars_train", "devkit"} <= names:
            return d
    return None


def _children_of(d, dirs):
    return [p for p in dirs if p.parent == d]


def find_aircraft(dirs, files):
    for d in _shallowest(dirs):
        if d.name == "fgvc-aircraft-2013b":
            return d.parent
    for d in _shallowest(dirs):
        for c in _children_of(d, dirs):
            if c.name == "fgvc-aircraft-2013b":
                return d
    return None


def find_cifar100(dirs, files):
    for d in _shallowest(dirs):
        if d.name == "cifar-100-python":
            return d
    return None


def find_cifar10(dirs, files):
    for d in _shallowest(dirs):
        if d.name in ("cifar-10-batches-py", "cifar-10-python"):
            return d
    return None


def find_imagenet(dirs, files):
    for d in _shallowest(dirs):
        low = d.name.lower().replace("_", "").replace("-", "")
        if "imagenet" in low or low == "in1k":
            names = {c.name for c in _children_of(d, dirs)}
            if {"train", "val"} <= names or "ImageNet_100" in d.name:
                return d
    return None


def find_inat(dirs, files):
    for d in _shallowest(dirs):
        if "inat" in d.name.lower() or "inaturalist" in d.name.lower():
            return d
    for f in files:
        if "val2018" in f.name:
            return f.parent.parent
    return None


FINDERS = {
    "herbarium19": find_herbarium,
    "cub200": find_cub,
    "stanford_cars": find_cars,
    "fgvc_aircraft": find_aircraft,
    "cifar100": find_cifar100,
    "cifar10": find_cifar10,
    "imagenet100": find_imagenet,
    "inat2018": find_inat,
}

_HINT_TOKENS = {
    "cub200": ("cub", "bird"),
    "stanford_cars": ("car",),
    "fgvc_aircraft": ("aircraft", "fgvc"),
    "herbarium19": ("herb",),
    "cifar10": ("cifar-10", "cifar10"),
    "cifar100": ("cifar-100", "cifar100"),
    "imagenet100": ("imagenet", "in1k"),
    "inat2018": ("inat", "naturalist"),
}


def maybe_extract_archives(dataset, files) -> Path | None:
    """Extract hinted zip/tar archives into /kaggle/working/extracted, return that root."""
    tokens = _HINT_TOKENS.get(dataset, ())
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    extracted_any = False
    for f in files:
        low = f.name.lower()
        if not (low.endswith(".zip") or low.endswith(".tar") or low.endswith(".tar.gz")
                or low.endswith(".tgz")):
            continue
        if tokens and not any(t in low for t in tokens):
            continue
        out = EXTRACT_DIR / f.stem.replace(".tar", "")
        if out.exists():
            continue
        print(f"[data-locate] extracting {f} -> {out} ...", flush=True)
        try:
            out.mkdir(parents=True, exist_ok=True)
            if low.endswith(".zip"):
                with zipfile.ZipFile(f) as z:
                    z.extractall(out)
            else:
                mode = "r:gz" if (low.endswith(".gz") or low.endswith(".tgz")) else "r:"
                with tarfile.open(f, mode) as t:
                    t.extractall(out)
            extracted_any = True
        except Exception as e:
            print(f"[data-locate] extraction failed for {f}: {e}", flush=True)
    return EXTRACT_DIR if extracted_any else None


def find_dino_offline(dirs, files):
    repo = None
    weights = None
    for f in files:
        if f.name == "hubconf.py" and repo is None:
            repo = f.parent
    for f in files:
        if f.name.startswith("dinov2_vitb14") and f.name.endswith(".pth") and weights is None:
            weights = f
    if repo is not None and "dinov2" not in repo.name.lower():
        better = [d for d in dirs if (d / "hubconf.py").is_file() and "dinov2" in d.name.lower()]
        if better:
            repo = sorted(better, key=lambda p: len(p.parts))[0]
    return repo, weights


# ---------------------------------------------------------------------------
# Command build (mirror modal_train.train() verbatim)
# ---------------------------------------------------------------------------
def build_bacon_command(a, exp_root: str) -> list:
    """a: argparse.Namespace with the fields parsed in main(); exp_root: exact dir."""
    command = [
        sys.executable, "-m", "model.bacon",
        "--dataset-name", a.dataset_name,
        "--labeled-classes", str(a.labeled_classes),
        "--imb-ratio", str(a.imb_ratio),
        "--epochs", str(a.epochs),
        "--stop-epoch", str(a.epochs),
        "--batch-size", str(a.batch_size),
        "--num-workers", str(a.num_workers),
        "--exp-name", a.dataset_name,
        "--warmup-teacher-temp-epochs", str(min(50, a.epochs)),
        "--est-freq", "10",
        "--ce-warmup", "1",
        "--alpha", "0",
        "--beta", "0.5",
        "--use-exact-exp-root",
        "--exp-root", exp_root,
    ]

    if a.enable_pseudo_labeling:
        command.append("--enable-pseudo-labeling")
        command.extend(["--pseudo-mode", str(a.pseudo_mode)])
        command.extend(["--confidence-threshold", str(a.confidence_threshold)])
        command.extend(["--pseudo-top-ratio", str(a.pseudo_top_ratio)])
        command.extend(["--max-samples-per-class", str(a.max_samples_per_class)])
        command.extend(["--pseudo-update-freq", str(a.pseudo_update_freq)])
        command.extend(["--max-pseudo-iterations", str(a.max_pseudo_iterations)])
        command.extend(["--pseudo-warmup-epoch", str(a.pseudo_warmup_epoch)])
        command.extend(["--pseudo-conf-bar", str(a.pseudo_conf_bar)])
        command.extend(["--pseudo-bar-k", str(a.pseudo_bar_k)])
        command.extend(["--pseudo-min-hi", str(a.pseudo_min_hi)])
        if a.enable_novel_pseudo:
            command.append("--enable-novel-pseudo")
            command.extend(["--novel-warmup-epoch", str(a.novel_warmup_epoch)])
            command.extend(["--novel-update-freq", str(a.novel_update_freq)])
            command.extend(["--max-novel-iterations", str(a.max_novel_iterations)])
            command.extend(["--novel-max-samples", str(a.novel_max_samples)])
            command.extend(["--novel-jaccard-th", str(a.novel_jaccard_th)])
            command.extend(["--novel-agree-th", str(a.novel_agree_th)])
            command.extend(["--novel-min-size", str(a.novel_min_size)])

    command.extend(["--vis-freq", str(a.vis_freq)])
    command.extend(["--backbone", a.backbone.replace("-", "_")])

    if a.seed is not None and int(a.seed) >= 0:
        command.extend(["--seed", str(int(a.seed))])

    command.extend(["--early-stop-patience", str(int(a.early_stop_patience))])

    if a.resume_ckpt:
        command.extend(["--resume", str(a.resume_ckpt)])
    if a.time_budget_hours and float(a.time_budget_hours) > 0:
        command.extend(["--time-budget-hours", str(float(a.time_budget_hours))])
    if int(a.save_freq) > 0:
        command.extend(["--save-freq", str(int(a.save_freq))])

    if a.use_logit_adjust:
        command.append("--use-logit-adjust")
    if a.use_momentum_teacher:
        command.append("--use-momentum-teacher")
        command.extend(["--teacher-m0", str(a.teacher_m0)])

    if a.use_parts:
        command.append("--use-parts")
        command.extend(["--num-slots", str(a.num_slots)])
        command.extend(["--part-lambda", str(a.part_lambda)])
        if a.ablate_fused_ce:
            command.append("--ablate-fused-ce")
        if a.ablate_spatial_loss:
            command.append("--ablate-spatial-loss")
        if a.ablate_confidence:
            command.append("--ablate-confidence")
        if a.ablate_adaptive_capacity:
            command.append("--ablate-adaptive-capacity")
        if a.ablate_concat_eval:
            command.append("--ablate-concat-eval")

    if a.fp16:
        command.append("--fp16")

    if a.lr is not None:
        command.extend(["--lr", str(a.lr)])
    elif a.dataset_name == "cifar10":
        command.extend(["--lr", "0.01"])

    if a.extra_args:
        command.extend(a.extra_args.split())

    return command


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run BaCon (model/bacon.py) on Kaggle, modal_train.py parity")
    parser.add_argument("--baseline", default="bacon", help="(informational; BaCon trainer always used, like Modal)")
    parser.add_argument("--run-name", default="", help="Experiment dir name (default: auto bacon_<ds>_<cfg>_imbN)")
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", required=True)
    parser.add_argument("--labeled-classes", "--labeled_classes", dest="labeled_classes", type=int, default=None)
    parser.add_argument("--imb-ratio", "--imb_ratio", dest="imb_ratio", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--config", choices=list(CONFIG_MAPPING), default="Base")
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--vis-freq", "--vis_freq", dest="vis_freq", type=int, default=0)
    parser.add_argument("--save-freq", "--save_freq", dest="save_freq", type=int, default=20,
                        help="Epoch checkpoint interval (Kaggle 20GB quota; 0 = bacon.py default path)")
    parser.add_argument("--early-stop-patience", dest="early_stop_patience", type=int, default=50)
    parser.add_argument("--resume-ckpt", dest="resume_ckpt", default="", help="Checkpoint path readable in-container")
    parser.add_argument("--time-budget-hours", dest="time_budget_hours", type=float, default=8.5)
    parser.add_argument("--fp16", action="store_true", help="AMP mixed precision (fallback if T4/P100 OOM)")
    parser.add_argument("--use-logit-adjust", dest="use_logit_adjust", action="store_true")
    parser.add_argument("--use-momentum-teacher", dest="use_momentum_teacher", action="store_true")
    parser.add_argument("--teacher-m0", dest="teacher_m0", type=float, default=0.996)
    # Pseudo labeling params (modal launch defaults)
    parser.add_argument("--pseudo-mode", dest="pseudo_mode", type=int, default=1)
    parser.add_argument("--confidence-threshold", dest="confidence_threshold", type=float, default=0.9)
    parser.add_argument("--pseudo-top-ratio", dest="pseudo_top_ratio", type=float, default=0.8)
    parser.add_argument("--max-samples-per-class", dest="max_samples_per_class", type=int, default=500)
    parser.add_argument("--pseudo-update-freq", dest="pseudo_update_freq", type=int, default=10)
    parser.add_argument("--max-pseudo-iterations", dest="max_pseudo_iterations", type=int, default=20)
    parser.add_argument("--pseudo-warmup-epoch", dest="pseudo_warmup_epoch", type=int, default=30)
    parser.add_argument("--pseudo-conf-bar", dest="pseudo_conf_bar", type=float, default=0.5)
    parser.add_argument("--pseudo-bar-k", dest="pseudo_bar_k", type=float, default=2.0)
    parser.add_argument("--pseudo-min-hi", dest="pseudo_min_hi", type=int, default=10)
    # Novel-door params (modal launch defaults)
    parser.add_argument("--novel-warmup-epoch", dest="novel_warmup_epoch", type=int, default=50)
    parser.add_argument("--novel-update-freq", dest="novel_update_freq", type=int, default=10)
    parser.add_argument("--max-novel-iterations", dest="max_novel_iterations", type=int, default=2)
    parser.add_argument("--novel-max-samples", dest="novel_max_samples", type=int, default=100)
    parser.add_argument("--novel-jaccard-th", dest="novel_jaccard_th", type=float, default=0.6)
    parser.add_argument("--novel-agree-th", dest="novel_agree_th", type=float, default=0.7)
    parser.add_argument("--novel-min-size", dest="novel_min_size", type=int, default=10)
    # AdaPart params (modal launch defaults)
    parser.add_argument("--num-slots", dest="num_slots", type=int, default=3)
    parser.add_argument("--part-lambda", dest="part_lambda", type=float, default=0.5)
    parser.add_argument("--ablate-fused-ce", dest="ablate_fused_ce", action="store_true")
    parser.add_argument("--ablate-spatial-loss", dest="ablate_spatial_loss", action="store_true")
    parser.add_argument("--ablate-confidence", dest="ablate_confidence", action="store_true")
    parser.add_argument("--ablate-adaptive-capacity", dest="ablate_adaptive_capacity", action="store_true")
    parser.add_argument("--ablate-concat-eval", dest="ablate_concat_eval", action="store_true")
    parser.add_argument("--extra-args", dest="extra_args", default="", help="Raw extra flags appended to bacon.py")
    parser.add_argument("--exp-root-base", dest="exp_root_base", default="outputs/experiments")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = parser.parse_args()

    # dataset name + config mapping (same as modal_train.launch)
    args.dataset_name = map_dataset_name(args.dataset_name)
    cfg = CONFIG_MAPPING.get(args.config, {})
    for k, v in cfg.items():
        setattr(args, k, v)

    if args.labeled_classes is None:
        args.labeled_classes = _LABELED_CLASSES.get(args.dataset_name, 80)

    bundle_root = Path(__file__).resolve().parent
    os.chdir(bundle_root)
    run_name = args.run_name or auto_run_name(args.dataset_name, args.config, args.imb_ratio)
    exp_root = str(Path(args.exp_root_base) / run_name)

    # ---- pip guard (only inside Kaggle; avoids side effects in local dry-runs) ----
    on_kaggle = KAGGLE_INPUT.exists()
    if on_kaggle:
        for module, pkg in (("loguru", "loguru"), ("tensorboard", "tensorboard")):
            try:
                __import__(module)
            except ImportError:
                print(f"[pip-guard] installing {pkg} ...", flush=True)
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

    # ---- locate dataset + set env roots ----
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if on_kaggle:
        dirs, files, children = scan_roots()
        finder = FINDERS.get(args.dataset_name)
        root = finder(dirs, files) if finder else None
        if root is None:
            extra = maybe_extract_archives(args.dataset_name, files)
            if extra is not None:
                dirs, files, children = scan_roots(extra_root=extra)
                root = finder(dirs, files) if finder else None
        if root is None:
            print(f"[data-locate] ERROR: could not locate dataset '{args.dataset_name}' under /kaggle/input")
            print("Tree (depth-limited):")
            for d in sorted(set(dirs), key=lambda p: len(p.parts))[:200]:
                print("  ", d)
            sys.exit(2)
        env[_ENV_VAR_BY_DATASET.get(args.dataset_name, "DATA_ROOT")] = str(root)
        print(f"[data-locate] {args.dataset_name}: {_ENV_VAR_BY_DATASET.get(args.dataset_name)}={root}")

        # best-effort: export other found roots too (harmless)
        for ds, fn in FINDERS.items():
            if ds == args.dataset_name:
                continue
            r = fn(dirs, files)
            if r is not None:
                env[_ENV_VAR_BY_DATASET[ds]] = str(r)

        # offline DINOv2 (torch.hub repo + vitb14 weights)
        dino_repo, dino_weights = find_dino_offline(dirs, files)
        if dino_repo is not None:
            env["BACON_DINO_REPO"] = str(dino_repo)
            print(f"[dino-offline] BACON_DINO_REPO={dino_repo}")
        if dino_weights is not None:
            env["BACON_DINO_WEIGHTS"] = str(dino_weights)
            print(f"[dino-offline] BACON_DINO_WEIGHTS={dino_weights}")
        if dino_repo is None and dino_weights is None:
            print("[dino-offline] no offline bundle found -> torch.hub download (internet ON required)")

    # ---- sanity warnings (fail-loud but not fatal; loaders error loudly anyway) ----
    tpl = _SPLIT_DIR_TEMPLATE.get(args.dataset_name)
    if tpl:
        split_dir = bundle_root / "data_uq_idxs" / tpl.format(imb=args.imb_ratio)
        if not split_dir.is_dir():
            print(f"[WARN] missing data_uq_idxs/{split_dir.name} (bundle should contain it) -> loader may fail")
    if args.dataset_name == "imagenet100" and not (bundle_root / "splits" / "imagenet100").is_dir():
        print("[WARN] missing splits/imagenet100 txt lists -> imagenet loader may fail")

    command = build_bacon_command(args, exp_root)
    print("\n[Kaggle Runner] env roots:")
    for key in ("CUB_ROOT", "CARS_ROOT", "AIRCRAFT_ROOT", "HERBARIUM_ROOT",
                "CIFAR10_ROOT", "CIFAR100_ROOT", "IMAGENET_ROOT"):
        if key in env and env[key] != os.environ.get(key):
            print(f"  {key}={env[key]}")
    print(f"\n[Kaggle Runner] exp_root: {exp_root}")
    print("[Kaggle Runner] Launching (modal_train parity):\n  " + " ".join(command) + "\n")
    sys.stdout.flush()

    if args.dry_run:
        print("[dry-run] command built OK; not executing.")
        return

    process = subprocess.run(command, env=env)
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
