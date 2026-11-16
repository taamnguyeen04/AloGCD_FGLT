"""Kaggle Kernel Push Launcher (BaCon trainer, Modal-parity).

Packages the CURRENT BaCon repo (``model/bacon.py`` trainer — the exact same
code that ``modal_train.py`` runs on Modal) into a self-contained bundle,
creates kernel-metadata.json, and pushes to Kaggle to trigger background GPU
training (Save & Run All). You can shut down your PC immediately after.

The legacy ``gcd-hosts/prototail`` code is NOT used anywhere anymore:
  - trainer            -> python -m model.bacon  (same flags/config map as Modal)
  - bundle             -> model/ + data/ + util/ + config.py + runner +
                          data_uq_idxs/<target dataset> (+ splits/ for imagenet)
  - config mapping     -> identical to modal_train.launch()
  - results            -> /kaggle/working/outputs/experiments/<run_name>/...

Usage (same CLI habit as before):
    python kaggle_push.py --baseline bacon --dataset-name cars --imb-ratio 50 --config Full_wo_Tail
    python kaggle_push.py --baseline bacon --dataset-name herb19 --imb-ratio 100 --config Full
Extra bacon.py flags can be forwarded with --extra-args "...". --baseline is
informational only (the BaCon trainer is always used, exactly like Modal).
"""

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import zipfile
from pathlib import Path


def _get_active_kaggle_username() -> str:
    """Read active Kaggle username from kaggle CLI / credentials."""
    try:
        import kaggle
        cfg = getattr(kaggle.api, "config_values", {}) or {}
        username = cfg.get("username")
        if username:
            return str(username)
    except Exception:
        pass
    for k in ("KAGGLE_USERNAME", "KAGGLE_USER_NAME", "KAGGLE_USER"):
        v = os.environ.get(k)
        if v:
            return v
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.is_file():
        try:
            with open(kaggle_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "username" in data:
                    return str(data["username"])
        except Exception:
            pass
    return "minhtmnguyntrn"


def _split_prefixes(dataset: str) -> list:
    dl = (dataset or "").lower()
    if "cub" in dl:
        return ["cub", "cub200"]
    if "car" in dl:
        return ["car", "cars196", "cars"]
    if "air" in dl:
        return ["air", "aircraft"]
    if "herb" in dl:
        return ["herb", "herbarium19", "herbarium"]
    if "inat" in dl:
        return ["inat", "inat2018"]
    if "imagenet" in dl:
        return []  # imagenet uses splits/ txt lists, not data_uq_idxs
    return [dl] if dl else []


def _build_code_bundle_b64(repo_root: Path, target_dataset: str = "", target_imb: str | None = None) -> str:
    """Zip the BaCon trainer code (this repo only) + target-dataset splits into a b64 payload.

    Contents (zip root = repo layout expected by kaggle_run_bacon.py):
      kaggle_run_bacon.py  runner (locates datasets, sets env, builds modal-parity cmd)
      config.py            env-var data roots
      model/ data/ util/   trainer + loaders + utils (data/augmentations pkg included)
      data_uq_idxs/<target dataset split dirs>   (relative split files for imb datasets)
      splits/              (imagenet only)
    """
    required = [
        repo_root / "kaggle_run_bacon.py",
        repo_root / "config.py",
        repo_root / "model",
        repo_root / "data",
        repo_root / "util",
    ]
    missing = [str(p) for p in required if not (p.is_file() or p.is_dir())]
    if missing:
        raise FileNotFoundError("Missing BaCon repo files: " + ", ".join(missing))

    buf = io.BytesIO()
    n_files = 0
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        def add_file(path: Path, arcname: str):
            nonlocal n_files
            src = path.read_bytes()
            if path.suffix == ".py":
                compile(src, str(path), "exec")  # syntax-check before shipping
            z.writestr(arcname, src)
            n_files += 1

        def walk_dir(src_dir: Path):
            for root, dirs, files in os.walk(src_dir):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for f in sorted(files):
                    if f.endswith(".pyc"):
                        continue
                    p = Path(root) / f
                    add_file(p, p.relative_to(repo_root).as_posix())

        add_file(repo_root / "kaggle_run_bacon.py", "kaggle_run_bacon.py")
        add_file(repo_root / "config.py", "config.py")
        walk_dir(repo_root / "model")
        walk_dir(repo_root / "data")
        walk_dir(repo_root / "util")

        # data_uq_idxs/ — pack only the target dataset's split dirs (keep bundle < ~1MB)
        s_dir = repo_root / "data_uq_idxs"
        prefixes = _split_prefixes(target_dataset)
        if s_dir.is_dir() and prefixes:
            for child in sorted(s_dir.iterdir()):
                if not child.is_dir() or not any(child.name.lower().startswith(p) for p in prefixes):
                    continue
                cname = child.name.lower()
                # herbarium/inat splits are name-tagged; allow either the requested imb tag or 'natural'
                if ("herbarium" in cname or "inat" in cname) and target_imb not in (None, "", "natural"):
                    if f"imb{target_imb}" not in cname and "natural" not in cname:
                        continue
                for root, dirs, files in os.walk(child):
                    dirs[:] = [d for d in dirs if d != "__pycache__"]
                    for f in sorted(files):
                        p = Path(root) / f
                        add_file(p, "data_uq_idxs/" + p.relative_to(s_dir).as_posix())

        # splits/ — imagenet only
        if "imagenet" in (target_dataset or "").lower():
            splits_dir = repo_root / "splits"
            if splits_dir.is_dir():
                for root, dirs, files in os.walk(splits_dir):
                    dirs[:] = [d for d in dirs if d != "__pycache__"]
                    for f in sorted(files):
                        p = Path(root) / f
                        add_file(p, "splits/" + p.relative_to(splits_dir).as_posix())

    print(f"[bundle] {n_files} files packed ({len(buf.getvalue())/1024:.1f} KB raw)")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _safe_clean_staging_dir(staging_dir: Path) -> None:
    """Safely clean and create staging directory, resilient to Windows file handles."""
    if staging_dir.exists():
        try:
            shutil.rmtree(staging_dir, ignore_errors=True)
        except Exception:
            pass
    if staging_dir.exists():
        for child in staging_dir.iterdir():
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
            except Exception:
                pass
    staging_dir.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Push BaCon (model/bacon.py, modal parity) training job to Kaggle GPU")
    parser.add_argument("--baseline", choices=["bacon", "simgcd"], default="bacon",
                        help="informational only — the BaCon trainer is always used (like modal_train.py)")
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name",
                        choices=["cub", "cub200", "cars", "car", "scars", "aircraft", "air", "fgvc_aircraft",
                                 "herbarium", "herbarium19", "herb19", "inat", "inat2018", "inaturalist",
                                 "inaturalist2018", "cifar100", "cifar-100", "cifar10", "cifar-10",
                                 "imagenet100", "imagenet-100", "imagenet"],
                        default="cub")
    parser.add_argument("--imb-ratio", "--imb_ratio", dest="imb_ratio", type=str, default=None,
                        help="Imbalance ratio (natural/omitted only for inat2018; herb19 uses natural split internally)")
    parser.add_argument("--config", choices=["Base", "Pseudo", "Prototype", "Prototype_Tailaware",
                                             "Full_wo_Tail", "Full"], default="Base",
                        help="same mapping as modal_train.launch()")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument("--vis-freq", "--vis_freq", dest="vis_freq", type=int, default=0,
                        help="PCA/t-SNE vis interval on Kaggle (0 = off, saves time+quota)")
    parser.add_argument("--save-freq", "--save_freq", dest="save_freq", type=int, default=20,
                        help="epoch checkpoint interval (default 20; bacon.py default 5 would blow the 20GB quota)")
    parser.add_argument("--time-budget-hours", dest="time_budget_hours", type=float, default=8.5,
                        help="graceful stop before Kaggle's ~9h GPU session cap (0 = disabled)")
    parser.add_argument("--early-stop-patience", dest="early_stop_patience", type=int, default=50)
    parser.add_argument("--resume-ckpt", dest="resume_ckpt", default="",
                        help="checkpoint path readable in-container, e.g. /kaggle/input/<dataset>/outputs/experiments/<run>/checkpoints/model_best.pt")
    parser.add_argument("--fp16", action="store_true", help="AMP mixed precision (fallback if T4/P100 OOM)")
    parser.add_argument("--use-logit-adjust", dest="use_logit_adjust", action="store_true")
    parser.add_argument("--use-momentum-teacher", dest="use_momentum_teacher", action="store_true")
    parser.add_argument("--teacher-m0", dest="teacher_m0", type=float, default=0.996)
    parser.add_argument("--extra-args", dest="extra_args", default="",
                        help="raw extra flags forwarded to bacon.py (overrides config-derived flags)")
    parser.add_argument("--exp-name-suffix", dest="exp_name_suffix", default="",
                        help="suffix appended to the experiment/run name")
    # legacy flags accepted for CLI habit compatibility; no-ops on the BaCon path
    parser.add_argument("--kmeans-backend", choices=["sklearn_cpu", "faiss_gpu"], default="sklearn_cpu",
                        help="(ignored — BaCon trainer always uses sklearn KMeans)")
    parser.add_argument("--eval-interval", "--eval_interval", dest="eval_interval", type=int, default=5,
                        help="(ignored — BaCon uses --est-freq 10 like Modal)")
    # Kaggle plumbing
    parser.add_argument("--kaggle-user", "--kaggle_user", dest="kaggle_user", type=str, default=None)
    parser.add_argument("--dataset-slug", "--dataset_slug", dest="dataset_slug", type=str, default="nguyntrnminhtm/gcd-data")
    parser.add_argument("--extra-datasets", "--extra_datasets", dest="extra_datasets", type=str, default=None,
                        help="Comma-separated additional Kaggle dataset slugs (e.g. nguyeentaam/dinov2-offline-bundle)")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Build kernel locally without pushing")
    args = parser.parse_args()
    name_map = {
        "car": "cars", "scars": "cars", "air": "aircraft", "fgvc_aircraft": "aircraft",
        "cub200": "cub", "herbarium": "herbarium19", "herb19": "herbarium19",
        "inat": "inat2018", "inaturalist": "inat2018", "inaturalist2018": "inat2018",
        "cifar-100": "cifar100", "cifar-10": "cifar10",
        "imagenet-100": "imagenet100", "imagenet": "imagenet100",
    }
    dataset_name = name_map.get(args.dataset_name.lower(), args.dataset_name.lower())

    repo_root = Path(__file__).resolve().parent
    kaggle_user = args.kaggle_user or _get_active_kaggle_username()

    # natural imbalance only applies to inat2018 (herb19 is natural inside its loader)
    is_inat = dataset_name == "inat2018"
    is_natural = is_inat and (args.imb_ratio is None or str(args.imb_ratio).lower() in ("natural", "none", "0", ""))
    if is_natural:
        imb_ratio = None
    elif args.imb_ratio is None:
        imb_ratio = "20"
    else:
        imb_ratio = str(args.imb_ratio).strip().lower()
        if not imb_ratio.isdigit():
            if is_inat and imb_ratio in ("natural", "none", ""):
                imb_ratio = None
            else:
                parser.error(f"--imb-ratio must be an integer (got '{args.imb_ratio}')")

    exp_name = f"{args.baseline}_{dataset_name}_{args.config}"
    if imb_ratio is not None:
        exp_name += f"_imb{imb_ratio}"
    if args.exp_name_suffix:
        exp_name += f"_{args.exp_name_suffix}"
    kernel_slug = exp_name.lower().replace("_", "-")

    staging_dir = repo_root / "build_kaggle_kernel"
    _safe_clean_staging_dir(staging_dir)

    print("Packaging BaCon trainer code (model/bacon.py, modal parity)...")
    b64_bundle = _build_code_bundle_b64(repo_root, target_dataset=dataset_name, target_imb=imb_ratio)
    if len(b64_bundle) > 900_000:
        print(f"[WARN] bundle b64 is {len(b64_bundle)} chars — may exceed Kaggle's 1MB kernel file limit")

    # Kaggle dataset sources attached to the kernel
    dataset_list = []
    for slug in args.dataset_slug.split(","):
        slug = slug.strip()
        if slug and slug not in dataset_list:
            dataset_list.append(slug)
    if args.extra_datasets:
        for slug in args.extra_datasets.split(","):
            slug = slug.strip()
            if slug and slug not in dataset_list:
                dataset_list.append(slug)
    if dataset_name == "cifar100":
        for slug in reversed(["minhtmnguyntrn/cifar100-gcd", "fedesoriano/cifar100"]):
            if slug in dataset_list:
                dataset_list.remove(slug)
            dataset_list.insert(0, slug)
    if dataset_name == "imagenet100":
        for slug in reversed(["minhtmnguyntrn/imagenet100-cmc"]):
            if slug in dataset_list:
                dataset_list.remove(slug)
            dataset_list.insert(0, slug)

    # runner command line (executed inside the kernel after unpacking)
    runner_args = [
        "kaggle_run_bacon.py",
        "--dataset-name", dataset_name,
        "--config", args.config,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--backbone", args.backbone,
        "--vis-freq", str(args.vis_freq),
        "--save-freq", str(args.save_freq),
        "--time-budget-hours", str(args.time_budget_hours),
        "--early-stop-patience", str(args.early_stop_patience),
        "--run-name", exp_name,
        "--imb-ratio", str(imb_ratio if imb_ratio is not None else (100 if is_inat else 20)),
    ]
    if args.lr is not None:
        runner_args += ["--lr", str(args.lr)]
    if args.seed is not None and args.seed >= 0:
        runner_args += ["--seed", str(args.seed)]
    if args.fp16:
        runner_args += ["--fp16"]
    if args.use_logit_adjust:
        runner_args += ["--use-logit-adjust"]
    if args.use_momentum_teacher:
        runner_args += ["--use-momentum-teacher", "--teacher-m0", str(args.teacher_m0)]
    if args.resume_ckpt:
        runner_args += ["--resume-ckpt", args.resume_ckpt]
    if args.extra_args:
        runner_args += ["--extra-args", args.extra_args]

    run_script_content = f'''# Self-contained BaCon runner for Kaggle GPU (modal_train.py parity)
import base64
import io
import json
import os
import subprocess
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import zipfile

# 1. Unpack the BaCon code bundle into the current working directory (/kaggle/working)
BUNDLE_B64 = """{b64_bundle}"""
target_dir = os.getcwd()
print(f"[Kaggle Bootstrap] Unpacking BaCon code into {{target_dir}}...")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64.encode("ascii")))) as z:
    z.extractall(target_dir)

print("[Kaggle Bootstrap] Code unpacked. Top-level files:")
for item in sorted(os.listdir(target_dir)):
    print(" -", item)
sys.stdout.flush()

# 2. Run the BaCon trainer (model/bacon.py) with modal_train.py-equivalent arguments
cmd = [sys.executable] + {json.dumps(runner_args)}

print("\\n[Kaggle Runner] Launching:\\n" + " ".join(cmd) + "\\n")
sys.stdout.flush()
process = subprocess.run(cmd)
sys.exit(process.returncode)
'''
    with open(staging_dir / "run_script.py", "w", encoding="utf-8") as f:
        f.write(run_script_content)

    kernel_title = exp_name if len(exp_name) <= 50 else exp_name[:50]
    metadata = {
        "id": f"{kaggle_user}/{kernel_slug}",
        "title": kernel_title,
        "code_file": "run_script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": dataset_list,
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(staging_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 65)
    print("        BACON KAGGLE CLI PUSH LAUNCHER (modal_train.py parity)")
    print("=" * 65)
    print(f"Active Account:  {kaggle_user}")
    print(f"Kernel ID:       {kaggle_user}/{kernel_slug}")
    print(f"Experiment:      {exp_name}")
    print(f"Baseline:        {args.baseline.upper()} (informational — BaCon trainer always used)")
    print(f"Dataset:         {dataset_name.upper()} (imb={imb_ratio if imb_ratio is not None else 'natural'})")
    print(f"Config Ablation: {args.config}  (modal launch() mapping)")
    print(f"Trainer:         python -m model.bacon  (current repo code, same as Modal)")
    print(f"GPU:             Kaggle T4x2 / P100 (enable_gpu: true; add --fp16 or lower --batch-size if OOM)")
    print(f"Time budget:     {args.time_budget_hours}h | save-freq {args.save_freq} | vis-freq {args.vis_freq} | workers {args.num_workers}")
    print(f"Code Payload:    {len(b64_bundle)} chars (~{len(b64_bundle)//1024} KB self-extracting)")
    print("=" * 65)

    # Validate locally: let the runner build (and print) the exact command it will run
    print("\n[Dry-run validation] runner builds the exact command:")
    validation = subprocess.run(
        [sys.executable, str(repo_root / "kaggle_run_bacon.py"), *runner_args[1:], "--dry-run"],
        cwd=str(repo_root), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = (validation.stdout or "") + (validation.stderr or "")
    print(out.strip() if out.strip() else "(no output)")
    if validation.returncode != 0:
        print("\n[ERROR] runner validation failed — aborting (fix the error above).")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN] Kernel packaged successfully at:", staging_dir)
        return

    print("\nPushing kernel to Kaggle...")
    sys.stdout.flush()
    cmd_push = ["kaggle", "kernels", "push", "-p", str(staging_dir)]
    res = subprocess.run(cmd_push, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(res.stderr.strip())

    combined_output = (res.stdout or "") + (res.stderr or "")
    if res.returncode != 0 and ("409" in combined_output or "already in use" in combined_output.lower()):
        print("\n[INFO] Detected 409 title conflict (soft-deleted notebook). Retrying with unique timestamp suffix...")
        import time
        ts_suffix = str(int(time.time()))[-4:]
        metadata["title"] = f"BaCon {exp_name} {ts_suffix}"
        metadata["id"] = f"{kaggle_user}/{kernel_slug}-{ts_suffix}"
        with open(staging_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        res = subprocess.run(cmd_push, capture_output=True, text=True)
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr:
            print(res.stderr.strip())
        combined_output = (res.stdout or "") + (res.stderr or "")

    if res.returncode == 0 and "Kernel push error:" not in combined_output:
        print("\n" + "=" * 65)
        print("  >>> SUCCESS! Kernel pushed — background training started <<<")
        print(f"  Trainer: model/bacon.py (current repo code, same as Modal)")
        print(f"  Status : kaggle kernels status {kaggle_user}/{kernel_slug}")
        print(f"  Results: kaggle kernels output {kaggle_user}/{kernel_slug} -p <dir>")
        print(f"           -> outputs/experiments/{exp_name}/ (log, final_report, checkpoints)")
        print("  Resume : attach the finished kernel's output as a dataset, then")
        print("           --resume-ckpt /kaggle/input/<dataset>/outputs/experiments/")
        print(f"                  {exp_name}/checkpoints/model_best.pt")
        print("=" * 65)
    else:
        print("\n" + "=" * 65)
        print("  >>> PUSH FAILED! Kaggle rejected the kernel push <<<")
        print("  Reason: check the error above (e.g. GPU quota or credentials).")
        print("=" * 65)


if __name__ == "__main__":
    main()
