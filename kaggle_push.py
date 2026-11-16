"""Kaggle Kernel Push Launcher.

Automatically packages prototail code as a self-contained bundle,
creates kernel-metadata.json, and pushes to Kaggle to trigger GPU T4
background training (Save & Run All).
You can shut down your PC immediately after running this script.
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
from datetime import datetime
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


def _build_code_bundle_b64(repo_root: Path, target_dataset: str = "", target_imb: int | str = "") -> str:
    """Zip kaggle_train.py, src/, configs/, and dataset-specific splits into an in-memory base64 payload."""
    buf = io.BytesIO()
    prototail_candidates = [
        repo_root,
        Path(r"C:\Users\tam\Documents\GitHub\gcd-hosts\prototail"),
        repo_root.parent / "gcd-hosts" / "prototail",
        repo_root.parent / "prototail",
    ]
    p_root = None
    for c in prototail_candidates:
        if (c / "src/prototail_gcd/train.py").is_file():
            p_root = c
            break
    if p_root is None:
        raise FileNotFoundError("Could not find prototail source root directory")

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        # 1. kaggle_train.py
        z.write(p_root / "kaggle_train.py", "kaggle_train.py")
        # 2. src/
        for root, dirs, files in os.walk(p_root / "src"):
            if "__pycache__" in root:
                continue
            for f in files:
                p = Path(root) / f
                z.write(p, p.relative_to(p_root))
        # 3. configs/
        for root, dirs, files in os.walk(p_root / "configs"):
            if "__pycache__" in root:
                continue
            for f in files:
                p = Path(root) / f
                z.write(p, p.relative_to(p_root))

        # 4. data_uq_idxs/ (pack only the target dataset's splits to stay under Kaggle's 1MB limit)
        splits_candidates = [
            p_root / "data_uq_idxs",
            repo_root / "data_uq_idxs",
            Path(r"C:\Users\tam\Documents\GitHub\gcd-hosts\prototail\data_uq_idxs"),
            Path(r"C:\Users\tam\Documents\GitHub\BaCon\data_uq_idxs"),
        ]
        s_dir = None
        for cand in splits_candidates:
            if cand.is_dir():
                s_dir = cand
                break
        if s_dir is not None:
            dname_lower = (target_dataset or "").lower()
            prefixes = [dname_lower] if dname_lower else []
            if "cub" in dname_lower:
                prefixes.extend(["cub", "cub200"])
            elif "car" in dname_lower:
                prefixes.extend(["car", "cars196", "cars"])
            elif "air" in dname_lower:
                prefixes.extend(["air", "aircraft"])
            elif "herb" in dname_lower:
                prefixes.extend(["herb", "herbarium19", "herbarium"])
            elif "inat" in dname_lower:
                prefixes.extend(["inat", "inat2018"])
            elif "cifar100" in dname_lower:
                prefixes.append("cifar100")
            elif "cifar10" in dname_lower:
                prefixes.append("cifar10")

            for child in s_dir.iterdir():
                if not child.is_dir():
                    continue
                cname = child.name.lower()
                if prefixes and not any(cname.startswith(p) for p in prefixes):
                    continue
                if target_imb and ("inat" in cname or "herbarium" in cname):
                    imb_tag = "natural" if str(target_imb).lower() == "natural" else f"imb{target_imb}"
                    if imb_tag not in cname:
                        continue
                for root, dirs, files in os.walk(child):
                    if "__pycache__" in root:
                        continue
                    for f in files:
                        p = Path(root) / f
                        arcname = "data_uq_idxs/" + p.relative_to(s_dir).as_posix()
                        z.write(p, arcname)

        # 4b. splits/ (pack splits/imagenet100 when targeting imagenet)
        dname_lower = (target_dataset or "").lower()
        if "imagenet" in dname_lower:
            splits_dir = None
            for cand_sp in [p_root / "splits", repo_root / "splits", Path(r"C:\Users\tam\Documents\GitHub\BaCon\splits")]:
                if cand_sp.is_dir():
                    splits_dir = cand_sp
                    break
            if splits_dir is not None:
                for root, dirs, files in os.walk(splits_dir):
                    if "__pycache__" in root:
                        continue
                    for f in files:
                        p = Path(root) / f
                        arcname = "splits/" + p.relative_to(splits_dir).as_posix()
                        z.write(p, arcname)

    return base64.b64encode(buf.getvalue()).decode("ascii")


def _safe_clean_staging_dir(staging_dir: Path) -> None:
    """Safely clean and create staging directory, resilient to Windows file handles."""
    if staging_dir.exists():
        try:
            shutil.rmtree(staging_dir, ignore_errors=True)
        except Exception:
            pass
    # If the directory still exists (e.g. held by an active handle or indexing), clean remaining children
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
    parser = argparse.ArgumentParser(description="Push ProtoTail training job to Kaggle GPU")
    parser.add_argument("--baseline", choices=["simgcd", "bacon"], default="simgcd")
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", choices=["cub", "cub200", "cars", "car", "scars", "aircraft", "air", "fgvc_aircraft", "herbarium", "herbarium19", "herb19", "inat", "inat2018", "inaturalist", "inaturalist2018", "cifar100", "cifar-100", "cifar10", "cifar-10", "imagenet100", "imagenet-100", "imagenet"], default="cub")
    parser.add_argument("--imb-ratio", "--imb_ratio", dest="imb_ratio", type=str, default=None, help="Imbalance ratio (optional for inat2018)")
    parser.add_argument(
        "--config",
        choices=["Base", "Pseudo", "Prototype", "Prototype_Tailaware", "Full_wo_Tail", "Full"],
        default="Base",
    )
    parser.add_argument("--kmeans-backend", choices=["sklearn_cpu", "faiss_gpu"], default="sklearn_cpu", help="KMeans clustering backend (default: sklearn_cpu)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-interval", "--eval_interval", dest="eval_interval", type=int, default=5)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128)
    parser.add_argument("--kaggle-user", "--kaggle_user", dest="kaggle_user", type=str, default=None)
    parser.add_argument("--dataset-slug", "--dataset_slug", dest="dataset_slug", type=str, default="nguyntrnminhtm/gcd-data")
    parser.add_argument("--extra-datasets", "--extra_datasets", dest="extra_datasets", type=str, default=None,
                        help="Comma-separated additional Kaggle dataset slugs (e.g. nguyeentaam/dinov2-offline-bundle)")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Build kernel locally without pushing")
    args = parser.parse_args()
    name_map = {
        "car": "cars",
        "scars": "cars",
        "air": "aircraft",
        "fgvc_aircraft": "aircraft",
        "cub200": "cub",
        "herbarium": "herbarium19",
        "herb19": "herbarium19",
        "inat": "inat2018",
        "inat2018": "inat2018",
        "inaturalist": "inat2018",
        "inaturalist2018": "inat2018",
        "cifar-100": "cifar100",
        "cifar-10": "cifar10",
        "imagenet-100": "imagenet100",
        "imagenet": "imagenet100",
    }
    args.dataset_name = name_map.get(args.dataset_name.lower(), args.dataset_name.lower())

    repo_root = Path(__file__).resolve().parent
    kaggle_user = args.kaggle_user or _get_active_kaggle_username()

    # Determine flags based on config
    flags = []
    if args.config == "Base":
        pass
    elif args.config == "Pseudo":
        flags.extend(["--enable-pseudo-labeling", "--enable-novel-pseudo"])
    elif args.config == "Prototype":
        flags.extend(["--use-parts", "--ablate-adaptive-capacity"])
    elif args.config == "Prototype_Tailaware":
        flags.extend(["--use-parts", "--gate-warm-start-from-support"])
    elif args.config == "Full_wo_Tail":
        flags.extend(["--use-parts", "--ablate-adaptive-capacity", "--enable-pseudo-labeling", "--enable-novel-pseudo"])
    elif args.config == "Full":
        flags.extend(["--use-parts", "--enable-pseudo-labeling", "--enable-novel-pseudo", "--gate-warm-start-from-support"])

    is_natural = (args.dataset_name in ("inat", "inat2018", "inaturalist", "inaturalist2018")) and (args.imb_ratio is None or str(args.imb_ratio).lower() in ("natural", "none", "0", ""))
    if is_natural:
        args.imb_ratio = None
        exp_name = f"{args.baseline}_{args.dataset_name}_{args.config}"
        kernel_slug = f"prototail-{args.baseline}-{args.dataset_name}-{args.config.lower()}".replace("_", "-")
    else:
        if args.imb_ratio is None:
            args.imb_ratio = "20"
        exp_name = f"{args.baseline}_{args.dataset_name}_{args.config}_imb{args.imb_ratio}"
        kernel_slug = f"prototail-{args.baseline}-{args.dataset_name}-{args.config.lower()}-imb{args.imb_ratio}".replace("_", "-")

    # Staging folder for kernel push
    staging_dir = repo_root / "build_kaggle_kernel"
    _safe_clean_staging_dir(staging_dir)

    # Build self-contained code bundle
    print("Packaging ProtoTail codebase into self-contained bundle...")
    b64_bundle = _build_code_bundle_b64(repo_root, target_dataset=args.dataset_name, target_imb=args.imb_ratio)

    # Generate self-unpacking runner script for Kaggle
    # Collect all dataset sources
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
    if args.dataset_name in ("cifar100", "cifar-100"):
        for slug in reversed(["minhtmnguyntrn/cifar100-gcd", "fedesoriano/cifar100"]):
            if slug in dataset_list:
                dataset_list.remove(slug)
            dataset_list.insert(0, slug)
    if args.dataset_name in ("imagenet100", "imagenet-100", "imagenet"):
        for slug in reversed(["minhtmnguyntrn/imagenet100-cmc"]):
            if slug in dataset_list:
                dataset_list.remove(slug)
            dataset_list.insert(0, slug)
    primary_data_slug = dataset_list[0] if dataset_list else args.dataset_slug

    flag_str = " ".join(flags)
    run_script_content = f'''# Self-contained ProtoTail runner for Kaggle GPU
import base64
import io
import os
import subprocess
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import zipfile

# 1. Unpack code bundle directly into current working directory
BUNDLE_B64 = """{b64_bundle}"""
target_dir = os.getcwd()
print(f"[Kaggle Bootstrap] Unpacking ProtoTail code into {{target_dir}}...")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64.encode("ascii")))) as z:
    z.extractall(target_dir)

print("[Kaggle Bootstrap] Code unpacked successfully. Available files:")
for item in os.listdir(target_dir):
    print(" -", item)
sys.stdout.flush()

# 2. Execute training
cmd = [
    sys.executable,
    os.path.join(target_dir, "kaggle_train.py"),
    "--baseline", "{args.baseline}",
    "--dataset-slug", "{primary_data_slug}",
    "--dataset-name", "{args.dataset_name}",
    "--imb-ratio", "{args.imb_ratio}",
    "--epochs", "{args.epochs}",
    "--eval-interval", "{args.eval_interval}",
    "--batch-size", "{args.batch_size}",
    "--run-name", "{exp_name}",
    "--kmeans-backend", "{args.kmeans_backend}",
]
extra_flags = {flags}
cmd.extend(extra_flags)

print("\\n[Kaggle Runner] Launching command:\\n" + " ".join(cmd) + "\\n")
sys.stdout.flush()

process = subprocess.run(cmd)
sys.exit(process.returncode)
'''
    with open(staging_dir / "run_script.py", "w", encoding="utf-8") as f:
        f.write(run_script_content)

    # Create kernel-metadata.json (Kaggle API enforces title <= 50 characters)
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
    print("        PROTOTAIL KAGGLE CLI PUSH LAUNCHER")
    print("=" * 65)
    print(f"Active Account:  {kaggle_user}")
    print(f"Kernel ID:       {kaggle_user}/{kernel_slug}")
    print(f"Configuration:   {exp_name}")
    print(f"Baseline:        {args.baseline.upper()}")
    print(f"Dataset:         {args.dataset_name.upper()} (imb={args.imb_ratio})")
    print(f"Config Ablation: {args.config}")
    print(f"Flags:           {flag_str or '(none - pure baseline)'}")
    print(f"GPU:             NVIDIA T4 (enable_gpu: true)")
    print(f"Code Payload:    {len(b64_bundle)} chars (~{len(b64_bundle)//1024} KB self-extracting)")
    print("=" * 65)
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
        metadata["title"] = f"ProtoTail {exp_name} {ts_suffix}"
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
        print("  >>> SUCCESS! Kernel successfully pushed to Kaggle GPU <<<")
        print(f"  Background training on T4 GPU for: {kaggle_user}")
        print(f"  Status check: kaggle kernels status {kaggle_user}/{kernel_slug}")
        print("=" * 65)
    else:
        print("\n" + "=" * 65)
        print(f"  >>> PUSH FAILED! Kaggle rejected the kernel push <<<")
        print(f"  Reason: Check the error above (e.g. GPU session quota or credentials).")
        print("=" * 65)


if __name__ == "__main__":
    main()
