"""Upload & extract ImageNet-100 CMC zip parts directly on Modal Volume 'bacon-storage'.

Usage:
  Step 1 (upload 10 zips from local PC to Modal volume):
    modal volume put bacon-storage "D:\\data\\kaggle_datasets\\imagenet100_cmc" data/imagenet_zips

  Step 2 (extract zips on Modal CPU -> /bacon-storage/data/imagenet/train & val):
    modal run scripts/modal_extract_imagenet.py
"""

import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

STORAGE_DIR = "/bacon-storage"
ZIPS_DIR = f"{STORAGE_DIR}/data/imagenet_zips"
TARGET_DIR = f"{STORAGE_DIR}/data/imagenet"

app = modal.App("bacon-extract-imagenet100")
storage_volume = modal.Volume.from_name("bacon-storage", create_if_missing=True)


def _extract_one(zp: Path, target_dir: Path) -> tuple[str, int]:
    print(f"[EXTRACT] Start: {zp.name} ...", flush=True)
    with zipfile.ZipFile(zp, "r") as zf:
        members = zf.namelist()
        zf.extractall(target_dir)
    print(f"[EXTRACT] Done:  {zp.name} ({len(members)} entries)", flush=True)
    return zp.name, len(members)


@app.function(
    image=modal.Image.debian_slim(python_version="3.10"),
    cpu=4.0,
    memory=8192,
    timeout=3600,
    volumes={STORAGE_DIR: storage_volume},
)
def extract_imagenet100(keep_zips: bool = False) -> None:
    storage_volume.reload()
    zips_root = Path(ZIPS_DIR)
    target_dir = Path(TARGET_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(zips_root.rglob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(
            f"No .zip files found under {ZIPS_DIR}. "
            f"Run: modal volume put bacon-storage \"D:\\data\\kaggle_datasets\\imagenet100_cmc\" data/imagenet_zips"
        )

    print(f"[INFO] Found {len(zip_files)} zip file(s) under {ZIPS_DIR}:", flush=True)
    for z in zip_files:
        print(f"  - {z} ({z.stat().st_size / (1024**2):.1f} MB)", flush=True)

    total_entries = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, cnt in pool.map(lambda p: _extract_one(p, target_dir), zip_files):
            total_entries += cnt

    # Verify the exact sample file that ImageNet-100 LT loader expects
    train_dir = target_dir / "train"
    val_dir = target_dir / "val"
    sample = train_dir / "n02869837" / "n02869837_000001.jpg"
    n_train_cls = sum(1 for p in train_dir.iterdir() if p.is_dir()) if train_dir.exists() else 0
    n_val_cls = sum(1 for p in val_dir.iterdir() if p.is_dir()) if val_dir.exists() else 0

    print("\n" + "=" * 60, flush=True)
    print(f"[VERIFY] Extracted {total_entries} total entries into {TARGET_DIR}", flush=True)
    print(f"[VERIFY] train classes: {n_train_cls} | val classes: {n_val_cls}", flush=True)
    print(f"[VERIFY] Sample check ({sample}): {'OK' if sample.is_file() else 'MISSING'}", flush=True)
    print("=" * 60, flush=True)

    if not sample.is_file():
        raise RuntimeError(f"Extraction finished but expected file is missing: {sample}")

    if not keep_zips:
        print(f"[CLEANUP] Removing uploaded zip folder {ZIPS_DIR} to free ~8.5GB...", flush=True)
        shutil.rmtree(zips_root, ignore_errors=True)

    print("[VOLUME] Committing bacon-storage volume...", flush=True)
    storage_volume.commit()
    print("[DONE] ImageNet-100 is ready on Modal at /bacon-storage/data/imagenet!", flush=True)


@app.local_entrypoint()
def main(keep_zips: bool = False):
    extract_imagenet100.remote(keep_zips=keep_zips)
