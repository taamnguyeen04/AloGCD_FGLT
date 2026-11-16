"""
Downloader and Kaggle Uploader for ImageNet100-CMC from Hugging Face (asafaa/imagenet100-cmc).

Features:
- Resumable downloads (Range requests) with chunked writing.
- Validates file sizes against remote Content-Length.
- Copies adapted CMC split files into target dataset.
- Generates Kaggle dataset metadata and uploads via Kaggle CLI.
"""

import os
import sys
import time
import json
import shutil
import urllib.request
import argparse
from pathlib import Path

HF_BASE_URL = "https://huggingface.co/datasets/asafaa/imagenet100-cmc/resolve/main"

FILES = [
    # Metadata
    "Labels.json",
    "selected_synsets.txt",
    "stats.json",
    # Val splits (total ~340 MB)
    "imagenet100_cmc_val_part01.zip",
    "imagenet100_cmc_val_part02.zip",
    "imagenet100_cmc_val_part03.zip",
    "imagenet100_cmc_val_part04.zip",
    "imagenet100_cmc_val_part05.zip",
    # Train splits (total ~7.7 GB)
    "imagenet100_cmc_train_part01.zip",
    "imagenet100_cmc_train_part02.zip",
    "imagenet100_cmc_train_part03.zip",
    "imagenet100_cmc_train_part04.zip",
    "imagenet100_cmc_train_part05.zip",
]


def download_file_with_resume(url: str, dest_path: Path, max_retries: int = 5) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    # Get remote size
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            remote_size = int(resp.headers.get("Content-Length", 0))
    except Exception as exc:
        print(f"  [Error] Failed to get remote header for {url}: {exc}")
        return False

    if dest_path.exists():
        local_size = dest_path.stat().st_size
        if remote_size > 0 and local_size == remote_size:
            print(f"  [Skip] {dest_path.name} already fully downloaded ({local_size / 1e6:.2f} MB)")
            return True

    for attempt in range(1, max_retries + 1):
        resume_byte = 0
        if temp_path.exists():
            resume_byte = temp_path.stat().st_size

        if remote_size > 0 and resume_byte >= remote_size:
            temp_path.replace(dest_path)
            return True

        headers = {"User-Agent": "Mozilla/5.0"}
        if resume_byte > 0:
            headers["Range"] = f"bytes={resume_byte}-"
            mode = "ab"
            print(f"  [Resume] {dest_path.name} from {resume_byte / 1e6:.2f} MB / {remote_size / 1e6:.2f} MB (attempt {attempt})")
        else:
            mode = "wb"
            print(f"  [Download] {dest_path.name} ({remote_size / 1e6:.2f} MB, attempt {attempt})")

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(temp_path, mode) as f:
                    chunk_size = 1024 * 1024  # 1MB
                    last_log = time.time()
                    downloaded = resume_byte
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_log >= 5.0 or (remote_size > 0 and downloaded >= remote_size):
                            pct = (downloaded / remote_size * 100) if remote_size > 0 else 0
                            print(f"    Progress: {downloaded / 1e6:.1f} MB / {remote_size / 1e6:.1f} MB ({pct:.1f}%)")
                            last_log = now

            if remote_size == 0 or temp_path.stat().st_size == remote_size:
                temp_path.replace(dest_path)
                print(f"  [Done] {dest_path.name} successfully saved.")
                return True
        except Exception as exc:
            print(f"  [Warning] Attempt {attempt} error: {exc}. Retrying in 5s...")
            time.sleep(5)

    return False


def get_active_kaggle_user() -> str | None:
    try:
        import kaggle
        kaggle.api.authenticate()
        return kaggle.api.get_config_value("username")
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Download and upload ImageNet100-CMC dataset")
    parser.add_argument("--dest", default=r"D:\data\kaggle_datasets\imagenet100_cmc",
                        help="Target local download directory")
    parser.add_argument("--slug", default=None,
                        help="Kaggle dataset slug (<username>/<dataset_name>)")
    parser.add_argument("--upload", action="store_true", default=False,
                        help="Upload to Kaggle after download completes")
    parser.add_argument("--val-only", action="store_true", default=False,
                        help="Download val splits and metadata only (~350 MB)")
    args = parser.parse_args()

    active_user = get_active_kaggle_user() or "minhtmnguyntrn"
    if args.slug is None:
        args.slug = f"{active_user}/imagenet100-cmc"
    elif "/" in args.slug and active_user:
        # If user passed a slug with a different username, warn and align with active user
        dname = args.slug.split("/")[-1]
        args.slug = f"{active_user}/{dname}"

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = [f for f in FILES if not args.val_only or "train" not in f]

    print("=================================================================")
    print("      IMAGENET100-CMC DOWNLOADER & KAGGLE UPLOADER")
    print(f"Destination:  {dest_dir}")
    print(f"Total files:  {len(files_to_download)}")
    print(f"Kaggle slug:  {args.slug}")
    print("=================================================================")

    start_time = time.time()
    for idx, filename in enumerate(files_to_download, 1):
        url = f"{HF_BASE_URL}/{filename}"
        dest_file = dest_dir / filename
        print(f"[{idx}/{len(files_to_download)}] Checking {filename}...")
        ok = download_file_with_resume(url, dest_file)
        if not ok:
            print(f"Failed to download {filename}. Aborting.")
            sys.exit(1)

    # Copy adapted split files if they exist in repo
    repo_root = Path(__file__).resolve().parent.parent
    splits_dir = repo_root / "splits" / "imagenet100"
    for split_name in ["imageNet_100_LT_train_cmc.txt", "ImageNet_100_test_cmc.txt"]:
        src_split = splits_dir / split_name
        if src_split.is_file():
            shutil.copy2(src_split, dest_dir / split_name)
            print(f"Copied {split_name} into dataset directory.")

    # Write Kaggle dataset metadata
    meta = {
        "title": args.slug.split("/")[-1],
        "id": args.slug,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(dest_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Generated dataset-metadata.json for {args.slug}")

    elapsed = time.time() - start_time
    print(f"\nAll files successfully downloaded in {elapsed / 60:.2f} minutes!")

    if args.upload:
        print("\nStarting Kaggle CLI upload...")
        import subprocess
        cmd = ["kaggle", "datasets", "create", "-p", str(dest_dir), "--dir-mode", "zip", "--public"]
        print("Running:", " ".join(cmd))
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print(f"Dataset uploaded successfully! Check https://www.kaggle.com/datasets/{args.slug}")
        else:
            print("Kaggle upload returned exit code:", res.returncode)


if __name__ == "__main__":
    main()
