"""Helper tool to download checkpoints from Kaggle kernels across AloGPU accounts
and optionally push them to Modal volume 'bacon-storage'.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_OUT_DIR = r"D:\data\checkpoints_bacon"
ALOGPU_ACCOUNTS_DIR = os.path.expanduser(r"~\AppData\Roaming\com.alogpu.app\kaggle\accounts")
KAGGLE_TOKEN_DST = os.path.expanduser(r"~\.kaggle\access_token")

TASKS = [
    {
        "num": 1,
        "account": "tam11",
        "slug": "sanjivinsmoker/bacon-cifar100-base-imb20",
        "dataset": "cifar100",
        "config": "Base",
        "imb": 20,
        "short_name": "cifar100_base_imb20",
    },
    {
        "num": 2,
        "account": "tam12",
        "slug": "tonytonychopperr/bacon-cifar100-full-imb20",
        "dataset": "cifar100",
        "config": "Full",
        "imb": 20,
        "short_name": "cifar100_full_imb20",
    },
    {
        "num": 3,
        "account": "tam13",
        "slug": "robinico/bacon-cifar100-base-imb50",
        "dataset": "cifar100",
        "config": "Base",
        "imb": 50,
        "short_name": "cifar100_base_imb50",
    },
    {
        "num": 4,
        "account": "tam14",
        "slug": "frankyshipwright/bacon-cifar100-full-imb50",
        "dataset": "cifar100",
        "config": "Full",
        "imb": 50,
        "short_name": "cifar100_full_imb50",
    },
    {
        "num": 5,
        "account": "tam16",
        "slug": "jinbehelmsman/bacon-cifar100-base-imb100",
        "dataset": "cifar100",
        "config": "Base",
        "imb": 100,
        "short_name": "cifar100_base_imb100",
    },
    {
        "num": 6,
        "account": "tam17",
        "slug": "mihawkdracule/bacon-cifar100-full-imb100",
        "dataset": "cifar100",
        "config": "Full",
        "imb": 100,
        "short_name": "cifar100_full_imb100",
    },
    {
        "num": 7,
        "account": "tam11",
        "slug": "sanjivinsmoker/bacon-imagenet100-base-imb20",
        "dataset": "imagenet100",
        "config": "Base",
        "imb": 20,
        "short_name": "imagenet100_base_imb20",
    },
    {
        "num": 8,
        "account": "tam12",
        "slug": "tonytonychopperr/bacon-imagenet100-pseudo-imb20",
        "dataset": "imagenet100",
        "config": "Pseudo",
        "imb": 20,
        "short_name": "imagenet100_pseudo_imb20",
    },
    {
        "num": 9,
        "account": "tam13",
        "slug": "robinico/bacon-imagenet100-prototype-imb20",
        "dataset": "imagenet100",
        "config": "Prototype",
        "imb": 20,
        "short_name": "imagenet100_prototype_imb20",
    },
    {
        "num": 10,
        "account": "tam14",
        "slug": "frankyshipwright/bacon-imagenet100-prototype-tailaware-imb20",
        "dataset": "imagenet100",
        "config": "Prototype_Tailaware",
        "imb": 20,
        "short_name": "imagenet100_prototype_tailaware_imb20",
    },
    {
        "num": 11,
        "account": "tam16",
        "slug": "jinbehelmsman/bacon-imagenet100-full-wo-tail-imb20",
        "dataset": "imagenet100",
        "config": "Full_wo_Tail",
        "imb": 20,
        "short_name": "imagenet100_full_wo_tail_imb20",
    },
    {
        "num": 12,
        "account": "tam17",
        "slug": "mihawkdracule/bacon-imagenet100-full-imb20",
        "dataset": "imagenet100",
        "config": "Full",
        "imb": 20,
        "short_name": "imagenet100_full_imb20",
    },
]


def switch_kaggle_token(account: str) -> bool:
    src = os.path.join(ALOGPU_ACCOUNTS_DIR, account, "token")
    if not os.path.isfile(src):
        print(f"[ERROR] Cannot find token file for account '{account}' at {src}")
        return False
    os.makedirs(os.path.dirname(KAGGLE_TOKEN_DST), exist_ok=True)
    shutil.copyfile(src, KAGGLE_TOKEN_DST)
    print(f"[AUTH] Successfully switched Kaggle token to account '{account}'")
    return True


def download_kernel_output(slug: str, target_dir: str) -> bool:
    os.makedirs(target_dir, exist_ok=True)
    print(f"[DOWNLOAD] Downloading output of '{slug}' to {target_dir} ...")
    cmd = ["kaggle", "kernels", "output", slug, "-p", target_dir]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Download failed: {res.stderr.strip() or res.stdout.strip()}")
        return False
    print("[DOWNLOAD] Completed successfully.")
    return True


def clean_and_collect_ckpts(target_dir: str, short_name: str) -> dict:
    """Finds model_last.pt and best checkpoint, copies them to root of target_dir,
    and removes non-checkpoint bloated files (e.g. data_uq_idxs, source code)."""
    ckpts = glob.glob(os.path.join(target_dir, "**", "*.pt"), recursive=True)
    found = {}
    for p in ckpts:
        base = os.path.basename(p)
        if base == "model_last.pt":
            found["last"] = p
        elif "model_epoch" in base or "model_best" in base:
            found[base] = p

    # Keep organized copies
    last_ckpt = found.get("last")
    res_paths = {}
    if last_ckpt:
        dest_last = os.path.join(target_dir, f"{short_name}_last.pt")
        if last_ckpt != dest_last:
            shutil.copyfile(last_ckpt, dest_last)
        res_paths["last"] = dest_last
        print(f"  -> Extracted Last Checkpoint: {dest_last} ({os.path.getsize(dest_last)/(1024*1024):.1f} MB)")

    # Clean up non-pt debris if outputs/ exists
    outputs_subdir = os.path.join(target_dir, "outputs")
    data_subdir = os.path.join(target_dir, "data")
    data_uq_subdir = os.path.join(target_dir, "data_uq_idxs")
    for d in [data_subdir, data_uq_subdir]:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)

    return res_paths


def push_to_modal(local_path: str, remote_filename: str) -> bool:
    print(f"[MODAL] Uploading {local_path} -> modal volume 'bacon-storage' (checkpoints/{remote_filename}) ...")
    cmd = ["modal", "volume", "put", "bacon-storage", local_path, f"checkpoints/{remote_filename}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Modal upload failed: {res.stderr.strip() or res.stdout.strip()}")
        return False
    print(f"[MODAL] Uploaded successfully: /bacon-storage/checkpoints/{remote_filename}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle checkpoints and push to Modal")
    parser.add_argument("--tasks", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                        help="Task numbers to download (1-6: CIFAR-100, 7-12: ImageNet-100)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Local directory to store checkpoints")
    parser.add_argument("--push-modal", action="store_true", help="Automatically push downloaded checkpoints to Modal")
    args = parser.parse_args()

    selected = [t for t in TASKS if t["num"] in args.tasks]
    if not selected:
        print(f"No matching tasks found for {args.tasks}. Available: 1..12")
        return

    print("=" * 70)
    print(f"CHECKPOINT SYNC TOOL: Kaggle -> Local ({args.out_dir}) -> Modal")
    print("=" * 70)

    downloaded = []
    for t in selected:
        print(f"\n--- Processing Task {t['num']}: {t['slug']} (Account: {t['account']}) ---")
        if not switch_kaggle_token(t["account"]):
            continue
        task_dir = os.path.join(args.out_dir, t["short_name"])
        if download_kernel_output(t["slug"], task_dir):
            res_ckpts = clean_and_collect_ckpts(task_dir, t["short_name"])
            if "last" in res_ckpts:
                t["local_last"] = res_ckpts["last"]
                t["modal_name"] = f"{t['short_name']}_last.pt"
                downloaded.append(t)
                if args.push_modal:
                    push_to_modal(t["local_last"], t["modal_name"])

    print("\n" + "=" * 70)
    print("SUMMARY & READY-TO-RUN COMMANDS")
    print("=" * 70)
    for t in downloaded:
        print(f"\n[Task {t['num']}] {t['dataset'].upper()} - {t['config']} (imb{t['imb']}):")
        print(f"  Local file: {t.get('local_last')}")
        remote_path = f"/bacon-storage/checkpoints/{t['modal_name']}"
        print(f"  Modal volume path: {remote_path}")
        print("  1. Lệnh đẩy lên Modal volume (nếu chưa đẩy tự động):")
        print(f"     modal volume put bacon-storage \"{t.get('local_last')}\" checkpoints/{t['modal_name']}")
        print("  2. Lệnh chạy tiếp sức (resume) trên Modal A100 GPU:")
        print(f"     modal run -d modal_train.py::launch --dataset-name {t['dataset']} --imb-ratio {t['imb']} --config {t['config']} --epochs 200 --resume-ckpt \"{remote_path}\"")


if __name__ == "__main__":
    main()
