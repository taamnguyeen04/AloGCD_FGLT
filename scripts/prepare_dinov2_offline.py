"""Prepare an offline DINOv2 bundle for Kaggle datasets.

Output layout (upload this folder as one Kaggle dataset):

    <output>/
        dinov2_repo/                      # hub repo with hubconf.py at the top
        checkpoints/dinov2_vitb14_pretrain.pth

Sources, in order of preference:
  1. The local torch hub cache (repo: <hub>/facebookresearch_dinov2_main,
     weights: <hub>/checkpoints/dinov2_vitb14_pretrain.pth) — instant, no download.
  2. Direct download from GitHub (repo zipball) and dl.fbaipublicfiles.com (weights).

Then on Kaggle: attach the dataset to the notebook; ``load_backbone`` finds it
automatically by scanning /kaggle/input for hubconf.py + the checkpoint file.

Usage:
    python scripts/prepare_dinov2_offline.py --output dinov2_offline
    python scripts/prepare_dinov2_offline.py --output dinov2_offline --reg
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import torch

REPO_ZIPBALL_URLS = {
    "dinov2_vitb14": "https://github.com/facebookresearch/dinov2/zipball/main",
    "dinov2_vitb14_reg": "https://github.com/facebookresearch/dinov2/zipball/main",
    "dino_vitb16": "https://github.com/facebookresearch/dino/zipball/main",
}
CHECKPOINT_URLS = {
    "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    "dinov2_vitb14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    "dino_vitb16": "https://dl.fbaipublicfiles.com/dino/dino_vitb16/dino_vitb16_pretrain/dino_vitb16_pretrain.pth",
}
CACHE_REPO_DIRS = {
    "dinov2_vitb14": "facebookresearch_dinov2_main",
    "dinov2_vitb14_reg": "facebookresearch_dinov2_main",
    "dino_vitb16": "facebookresearch_dino_main",
}
_SKIP_PARTS = {"__pycache__", ".github", ".git", "docs", "notebooks", "scripts", "tests"}


def _copy_repo_from_cache(backbone: str, target: Path) -> Path | None:
    hub_dir = Path(torch.hub.get_dir())
    cached = hub_dir / CACHE_REPO_DIRS[backbone]
    if not (cached / "hubconf.py").is_file():
        return None
    print(f"[Prepare] Copying repo from torch hub cache: {cached}")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        cached,
        target,
        ignore=shutil.ignore_patterns(*_SKIP_PARTS),
    )
    return target


def _download_repo(backbone: str, target: Path) -> Path:
    url = REPO_ZIPBALL_URLS[backbone]
    print(f"[Prepare] Downloading repo zipball: {url}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "repo.zip"
        urllib.request.urlretrieve(url, archive)
        extract_dir = Path(tmp) / "extracted"
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extract_dir)
        inner = [item for item in extract_dir.iterdir() if item.is_dir()]
        if len(inner) != 1:
            raise RuntimeError(f"Unexpected zipball layout under {extract_dir}")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            inner[0],
            target,
            ignore=shutil.ignore_patterns(*_SKIP_PARTS),
        )
    return target


def _stage_checkpoint(backbone: str, destination: Path) -> Path:
    hub_dir = Path(torch.hub.get_dir())
    cached = hub_dir / "checkpoints" / Path(CHECKPOINT_URLS[backbone]).name
    if cached.is_file():
        print(f"[Prepare] Using checkpoint from torch hub cache: {cached}")
        shutil.copyfile(cached, destination)
        return destination
    url = CHECKPOINT_URLS[backbone]
    print(f"[Prepare] Downloading checkpoint: {url}")
    urllib.request.urlretrieve(url, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dinov2_offline"))
    parser.add_argument(
        "--backbone",
        choices=tuple(CHECKPOINT_URLS),
        default="dinov2_vitb14",
        help="Which backbone bundle to prepare",
    )
    args = parser.parse_args()

    backbone = args.backbone
    if backbone == "dinov2_vitb14_reg":
        # The reg variant shares the repo with the base model; keep a single bundle.
        backbone = "dinov2_vitb14"
        print("[Prepare] reg variant: sharing the dinov2 repo bundle with the base model")

    repo_target = args.output / "dinov2_repo"
    checkpoint_target = args.output / "checkpoints" / Path(CHECKPOINT_URLS[backbone]).name
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)

    repo = _copy_repo_from_cache(backbone, repo_target)
    if repo is None:
        repo = _download_repo(backbone, repo_target)
    _stage_checkpoint(backbone, checkpoint_target)

    size_mb = checkpoint_target.stat().st_size / 1e6
    print("=" * 60)
    print(f"Offline bundle ready at: {args.output.resolve()}")
    print(f"  repo:       {repo}")
    print(f"  checkpoint: {checkpoint_target} ({size_mb:.0f} MB)")
    print("Upload this folder as a Kaggle dataset (or run:")
    print(f"  kaggle datasets create -p {args.output})")
    print("then attach it to the notebook — load_backbone finds it automatically.")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
