"""Backbone factory + offline-first loading + compat helpers for DINOv1 and DINOv2.

Features adapted from Prototail:
- Offline-first: loads directly from local torch hub cache, environment overrides
  (``BACON_DINO_REPO`` / ``PROTOTAIL_DINO_REPO``, ``BACON_DINO_WEIGHTS`` / ``PROTOTAIL_DINO_WEIGHTS``),
  or Kaggle dataset mounts (``/kaggle/input``) before touching the network.
- SSL Certificate fallback: works around CDN certificate expiration (dl.fbaipublicfiles.com)
  with robust pre-fetching into ``<torch.hub.get_dir()>/checkpoints/``.
- Robust remote retry loop (5 attempts with exponential backoff) with skip_validation=True.
- Full compatibility with BaCon's token slicing, AdaPart gating, and fine-tuning.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

SUPPORTED_BACKBONES = ("dino_vitb16", "dinov2_vitb14", "dinov2_vitb14_reg")

_HUB_REPO = {
    "dino_vitb16": ("facebookresearch/dino:main", "dino_vitb16"),
    "dinov2_vitb14": ("facebookresearch/dinov2:main", "dinov2_vitb14"),
    "dinov2_vitb14_reg": ("facebookresearch/dinov2:main", "dinov2_vitb14_reg"),
}

_BACKBONE_SPECS = {
    "dino_vitb16": {"feat_dim": 768, "feature_dim": 768, "patch_size": 16, "image_size": 224},
    "dinov2_vitb14": {"feat_dim": 768, "feature_dim": 768, "patch_size": 14, "image_size": 224},
    "dinov2_vitb14_reg": {"feat_dim": 768, "feature_dim": 768, "patch_size": 14, "image_size": 224},
}

_CHECKPOINT_FILES = {
    "dino_vitb16": "dino_vitb16_pretrain.pth",
    "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
    "dinov2_vitb14_reg": "dinov2_vitb14_reg4_pretrain.pth",
}

_WEIGHT_URLS = {
    "dinov2_vitb14": (
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/"
        "dinov2_vitb14_pretrain.pth"
    ),
    "dinov2_vitb14_reg": (
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/"
        "dinov2_vitb14_reg4_pretrain.pth"
    ),
    "dino_vitb16": (
        "https://dl.fbaipublicfiles.com/dino/dino_vitb16/dino_vitb16_pretrain/"
        "dino_vitb16_pretrain.pth"
    ),
}

_REPO_ENV_VARS = ("BACON_DINO_REPO", "PROTOTAIL_DINO_REPO", "DINO_REPO")
_WEIGHTS_ENV_VARS = ("BACON_DINO_WEIGHTS", "PROTOTAIL_DINO_WEIGHTS", "DINO_WEIGHTS")

_KAGGLE_SCAN_ROOTS: tuple[Path, ...] = (
    Path("/kaggle/input"),
    Path("/kaggle/working"),
)
_SCAN_MAX_DEPTH = 3
_SCAN_DIR_LIMIT = 4000

_OFFLINE_FIX_INSTRUCTIONS = """No usable network route to the checkpoint host (DNS unreachable or certificates rejected) and no offline DINO backbone assets were found.
Fix (pick one):
  1. Kaggle notebook: Settings -> Internet -> ON (requires a phone-verified account).
  2. Attach an offline dataset with the DINOv2 repo + weights:
       python scripts/prepare_dinov2_offline.py --output dinov2_offline
     then upload the `dinov2_offline` folder as a Kaggle dataset and add it
     to the notebook (it is mounted under /kaggle/input and found automatically).
  3. Point env overrides at local copies:
       BACON_DINO_REPO=/path/to/dinov2-repo   (dir with hubconf.py)
       BACON_DINO_WEIGHTS=/path/to/dinov2_vitb14_pretrain.pth"""


def _hubconf_mentions(hubconf_path: Path, entrypoint: str) -> bool:
    try:
        text = hubconf_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return entrypoint in text[:16384]


def _walk_limited(root: Path, max_depth: int = _SCAN_MAX_DEPTH) -> Iterable[tuple[Path, list[str], list[str]]]:
    base_depth = len(root.parts)
    visited = 0
    for current, dirnames, filenames in os.walk(root):
        depth = len(Path(current).parts) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        visited += 1
        if visited > _SCAN_DIR_LIMIT:
            return
        yield Path(current), dirnames, filenames


def find_local_repo(
    name: str,
    *,
    hub_dir: Path | None = None,
    extra_roots: Sequence[Path] = (),
) -> Path | None:
    if name not in _HUB_REPO:
        return None
    repository, entrypoint = _HUB_REPO[name]

    explicit: list[Path] = []
    for var in _REPO_ENV_VARS:
        env_val = os.environ.get(var)
        if env_val:
            explicit.append(Path(env_val))

    hub_dir = hub_dir if hub_dir is not None else Path(torch.hub.get_dir())
    explicit.append(hub_dir / repository.replace("/", "_").replace(":", "_"))
    explicit.append(hub_dir / "facebookresearch_dinov2_main")
    explicit.append(hub_dir / "facebookresearch_dino_main")

    for candidate in explicit:
        hubconf = candidate / "hubconf.py"
        if hubconf.is_file() and _hubconf_mentions(hubconf, entrypoint):
            return candidate

    for root in (*extra_roots, *_KAGGLE_SCAN_ROOTS):
        if not root.is_dir():
            continue
        for current, _dirnames, filenames in _walk_limited(root):
            if "hubconf.py" not in filenames:
                continue
            hubconf = current / "hubconf.py"
            if _hubconf_mentions(hubconf, entrypoint):
                return current
    return None


def _compatible_weights_file(filename: str, name: str) -> bool:
    stem = Path(filename).stem
    is_reg_variant = "_reg" in stem
    if name.endswith("_reg") != is_reg_variant:
        return False
    return stem.startswith(name)


def _find_weights_in_root(root: Path, name: str, expected: str) -> Path | None:
    exact: Path | None = None
    fuzzy: Path | None = None
    for current, _dirnames, filenames in _walk_limited(root):
        for filename in filenames:
            if not filename.endswith((".pth", ".pt", ".bin")):
                continue
            if filename == expected:
                return current / filename
            if fuzzy is None and _compatible_weights_file(filename, name):
                fuzzy = current / filename
    return fuzzy


def ensure_offline_checkpoint(
    name: str,
    pretrained: bool = True,
    *,
    hub_dir: Path | None = None,
    repo_dir: Path | None = None,
    extra_roots: Sequence[Path] = (),
) -> Path | None:
    if not pretrained:
        return None
    expected = _CHECKPOINT_FILES.get(name)
    if expected is None:
        return None

    hub_dir = hub_dir if hub_dir is not None else Path(torch.hub.get_dir())
    destination = hub_dir / "checkpoints" / expected
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    source: Path | None = None
    for var in _WEIGHTS_ENV_VARS:
        env_val = os.environ.get(var)
        if env_val and Path(env_val).is_file():
            source = Path(env_val)
            break

    if source is None and repo_dir is not None:
        for candidate in (
            repo_dir / "checkpoints" / expected,
            repo_dir / expected,
            repo_dir.parent / "checkpoints" / expected,
            repo_dir.parent / expected,
        ):
            if candidate.is_file():
                source = candidate
                break

    if source is None:
        for root in (*extra_roots, *_KAGGLE_SCAN_ROOTS):
            if not root.is_dir():
                continue
            source = _find_weights_in_root(root, name, expected)
            if source is not None:
                break

    if source is None:
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not (destination.is_file() and destination.stat().st_size == source.stat().st_size):
        staging = destination.with_name(destination.name + ".part")
        shutil.copyfile(source, staging)
        os.replace(staging, destination)
    return destination


def locate_offline_assets(
    name: str = "dinov2_vitb14",
    pretrained: bool = True,
) -> tuple[Path | None, Path | None]:
    repo_dir = find_local_repo(name)
    checkpoint = ensure_offline_checkpoint(
        name, pretrained=pretrained, repo_dir=repo_dir
    )
    return repo_dir, checkpoint


def probe_internet(timeout: float = 4.0) -> bool:
    try:
        socket.create_connection(("github.com", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def _open_url_certificate_fallback(url: str, timeout: float = 60.0):
    try:
        return urllib.request.urlopen(url, timeout=timeout)
    except (urllib.error.URLError, ssl.SSLError) as error:
        reason = getattr(error, "reason", error)
        if not isinstance(reason, ssl.SSLError):
            raise
        print(
            f"[DINO Backbone Loader] Certificate verification failed for checkpoint host ({reason}); "
            "retrying WITHOUT certificate verification for this file only.",
            flush=True,
        )
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return urllib.request.urlopen(url, timeout=timeout, context=context)


def fetch_weights_from_url(
    name: str,
    pretrained: bool = True,
    *,
    hub_dir: Path | None = None,
    timeout: float = 60.0,
) -> Path | None:
    if not pretrained:
        return None
    expected = _CHECKPOINT_FILES.get(name)
    url = _WEIGHT_URLS.get(name)
    if expected is None or url is None:
        return None
    hub_dir = hub_dir if hub_dir is not None else Path(torch.hub.get_dir())
    destination = hub_dir / "checkpoints" / expected
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    staging = destination.with_name(destination.name + ".part")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _open_url_certificate_fallback(url, timeout=timeout) as response, \
                staging.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        os.replace(staging, destination)
        size_mb = destination.stat().st_size / 1e6
        print(
            f"[DINO Backbone Loader] Fetched checkpoint {expected} ({size_mb:.0f} MB) into hub cache.",
            flush=True,
        )
        return destination
    except Exception as error:
        try:
            if staging.exists():
                staging.unlink()
        except OSError:
            pass
        print(
            f"[DINO Backbone Loader] Checkpoint download failed: {error}",
            flush=True,
        )
        return None


def load_backbone(
    name: str = "dinov2_vitb14",
    pretrained: bool = True,
    *,
    extra_roots: Sequence[Path] = (),
) -> Any:
    """Load a ViT backbone with offline-first resolution, cert fallback, and retry."""
    if name not in _HUB_REPO:
        raise ValueError(
            f"Unknown backbone {name!r}; choose from {SUPPORTED_BACKBONES}"
        )

    repository, entrypoint = _HUB_REPO[name]

    # 1. Offline-first: repo from env override / hub cache / Kaggle dataset mounts
    repo_dir = find_local_repo(name, extra_roots=extra_roots)
    checkpoint = ensure_offline_checkpoint(
        name, pretrained=pretrained, repo_dir=repo_dir, extra_roots=extra_roots
    )
    if pretrained and checkpoint is None:
        checkpoint = fetch_weights_from_url(name, pretrained=pretrained)

    if repo_dir is not None:
        if pretrained and checkpoint is None and not probe_internet():
            raise RuntimeError(
                f"Offline checkpoint for {name!r} not found next to {repo_dir}.\n"
                + _OFFLINE_FIX_INSTRUCTIONS
            )
        try:
            kwargs = {"pretrained": pretrained} if name.startswith("dinov2_") else {}
            source_note = str(checkpoint) if checkpoint else "weights not pre-cached"
            print(
                f"[DINO Backbone Loader] Loading {name} offline from {repo_dir} (checkpoint: {source_note})",
                flush=True,
            )
            return torch.hub.load(
                str(repo_dir), entrypoint, source="local", **kwargs
            )
        except Exception as error:
            if not probe_internet():
                raise RuntimeError(
                    f"Failed to load {name!r} offline from {repo_dir}: {error}\n"
                    + _OFFLINE_FIX_INSTRUCTIONS
                ) from error
            print(
                f"[DINO Backbone Loader] Offline repo load failed ({error}); falling back to hub download.",
                flush=True,
            )

    # 2. No local repo and no Internet: fail fast with actionable instructions
    if repo_dir is None and not probe_internet():
        raise RuntimeError(
            f"Cannot load backbone {name!r} from {repository!r}: "
            "no local copy found and Internet is unreachable.\n"
            + _OFFLINE_FIX_INSTRUCTIONS
        )

    # 3. Remote load with retry logic and skip_validation=True
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            kwargs = {
                "skip_validation": True,
                "trust_repo": True,
            }
            if name.startswith("dinov2_"):
                kwargs["pretrained"] = pretrained
            return torch.hub.load(repository, entrypoint, **kwargs)
        except Exception as error:
            last_error = error
            delay = attempt * 3
            print(
                f"[DINO Backbone Loader] Hub load attempt {attempt}/5 failed: {error}. "
                f"Retrying in {delay}s...",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to load backbone {name!r} from {repository!r} after 5 attempts.\n"
        + _OFFLINE_FIX_INSTRUCTIONS
    ) from last_error


def backbone_spec(name: str) -> dict[str, int]:
    try:
        return dict(_BACKBONE_SPECS[name])
    except KeyError as error:
        raise ValueError(f"Unknown backbone {name!r}. Choose from {SUPPORTED_BACKBONES}") from error


def prepare_tokens(backbone: Any, images: torch.Tensor) -> torch.Tensor:
    """Call the right token-prep entrypoint for v1/v2 backbones."""
    if hasattr(backbone, "prepare_tokens_with_masks"):
        return backbone.prepare_tokens_with_masks(images)
    return backbone.prepare_tokens(images)


def iter_blocks(backbone: Any):
    """Yield transformer blocks in order, flattening BlockChunk wrappers."""
    for blk in backbone.blocks:
        if isinstance(blk, torch.nn.ModuleList) and not hasattr(blk, "norm1"):
            for sub in blk:
                if isinstance(sub, torch.nn.Identity):
                    continue
                yield sub
        else:
            yield blk


def num_blocks(backbone: Any) -> int:
    return sum(1 for _ in iter_blocks(backbone))


def num_register_tokens(backbone: Any) -> int:
    return int(getattr(backbone, "num_register_tokens", 0) or 0)


def split_cls_patches(x: torch.Tensor, backbone: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Split (B, 1 + n_reg + N, d) into CLS (B, d) and patches (B, N, d)."""
    n_reg = num_register_tokens(backbone)
    cls = x[:, 0]
    patches = x[:, 1 + n_reg:]
    return cls, patches


def forward_frozen_prefix(backbone: Any, images: torch.Tensor, grad_from_block: int = 11) -> torch.Tensor:
    """Run prepare_tokens + frozen blocks[0:grad_from_block]."""
    x = prepare_tokens(backbone, images)
    for i, blk in enumerate(iter_blocks(backbone)):
        if i < grad_from_block:
            x = blk(x)
    return x


def forward_blocks_from(x: torch.Tensor, backbone: Any, start: int = 0) -> torch.Tensor:
    """Run blocks[start:] in order, chaining outputs correctly."""
    out = x
    for i, blk in enumerate(iter_blocks(backbone)):
        if i >= start:
            out = blk(out)
    return out


def forward_backbone_tokens(backbone: Any, images: torch.Tensor, grad_from_block: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full forward up to norm, returning (cls, patches, full_tokens)."""
    x = prepare_tokens(backbone, images)
    for blk in iter_blocks(backbone):
        x = blk(x)
    x = backbone.norm(x)
    cls, patches = split_cls_patches(x, backbone)
    return cls, patches, x


_BLOCK_IDX_RE = re.compile(r"\.(\d+)(?=\.|$)")


def parse_block_index(param_name: str) -> int | None:
    """Return the transformer block index for a param name, or None."""
    if "block" not in param_name:
        return None
    matches = _BLOCK_IDX_RE.findall(param_name)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def is_late_block_param(param_name: str, late_index: int = 11) -> bool:
    """True if param belongs to block ``late_index`` (v1/v2 compatible)."""
    if f"block.{late_index}" in param_name or f"blocks.{late_index}" in param_name:
        return True
    return parse_block_index(param_name) == late_index


def set_finetune_blocks(backbone: Any, grad_from_block: int = 11) -> None:
    """Freeze everything, unfreeze blocks >= grad_from_block."""
    for p in backbone.parameters():
        p.requires_grad = False
    for name, p in backbone.named_parameters():
        idx = parse_block_index(name)
        if idx is not None and idx >= grad_from_block:
            p.requires_grad = True
