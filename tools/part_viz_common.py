"""Shared post-hoc part-visualization helpers (CPU-friendly, no training).

- load backbone arch + weights from a post-5.3.0 checkpoint (cl_backbone
  preferred, ce_backbone fallback) together with part_module / part_bank.
- part_forward(): backbone blocks -> norm -> LatentPartModule attention.
- CUB keypoint helpers: tolerant part_locs parser + original-image ->
  model-input coordinate mapping for the standard Resize/CenterCrop pipeline.

Heavy imports (model.*, data.*) live inside functions so unit tests can import
the pure helpers without datasets or hub downloads.
"""

import os

import torch


# ---------------------------------------------------------------------------
# Checkpoint / model loading
# ---------------------------------------------------------------------------

def load_backbone_and_parts(ckpt_path, backbone_name='dinov2_vitb14', device='cpu'):
    """Returns (backbone, part_module, part_bank, meta dict) on device, eval mode."""
    from model.backbone import load_backbone
    from model.part_modules import LatentPartModule, PartPrototypeBank

    ckpt = torch.load(ckpt_path, map_location='cpu')
    backbone = load_backbone(backbone_name, pretrained=False)
    loaded_from = None
    for key in ('cl_backbone', 'ce_backbone'):
        if ckpt.get(key) is not None:
            backbone.load_state_dict(ckpt[key], strict=True)
            loaded_from = key
            break
    if loaded_from is None:
        raise KeyError(f'{ckpt_path}: no cl_backbone/ce_backbone found.')
    sd_m, sd_b = ckpt.get('part_module'), ckpt.get('part_bank')
    if sd_m is None or sd_b is None:
        raise KeyError(
            f'{ckpt_path}: missing part_module/part_bank — pre-5.3.0 checkpoint. '
            'Re-run training on current code (5.3.0 saves parts).')
    M, d = sd_m['queries'].shape
    C = sd_b['prototypes'].shape[0]
    part_module = LatentPartModule(dim=int(d), num_slots=int(M))
    part_module.load_state_dict(sd_m, strict=True)
    part_bank = PartPrototypeBank(num_classes=int(C), num_slots=int(M), dim=int(d))
    part_bank.load_state_dict(sd_b, strict=True)
    backbone.to(device).eval()
    part_module.to(device).eval()
    part_bank.to(device).eval()
    meta = {'M': int(M), 'd': int(d), 'C': int(C),
            'weights_from': loaded_from,
            'ckpt_backbone': ckpt.get('args_backbone', 'unknown')}
    return backbone, part_module, part_bank, meta


def part_forward(backbone, part_module, images):
    """images: (B,3,H,W) transformed. Returns r_norm (B,M,d), A (B,M,N)."""
    import torch.nn.functional as F  # noqa: F401 (kept for parity with train path)
    from model.backbone import iter_blocks, prepare_tokens
    with torch.no_grad():
        x = prepare_tokens(backbone, images)
        for blk in iter_blocks(backbone):
            x = blk(x)
        x = backbone.norm(x)
        n_reg = int(getattr(backbone, 'num_register_tokens', 0) or 0)
        patches = x[:, 1 + n_reg:]
        r_norm, A = part_module(patches)
    return r_norm, A


# ---------------------------------------------------------------------------
# Test transform + CUB test loader (CUB only; extend per dataset as needed)
# ---------------------------------------------------------------------------

def imagenet_test_transform(image_size=224, interpolation=3, crop_pct=0.875):
    from data.augmentations import get_transform
    import argparse
    _, test_transform = get_transform(
        'imagenet', image_size=image_size,
        args=argparse.Namespace(interpolation=interpolation, crop_pct=crop_pct))
    return test_transform


def cub_test_subset(cub_root, classes, transform, max_per_class=30, seed=0):
    """CUB test images of the requested classes. Returns list of
    (image_tensor, label, img_path)."""
    import numpy as np
    from data.cub import CUBDataset
    ds = CUBDataset(root=cub_root, train=False, transform=transform)
    rng = np.random.RandomState(seed)
    by_cls = {}
    for i, t in enumerate(ds.targets):
        t = int(t)
        if t in set(classes):
            by_cls.setdefault(t, []).append(i)
    items = []
    for c in sorted(by_cls):
        idxs = by_cls[c]
        if len(idxs) > max_per_class:
            idxs = sorted(rng.choice(idxs, size=max_per_class, replace=False).tolist())
        for i in idxs:
            img, t, _ = ds[i]
            items.append((img, int(t), ds.img_path[i]))
    return items


# ---------------------------------------------------------------------------
# CUB keypoint helpers
# ---------------------------------------------------------------------------

CUB_PART_NAMES = ['beak', 'forehead', 'crown', 'throat', 'breast', 'belly',
                  'back', 'wing', 'tail', 'leg', 'nape', 'eye', 'bill',
                  'throat2', 'leg2']
# NOTE: canonical CUB has 15 parts; names above are positional placeholders —
# the hit-rate math only needs integer part_ids, names are for display.


def parse_part_locs(path):
    """Tolerant parser for CUB parts/part_locs.txt.

    Accepts whitespace- or comma-separated rows of
    <image_id> <part_id> <x> <y> <visible>.
    Returns {(image_id, part_id): (x, y, visible)}.
    Raises FileNotFoundError with a helpful message when missing.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'{path} not found. CUB part annotations live under '
            '<CUB_ROOT>/parts/part_locs.txt in the official release.')
    locs = {}
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip().replace(',', ' ')
            if not line:
                continue
            toks = line.split()
            if len(toks) < 5:
                continue
            try:
                iid, pid = int(float(toks[0])), int(float(toks[1]))
                x, y, v = float(toks[2]), float(toks[3]), int(float(toks[4]))
            except ValueError:
                continue
            locs[(iid, pid)] = (x, y, v)
    if not locs:
        raise ValueError(f'{path}: parsed 0 keypoints — unexpected format. '
                         'Expected rows of <image_id> <part_id> <x> <y> <visible>.')
    return locs


def map_keypoint_to_input(x, y, W0, H0, image_size=224, crop_pct=0.875):
    """Map an original-image keypoint into model-input (crop) coordinates.

    Mirrors torchvision Resize(small_edge=rs) + CenterCrop(image_size) with
    rs = int(image_size / crop_pct). Returns (xi, yi, inside_bool).
    """
    rs = int(image_size / crop_pct)
    scale = rs / min(W0, H0)
    W1, H1 = W0 * scale, H0 * scale
    left, top = (W1 - image_size) / 2.0, (H1 - image_size) / 2.0
    xi, yi = x * scale - left, y * scale - top
    inside = (0 <= xi < image_size) and (0 <= yi < image_size)
    return xi, yi, inside


def patch_grid_centers(h, image_size=224):
    """(h*h, 2) cell centers in input-pixel coords, row-major (matches A layout)."""
    import numpy as np
    step = image_size / h
    ax = (np.arange(h) + 0.5) * step
    yy, xx = np.meshgrid(ax, ax, indexing='ij')
    return np.stack([xx.ravel(), yy.ravel()], axis=1)
