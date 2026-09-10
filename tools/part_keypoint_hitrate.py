"""Part-keypoint hit-rate (viec 5.1.4) — quantitative prototype grounding.

CUB-200-2011 ships 15 part keypoints per image (parts/part_locs.txt). For each
slot m and keypoint p, measures the attention mass of m within radius r of p;
slot->part assignment by majority vote; hit = assigned-part mass >= 0.5.
Reports per-slot hit-rate + overall + random-area baseline (pi*r^2).

Needs: post-5.3.0 checkpoint, CUB data WITH parts/ dir, backbone weights.
CPU-friendly (default 500 images).

Usage:
    python tools/part_keypoint_hitrate.py --ckpt <model.pt> \\
        --cub-root D:/data/CUB_200_2011 --backbone dinov2_vitb14 \\
        --max-images 500 --radius 0.1 --out dev_outputs/keypoint_hitrate.csv
"""

import argparse
import csv
import os
import sys


def compute_hit_stats(attn_masses, thresh=0.5):
    """attn_masses: {(slot, part): [mass per image]} -> per-slot report.

    Assignment: slot -> part with max mean mass. Hit: mass >= thresh.
    Returns {slot: {'part': pid, 'hit_rate': h, 'n': n}} + 'overall' + 'random'.
    """
    import numpy as np
    slots = sorted({s for s, _ in attn_masses})
    parts = sorted({p for _, p in attn_masses})
    report, hits, total = {}, [], 0
    for s in slots:
        means = {p: float(np.mean(attn_masses.get((s, p), [0.0]))) for p in parts}
        pid = max(means, key=means.get)
        arr = np.asarray(attn_masses.get((s, pid), []), dtype=float)
        h = float((arr >= thresh).mean()) if len(arr) else 0.0
        report[s] = {'part': pid, 'hit_rate': round(h, 4), 'n': int(len(arr))}
        hits.extend((arr >= thresh).tolist())
        total += len(arr)
    report['overall'] = {'hit_rate': round(float(np.mean(hits)), 4) if hits else 0.0,
                         'n': int(total)}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cub-root', default=None)
    ap.add_argument('--backbone', default='dinov2_vitb14')
    ap.add_argument('--max-images', type=int, default=500)
    ap.add_argument('--radius', type=float, default=0.1,
                    help='fraction of input side for hit radius')
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--crop-pct', type=float, default=0.875)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.part_viz_common import (imagenet_test_transform,
                                       load_backbone_and_parts, map_keypoint_to_input,
                                       parse_part_locs, patch_grid_centers)

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cub_root = a.cub_root or os.environ.get('CUB_ROOT', 'D:/data/CUB_200_2011')
    locs = parse_part_locs(os.path.join(cub_root, 'parts', 'part_locs.txt'))

    backbone, part_module, part_bank, meta = load_backbone_and_parts(
        a.ckpt, a.backbone, device)
    M = meta['M']
    print(f'[keypoint] M={M} d={meta["d"]} weights_from={meta["weights_from"]}')

    # image_id -> path: CUBDataset ordering is sorted image_id
    from data.cub import CUBDataset
    ds = CUBDataset(root=cub_root, train=False, transform=None)
    transform = imagenet_test_transform(a.image_size, 3, a.crop_pct)
    assert len(ds.img_path) > 0, 'empty CUB test set — check --cub-root'

    rng = np.random.RandomState(a.seed)
    order = rng.permutation(len(ds.img_path))[:a.max_images]

    masses = {}
    n_used = n_skipped = 0
    backbone_h = None
    with torch.no_grad():
        for start in range(0, len(order), a.batch_size):
            chunk_idx = order[start:start + a.batch_size]
            batch, info = [], []
            for gi in chunk_idx:
                iid = int(ds.uq_idxs[gi])  # = image_id - 1  (see data/cub.py)
                with open(ds.img_path[gi], 'rb') as f:
                    pil = Image.open(f).convert('RGB')
                W0, H0 = pil.size
                batch.append(transform(pil))
                info.append((iid, W0, H0))
            from tools.part_viz_common import part_forward
            r_norm, A = part_forward(backbone, part_module,
                                     torch.stack(batch).to(device))
            A = A.cpu().numpy()
            if backbone_h is None:
                import math
                backbone_h = int(round(math.sqrt(A.shape[2])))
                assert backbone_h * backbone_h == A.shape[2], 'non-square A'
                centers = patch_grid_centers(backbone_h, a.image_size)
            for b, (iid, W0, H0) in enumerate(info):
                for pid in range(1, 16):
                    loc = locs.get((iid + 1, pid)) or locs.get((iid, pid))
                    if loc is None:
                        continue
                    x, y, v = loc
                    if not v:
                        continue
                    xi, yi, inside = map_keypoint_to_input(
                        x, y, W0, H0, a.image_size, a.crop_pct)
                    if not inside:
                        continue
                    d2 = ((centers - np.array([xi, yi])) ** 2).sum(axis=1)
                    cell = d2 <= (a.radius * a.image_size) ** 2
                    if not cell.any():
                        continue
                    for m in range(M):
                        masses.setdefault((m, pid), []).append(
                            float(A[b, m][cell].sum()))
                    n_used += 1
                n_skipped += 0
    print(f'[keypoint] scored pairs: {n_used} (images={len(order)})')
    if not masses:
        print('[keypoint] no scored pairs — check parts files / cub-root.')
        return 1
    report = compute_hit_stats(masses)
    rand = round(float(np.pi * a.radius ** 2), 4)
    print(f'[keypoint] random-area baseline ~ {rand}')
    for s in sorted(k for k in report if isinstance(k, int)):
        r = report[s]
        print(f"  slot {s}: best part {r['part']} hit-rate {r['hit_rate']} (n={r['n']})")
    print(f"  OVERALL hit-rate {report['overall']['hit_rate']} (n={report['overall']['n']})")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['slot', 'best_part', 'hit_rate', 'n', 'random_baseline'])
        for s in sorted(k for k in report if isinstance(k, int)):
            r = report[s]
            w.writerow([s, r['part'], r['hit_rate'], r['n'], rand])
        w.writerow(['overall', '-', report['overall']['hit_rate'],
                    report['overall']['n'], rand])
    print(f'[keypoint] saved {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
