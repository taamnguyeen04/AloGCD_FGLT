"""Rank CUB classes by how well the checkpoint clusters them.

Score per class = KMeans+Hungarian per-class accuracy + silhouette (part-space).
Dùng để chọn cụm đẹp đi viz thay vì đoán mò.

Usage:
    python tools/select_good_clusters.py --ckpt model_epoch68.pt --backbone dinov2_vitb14 --max-per-class 30 --top-k 8 --out dev_outputs/good_clusters.csv
"""
import argparse
import csv
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--backbone', default='dinov2_vitb14')
    ap.add_argument('--cub-root', default=None)
    ap.add_argument('--num-labeled', type=int, default=100)
    ap.add_argument('--max-per-class', type=int, default=30)
    ap.add_argument('--top-k', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.part_viz_common import (
        cub_test_subset, imagenet_test_transform,
        load_backbone_and_parts, part_forward,
    )

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cub_root = a.cub_root or os.environ.get('CUB_ROOT', 'D:/data/CUB_200_2011')
    classes = list(range(200))

    backbone, part_module, _, meta = load_backbone_and_parts(a.ckpt, a.backbone, device)
    print(f"[rank] M={meta['M']} weights_from={meta['weights_from']}")
    transform = imagenet_test_transform()
    items = cub_test_subset(cub_root, classes, transform, a.max_per_class, a.seed)
    print(f'[rank] {len(items)} images.')

    feats, labels = [], []
    with torch.no_grad():
        for i in range(0, len(items), 32):
            chunk = items[i:i + 32]
            imgs = torch.stack([im for im, _, _ in chunk]).to(device)
            r_norm, _ = part_forward(backbone, part_module, imgs)
            pool = r_norm.mean(dim=1)
            feats.append(torch.nn.functional.normalize(pool, dim=-1).cpu())
            labels.extend([t for _, t, _ in chunk])
    feats = torch.cat(feats).numpy()
    labels = np.array(labels)

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples
    from scipy.optimize import linear_sum_assignment

    K = len(np.unique(labels))
    pred = KMeans(n_clusters=K, random_state=a.seed, n_init=10).fit_predict(feats)
    # Hungarian map pred -> true
    conf = np.zeros((K, K), dtype=np.int64)
    uniq = sorted(np.unique(labels))
    idx_of = {c: i for i, c in enumerate(uniq)}
    np.add.at(conf, (pred, [idx_of[t] for t in labels]), 1)
    r, c = linear_sum_assignment(conf.max() - conf)
    pmap = dict(zip(r, [uniq[j] for j in c]))
    mapped = np.array([pmap[p] for p in pred])
    sil = silhouette_samples(feats, labels)

    rows = []
    for cl in uniq:
        m = labels == cl
        acc = float((mapped[m] == cl).mean()) if m.sum() else 0.0
        s = float(sil[m].mean()) if m.sum() else -1.0
        rows.append((cl, round(acc * 100, 2), round(s, 4), int(m.sum()),
                     'known' if cl < a.num_labeled else 'novel'))
    rows.sort(key=lambda r: (r[1], r[2]), reverse=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['class', 'acc', 'silhouette', 'n', 'split'])
        w.writerows(rows)
    print(f'[rank] saved {a.out}')
    print(f"--- TOP {a.top_k} ---")
    for cl, acc, s, n, sp in rows[:a.top_k]:
        print(f'  class {cl:3d} [{sp:5s}] acc={acc:5.1f} sil={s:.3f} n={n}')
    print('--- BOTTOM 5 ---')
    for cl, acc, s, n, sp in rows[-5:]:
        print(f'  class {cl:3d} [{sp:5s}] acc={acc:5.1f} sil={s:.3f} n={n}')
    print('[rank] NEXT classes: ' + ','.join(str(r[0]) for r in rows[:a.top_k]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
