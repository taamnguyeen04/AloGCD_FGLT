"""
Generate long-tailed index splits for CUB-200 in the same format BaCon's CIFAR
loaders consume (data_uq_idxs/<name>/{l_k,unl_k,unl_unk}_uq_idxs.pt).

Semantics (mirrors the shipped CIFAR-100 files, verified empirically):
  - The whole train half of CUB is subsampled to an exponential LT profile
    (BaCon's get_lt_dist: img_max * imb_ratio ** (i / (C-1))), with a
    minimum-count floor so tail classes never drop below --min-per-class.
  - Classes are randomly permuted before assigning profile ranks, so known /
    unknown class identity is not correlated with original label order.
  - First K classes of the permuted ranking = known (old); rest = unknown (new).
  - Known classes are split ~50/50 labeled / unlabeled per class.
    Unknown classes go entirely to the unlabeled pool.

Index space: positions into CUBDataset(train=True) ordering, i.e. sorted image_id - 1
(same convention as data/cub.py uq_idxs).
"""
import argparse
import os
from collections import Counter

import numpy as np
import torch

from config import cub_root


def read_cub_train_targets(root):
    labels = {}
    with open(os.path.join(root, 'image_class_labels.txt')) as f:
        for line in f:
            k, v = line.strip().split()
            labels[int(k)] = int(v) - 1
    split = {}
    with open(os.path.join(root, 'train_test_split.txt')) as f:
        for line in f:
            k, v = line.strip().split()
            split[int(k)] = int(v)
    targets = []
    for image_id in sorted(labels):
        if split[image_id] == 1:
            targets.append(labels[image_id])
    return np.array(targets)  # position i <-> uq_idx i


def get_lt_dist(cls_num, img_max, imb_factor):
    """BaCon's exponential profile (data/cifar.py:106-111)."""
    return [int(img_max * (imb_factor ** (i / (cls_num - 1.0)))) for i in range(cls_num)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=cub_root)
    p.add_argument('--num-known', type=int, default=100)
    p.add_argument('--total-classes', type=int, default=200)
    p.add_argument('--imb-ratio', type=int, required=True)
    p.add_argument('--img-max', type=int, default=30, help='max imgs/class; CUB train has 29-30')
    p.add_argument('--min-per-class', type=int, default=5,
                   help='profile floor; CUB train only has ~30/class so rho>=100 would zero out tails')
    p.add_argument('--anno-ratio', type=float, default=50.0, help='%% of known-class imgs that are labeled')
    p.add_argument('--seed', type=int, default=416)
    p.add_argument('--out-root', default='data_uq_idxs')
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    targets = read_cub_train_targets(args.root)
    n_total_cls = args.total_classes

    # Per-class available indices
    cls_to_idxs = {c: np.where(targets == c)[0] for c in range(n_total_cls)}

    # Random permutation of classes -> profile rank assignment
    perm = rng.permutation(n_total_cls)
    counts = {}
    for rank, c in enumerate(perm):
        want = max(args.min_per_class,
                   min(get_lt_dist(n_total_cls, args.img_max, 1.0 / args.imb_ratio)[rank],
                       len(cls_to_idxs[c])))
        chosen = rng.choice(cls_to_idxs[c], size=min(want, len(cls_to_idxs[c])), replace=False)
        counts[c] = np.sort(chosen)

    known_classes = sorted(perm[:args.num_known])
    unknown_classes = sorted(perm[args.num_known:])
    print(f'imb{args.imb_ratio}: known classes e.g. {known_classes[:5]}..., '
          f'unknown e.g. {unknown_classes[:5]}...')

    l_k, unl_k, unl_unk = [], [], []
    for c in known_classes:
        idxs = counts[c]
        n_labeled = int(round(len(idxs) * args.anno_ratio / 100.0))
        picked = rng.choice(len(idxs), size=n_labeled, replace=False)
        mask = np.zeros(len(idxs), dtype=bool)
        mask[picked] = True
        l_k.extend(idxs[mask])
        unl_k.extend(idxs[~mask])
    for c in unknown_classes:
        unl_unk.extend(counts[c])

    out_dir = os.path.join(args.out_root, f'cub200_k{args.num_known}_imbrho{args.imb_ratio}')
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in [('l_k_uq_idxs', l_k), ('unl_k_uq_idxs', unl_k),
                      ('unl_unk_uq_idxs', unl_unk)]:
        arr = np.sort(np.array(arr))
        assert len(np.unique(arr)) == len(arr), 'index collision'
        torch.save(arr, os.path.join(out_dir, name + '.pt'))

    # Report
    lt = [len(counts[c]) for c in known_classes + unknown_classes]
    print(f'labeled={len(l_k)}  unlabeled-known={len(unl_k)}  unlabeled-unknown={len(unl_unk)}')
    print(f'total kept={sum(lt)} / {len(targets)}   head={max(lt)}  tail={min(lt)}')


if __name__ == '__main__':
    main()
