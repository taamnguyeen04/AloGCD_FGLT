"""Generate long-tailed index splits for CIFAR-10/100 in BaCon's
``data_uq_idxs/<name>/{l_k,unl_k,unl_unk}_uq_idxs.pt`` format.

Recipe (reverse-engineered from the 8 shipped splits, all matched EXACTLY):
  - Per-class profile = BaCon's ``get_lt_dist`` (``data/cifar.py``):
    ``count[rank] = int(img_max * (1/imb) ** (rank / (C-1)))``.
  - ``img_max`` = 90% of the balanced size: 450 for CIFAR-100, 4500 for
    CIFAR-10 (verified: profile sums equal shipped totals 8/8).
  - Classes are randomly permuted (seed, default 416 = repo convention);
    profile rank follows permuted position; first K permuted = known.
    (Rank = class order is RULED OUT: shipped novel pools are far larger
    than any contiguous tail; totals fit random 80/20 subsets.)
  - Known classes split ~50/50 labeled/unlabeled per class
    (``round(n*0.5)``, ``rng.choice``); novel classes go fully unlabeled.
  - Index space = positions into torchvision train ordering (0..N-1),
    exactly ``CustomCIFAR*.uq_idxs``. Files: int64, disjoint sets.

Caveats:
  - The upstream seed/assignment is unrecoverable from the .pt files alone,
    so this reproduces the recipe family, not bit-identical upstream files.
    That is fine for imb10, which has no upstream reference at all.
  - Like the repo's CUB splits, known identity is permuted (not fixed
    0..K-1). Pass --no-permute for classic rank-ordered LT instead.

Usage (on server, auto-downloads CIFAR ~170MB on first run)::

    python tools/make_cifar_lt_splits.py --dataset cifar100 --imb-ratio 10
    python tools/make_cifar_lt_splits.py --dataset cifar10 --imb-ratio 10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def get_lt_dist(cls_num, img_max, imb_factor):
    """BaCon's exponential profile (data/cifar.py:106-111)."""
    return [int(img_max * (imb_factor ** (i / (cls_num - 1.0))))
            for i in range(cls_num)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['cifar10', 'cifar100'],
                   default='cifar100')
    p.add_argument('--num-known', type=int, default=None,
                   help='default: 5 for cifar10, 80 for cifar100')
    p.add_argument('--imb-ratio', type=int, required=True)
    p.add_argument('--img-max', type=int, default=None,
                   help='default: 4500 for cifar10, 450 for cifar100 (=90%% of balanced)')
    p.add_argument('--anno-ratio', type=float, default=50.0,
                   help='% of known-class imgs that are labeled')
    p.add_argument('--seed', type=int, default=416)
    p.add_argument('--no-permute', action='store_true',
                   help='rank = class order (classic LT); default permutes classes')
    p.add_argument('--cifar-root', default=None,
                   help='where to find/download CIFAR (default: config cifar_10/100_root)')
    p.add_argument('--out-root', default='data_uq_idxs')
    args = p.parse_args()

    if args.num_known is None:
        args.num_known = 5 if args.dataset == 'cifar10' else 80
    if args.img_max is None:
        args.img_max = 4500 if args.dataset == 'cifar10' else 450

    if args.cifar_root is None:
        from config import cifar_10_root, cifar_100_root
        args.cifar_root = cifar_10_root if args.dataset == 'cifar10' else cifar_100_root

    from torchvision.datasets import CIFAR10, CIFAR100
    cls = CIFAR10 if args.dataset == 'cifar10' else CIFAR100
    # transform=None -> PIL images; we only need .targets (labels in file order)
    ds = cls(root=args.cifar_root, train=True, download=True)
    targets = np.array(ds.targets)
    n_total_cls = int(targets.max()) + 1
    n = len(ds)
    print(f'{args.dataset}: N={n}, classes={n_total_cls}')

    cls_to_pos = {c: np.where(targets == c)[0] for c in range(n_total_cls)}

    rng = np.random.RandomState(args.seed)
    if args.no_permute:
        order = np.arange(n_total_cls)
    else:
        order = rng.permutation(n_total_cls)

    # Profile rank follows permuted position
    prof = get_lt_dist(n_total_cls, args.img_max, 1.0 / args.imb_ratio)
    counts = {}
    for rank, c in enumerate(order):
        want = min(prof[rank], len(cls_to_pos[c]))
        chosen = rng.choice(cls_to_pos[c], size=want, replace=False)
        counts[c] = np.sort(chosen)

    known_classes = sorted(order[:args.num_known].tolist())
    unknown_classes = sorted(order[args.num_known:].tolist())
    print(f'imb{args.imb_ratio}: known e.g. {known_classes[:5]}..., '
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

    l_k = np.array(l_k)
    unl_k = np.array(unl_k)
    unl_unk = np.array(unl_unk)
    assert len(np.unique(np.concatenate([l_k, unl_k, unl_unk]))) == \
        len(l_k) + len(unl_k) + len(unl_unk), 'index collision'
    assert max(l_k.max(), unl_k.max(), unl_unk.max()) < n, 'index out of range'

    k = args.num_known
    tag = 'cifar10_k5' if args.dataset == 'cifar10' else f'cifar100_k{k}'
    out_dir = os.path.join(args.out_root, f'{tag}_imb{args.imb_ratio}')
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in [('l_k_uq_idxs', l_k), ('unl_k_uq_idxs', unl_k),
                      ('unl_unk_uq_idxs', unl_unk)]:
        torch.save(torch.from_numpy(arr.astype(np.int64)),
                   os.path.join(out_dir, name + '.pt'))

    lt = [len(counts[c]) for c in known_classes + unknown_classes]
    print(f'saved -> {out_dir}')
    print(f'labeled={len(l_k)}  unlabeled-known={len(unl_k)}  '
          f'unlabeled-unknown={len(unl_unk)}')
    print(f'total kept={sum(lt)} / {n}   head={max(lt)}  tail={min(lt)}')


if __name__ == '__main__':
    main()
