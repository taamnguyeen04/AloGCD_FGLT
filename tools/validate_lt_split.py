"""5-minute check: do KIET's .pt split files match torchvision ordering?

Usage:
    python tools/validate_lt_split.py --dataset aircraft --imb 10
    python tools/validate_lt_split.py --dataset cars --imb 10

PASS = indexed class histogram matches split_meta.json (aircraft) or looks sane
       long-tailed with no empty class (cars, no meta committed).
FAIL = STOP, do not train. Ask split author for the maker script or remake splits
       (template: tools/make_cub_lt_splits.py).
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['aircraft', 'cars'])
    ap.add_argument('--imb', type=int, default=10)
    ap.add_argument('--k', type=int, default=None)
    a = ap.parse_args()

    import numpy as np
    import torch

    if a.dataset == 'aircraft':
        from data.fgvc_aircraft import AircraftDataset
        from config import aircraft_root
        k = a.k or 80
        split_dir = f'data_uq_idxs/aircraft_k{k}_imb{a.imb}'
        ds = AircraftDataset(root=aircraft_root, train=True, transform=None)
        C = 100
    else:
        from data.stanford_cars import CarsDataset
        from config import cars_root
        k = a.k or 98
        split_dir = f'data_uq_idxs/cars196_k{k}_imb{a.imb}'
        ds = CarsDataset(root=cars_root, train=True, transform=None)
        C = 196

    print(f'[validate] N_train={len(ds)} classes={C} split={split_dir}')
    files = {}
    for name in ('l_k_uq_idxs', 'unl_k_uq_idxs', 'unl_unk_uq_idxs'):
        p = os.path.join(split_dir, name + '.pt')
        assert os.path.isfile(p), f'missing {p}'
        files[name] = np.asarray(torch.load(p, weights_only=False))
        assert files[name].max() < len(ds), f'{name} index out of range'
    all_idx = np.concatenate(list(files.values()))
    got = Counter(int(t) for t in np.asarray(ds.targets)[all_idx])
    hist = np.bincount(np.asarray(ds.targets)[all_idx], minlength=C)
    print(f'[validate] indexed: n={len(all_idx)} classes_hit={len(got)}/{C} '
          f'max={hist.max()} min={hist.min()}')

    meta_path = os.path.join(split_dir, 'split_meta.json')
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path))
        want = {int(kk): int(vv) for kk, vv in
                (meta.get('requested_counts') or meta.get('actual_counts')).items()}
        tot_got = Counter()
        for name, idx in files.items():
            tot_got.update(int(t) for t in np.asarray(ds.targets)[idx])
        bad = {c: (tot_got.get(c, 0), want[c]) for c in want
               if tot_got.get(c, 0) != want[c]}
        if bad:
            print(f'[validate] FAIL: {len(bad)} mismatches, e.g. {dict(list(bad.items())[:5])}')
            return 1
        print('[validate] PASS: matches split_meta.json.')
        return 0
    if (hist == 0).any():
        print(f'[validate] FAIL: {(hist == 0).sum()} empty classes (no meta to compare).')
        return 1
    print('[validate] PASS (basic LT check, no meta committed for this split).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
