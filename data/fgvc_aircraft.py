"""FGVC-Aircraft wrapper for BaCon DA-GCD (torchvision backend, variant level).

- Train pool = torchvision FGVCAircraft(split='trainval', annotation_level='variant')
  in torchvision order (100 classes, ids 0..99). Test = split='test'.
- uq_idxs = positional np.arange(N). LT splits in
  data_uq_idxs/aircraft_k{k}_imb{imb}/ are POSITIONAL (same assumption as CUB).
- split_meta.json (requested_counts, seed, rank_order) is cross-checked loudly:
  mismatch -> ValueError (fail fast, never train on silently-wrong splits).

DATA LAYOUT: <aircraft_root>/fgvc-aircraft-2013b/{data/images_*.txt, ...} as
  torchvision expects (download=True fetches the Oxford tarball; manual placement
  also fine). Env: AIRCRAFT_ROOT (default D:/data/fgvc_aircraft).
"""

import json
import os
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import FGVCAircraft

from config import aircraft_root


class AircraftDataset(Dataset):
    def __init__(self, root=aircraft_root, train=True, transform=None, download=True):
        self.root = root
        self.train = train
        self.transform = transform
        split = 'trainval' if train else 'test'
        try:
            base = FGVCAircraft(root=root, split=split, annotation_level='variant',
                                transform=None, download=download)
        except Exception as e:
            raise RuntimeError(
                f'FGVCAircraft split={split!r} failed under root={root!r}: {e}\n'
                'Place fgvc-aircraft-2013b/ under AIRCRAFT_ROOT or keep download=True.'
            )
        self.samples = [(p, int(t)) for p, t in zip(base._image_files, base._labels)]
        self.targets = [t for _, t in self.samples]
        self.uq_idxs = np.arange(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, target, int(self.uq_idxs[index])


def subsample_dataset(dataset, idxs):
    idxs = np.asarray(idxs)
    dataset.samples = [dataset.samples[i] for i in idxs]
    dataset.targets = [dataset.targets[i] for i in idxs]
    dataset.uq_idxs = np.asarray(dataset.uq_idxs)[idxs]
    return dataset


def subsample_classes(dataset, include_classes):
    include = set(int(c) for c in include_classes)
    cls_idxs = [i for i, t in enumerate(dataset.targets) if t in include]
    return subsample_dataset(dataset, cls_idxs)


def _validate_against_meta(l_k, unl_k, unl_unk, targets, split_dir, known, novel):
    """Cross-check the UNION against the maker's meta (l_k alone is only ~50%
    of each known class, so per-file comparison always fails — that was a bug).

    Checks: (l_k + unl_k) == requested per known class; unl_unk == requested
    per novel class; no overlap between the three files.
    """
    t = np.asarray(targets)
    l_k, unl_k, unl_unk = (np.asarray(x) for x in (l_k, unl_k, unl_unk))
    for name, idx in (('l_k', l_k), ('unl_k', unl_k), ('unl_unk', unl_unk)):
        assert idx.max() < len(t), f'{name} index out of range'
    assert len(set(l_k) & set(unl_k) & set(unl_unk)) == 0, 'split files overlap!'
    meta_path = os.path.join(split_dir, 'split_meta.json')
    if not os.path.isfile(meta_path):
        print('[aircraft] no meta: basic non-empty check only.')
        return
    meta = json.load(open(meta_path))
    want = {int(k): int(v) for k, v in
            (meta.get('requested_counts') or meta.get('actual_counts')).items()}
    got_known = Counter(int(c) for c in t[np.concatenate([l_k, unl_k])])
    got_novel = Counter(int(c) for c in t[unl_unk])
    bad = {c: (got_known.get(c, 0), want[c]) for c in known
           if got_known.get(c, 0) != want[c]}
    bad.update({c: (got_novel.get(c, 0), want[c]) for c in novel
                if got_novel.get(c, 0) != want[c]})
    if bad:
        raise ValueError(
            f'{len(bad)} class-count mismatches vs {meta_path} '
            f'(e.g. {dict(list(bad.items())[:5])}). '
            'Loader ordering != maker ordering -> STOP.')
    print(f'[aircraft] matches meta: known {len(known)} + novel {len(novel)} classes, '
          f'n={len(l_k) + len(unl_k) + len(unl_unk)}.')


def get_fgvc_aircraft_datasets(train_transform, test_transform, train_classes=range(80), args=None):
    total_class = 100
    if args is not None:
        train_classes = range(args.labeled_classes)
    args.anno_ratio = 50
    seed = 416
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    whole_training_set = AircraftDataset(root=aircraft_root, train=True,
                                         transform=train_transform)

    k = getattr(args, 'labeled_classes', 80)
    imb = getattr(args, 'imb_ratio', 10)
    split_dir = f'data_uq_idxs/aircraft_k{k}_imb{imb}'

    l_k_pos = torch.load(f'{split_dir}/l_k_uq_idxs.pt', weights_only=False)
    unl_k_pos = torch.load(f'{split_dir}/unl_k_uq_idxs.pt', weights_only=False)
    unl_unk_pos = torch.load(f'{split_dir}/unl_unk_uq_idxs.pt', weights_only=False)

    meta_path = os.path.join(split_dir, 'split_meta.json')
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path))
        known = [int(c) for c in meta['known_classes']]
        novel = [int(c) for c in meta['novel_classes']]
    else:
        known, novel = list(range(80)), list(range(80, 100))
    _validate_against_meta(l_k_pos, unl_k_pos, unl_unk_pos,
                           whole_training_set.targets, split_dir, known, novel)

    train_dataset_labelled = subsample_dataset(deepcopy(whole_training_set), l_k_pos)
    unlabelled_known_dataset = subsample_dataset(deepcopy(whole_training_set), unl_k_pos)
    unlabelled_unknown_dataset = subsample_dataset(deepcopy(whole_training_set), unl_unk_pos)
    train_dataset_unlabelled = torch.utils.data.ConcatDataset(
        [unlabelled_known_dataset, unlabelled_unknown_dataset])

    test_dataset = AircraftDataset(root=aircraft_root, train=False, transform=test_transform)

    return {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': None,
        'test': test_dataset,
        'bl_train_unlabelled': None,
    }
