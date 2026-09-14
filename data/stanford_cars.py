"""Stanford Cars wrapper for BaCon DA-GCD (no torchvision download needed).

Torchvision's StanfordCars is unusable (upstream URL dead), so this parses the
official devkit .mat files directly (scipy, already a repo dependency).

EXPECTED LAYOUT under <cars_root>/ (CARS_ROOT, default D:/data/stanford_cars):
    cars_train/*.jpg            (8144 images, 00001.jpg ...)
    cars_test/*.jpg             (8041 images)
    devkit/cars_train_annos.mat
    devkit/cars_test_annos_withlabels.mat
  (Official Krause devkit; the Kaggle "stanford-car-dataset" dump has the same
  files. The jutrera by-classes-folder dump instead ships anno_train/test.csv +
  names.csv + class folders -> see _load_kaggle_csv fallback below.)

CANONICAL ORDER = sorted filename (== torchvision StanfordCars row order).
uq_idxs = positional np.arange(N). LT splits in
data_uq_idxs/cars196_k{k}_imb{imb}/ are POSITIONAL into this ordering
(same assumption as CUB; run tools/validate_lt_split.py before training).
"""

import glob
import os
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import cars_root

N_TRAIN, N_TEST, N_CLASS = 8144, 8041, 196


def _load_devkit_mat(mat_path):
    from scipy.io import loadmat
    mat = loadmat(mat_path, squeeze_me=True)
    annos = mat['annotations']
    items = []
    for a in np.atleast_1d(annos):
        fname = str(a['fname'])
        cls = int(a['class']) - 1  # 1-based -> 0-based
        items.append((fname, cls))
    items.sort(key=lambda x: x[0])
    return items


def _load_kaggle_csv(root, train):
    """jutrera-style dump: anno_*.csv WITHOUT header (filename,x1,y1,x2,y2,class
    1-based) + images under class folders (any depth, e.g. car_data/car_data/
    train/<class name>/*.jpg). Order = sorted filename (== devkit order)."""
    import pandas as pd
    csv_path = os.path.join(root, 'anno_train.csv' if train else 'anno_test.csv')
    df = pd.read_csv(csv_path, header=None)
    # One recursive walk for filename -> full path (16k files, seconds).
    path_by_name = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                path_by_name.setdefault(fn, os.path.join(dirpath, fn))
    items = []
    for fname, cls in zip(df[0].astype(str), df[5].astype(int)):
        fname = os.path.basename(fname)
        if fname not in path_by_name:
            raise FileNotFoundError(f'{fname} ({csv_path}) not found under {root}.')
        items.append((path_by_name[fname], int(cls) - 1))
    assert all(0 <= c < N_CLASS for _, c in items), 'class ids out of 0..195'
    items.sort(key=lambda x: os.path.basename(x[0]))
    return [(p, c) for p, c in items]


class CarsDataset(Dataset):
    def __init__(self, root=cars_root, train=True, transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        img_dir = os.path.join(root, 'cars_train' if train else 'cars_test')
        mat = os.path.join(root, 'devkit',
                           'cars_train_annos.mat' if train else 'cars_test_annos_withlabels.mat')
        use_csv = os.path.isfile(os.path.join(
            root, 'anno_train.csv' if train else 'anno_test.csv'))
        if os.path.isfile(mat) and os.path.isdir(img_dir):
            items = _load_devkit_mat(mat)
            self.samples = [(os.path.join(img_dir, f), c) for f, c in items]
        elif use_csv:
            self.samples = _load_kaggle_csv(root, train)  # full paths already
        else:
            try:
                seen = sorted(os.listdir(root))
            except OSError:
                seen = ['<unreadable>']
            raise FileNotFoundError(
                f'No Cars data under {root}. Saw: {seen}. Expected official layout '
                '(cars_train/, cars_test/, devkit/*.mat) or Kaggle dump (anno_*.csv).')
        if use_csv:
            print(f'[cars] Kaggle-CSV layout: {len(self.samples)} images.')
        else:
            missing = [p for p, _ in self.samples if not os.path.isfile(p)]
            if missing:
                raise FileNotFoundError(
                    f'{len(missing)} images missing under {img_dir} (e.g. {missing[0]}). '
                    'Check CARS_ROOT layout.')
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


def _validate_cars_splits(l_k, unl_k, unl_unk, targets, split_dir, k):
    """l_k/unl_k cover the K known classes, unl_unk the novel rest (complement).
    Per-file full-196 coverage must NOT be required (same bug class as the
    aircraft per-file check)."""
    t = np.asarray(targets)
    l_k, unl_k, unl_unk = (np.asarray(x) for x in (l_k, unl_k, unl_unk))
    for name, idx in (('l_k', l_k), ('unl_k', unl_k), ('unl_unk', unl_unk)):
        assert idx.max() < len(t), f'{name} index out of range'
    assert len(set(l_k) & set(unl_k) & set(unl_unk)) == 0, 'split files overlap!'
    known = set(int(c) for c in t[np.concatenate([l_k, unl_k])])
    novel = set(int(c) for c in t[unl_unk])
    assert known.isdisjoint(novel), f'known/novel overlap: {known & novel}'
    assert len(known) == k, f'known {len(known)} != k={k}'
    assert len(novel) == N_CLASS - k, f'novel {len(novel)} != {N_CLASS - k}'
    hist = np.bincount(t[np.concatenate([l_k, unl_k, unl_unk])], minlength=N_CLASS)
    print(f'[cars] splits OK: known {len(known)} + novel {len(novel)} classes, '
          f'n={len(l_k) + len(unl_k) + len(unl_unk)} max={hist.max()} min={hist.min()} '
          f'(from {split_dir})')


def get_stanford_cars_datasets(train_transform, test_transform, train_classes=range(98), args=None):
    if args is not None:
        train_classes = range(args.labeled_classes)
    args.anno_ratio = 50
    seed = 416
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    whole_training_set = CarsDataset(root=cars_root, train=True, transform=train_transform)
    assert len(whole_training_set) == N_TRAIN, \
        f'cars train N={len(whole_training_set)} != {N_TRAIN}: wrong data source?'

    k = getattr(args, 'labeled_classes', 98)
    imb = getattr(args, 'imb_ratio', 10)
    split_dir = f'data_uq_idxs/cars196_k{k}_imb{imb}'

    l_k_pos = torch.load(f'{split_dir}/l_k_uq_idxs.pt', weights_only=False)
    unl_k_pos = torch.load(f'{split_dir}/unl_k_uq_idxs.pt', weights_only=False)
    unl_unk_pos = torch.load(f'{split_dir}/unl_unk_uq_idxs.pt', weights_only=False)

    _validate_cars_splits(l_k_pos, unl_k_pos, unl_unk_pos,
                          whole_training_set.targets, split_dir, k)

    train_dataset_labelled = subsample_dataset(deepcopy(whole_training_set), l_k_pos)
    unlabelled_known_dataset = subsample_dataset(deepcopy(whole_training_set), unl_k_pos)
    unlabelled_unknown_dataset = subsample_dataset(deepcopy(whole_training_set), unl_unk_pos)
    train_dataset_unlabelled = torch.utils.data.ConcatDataset(
        [unlabelled_known_dataset, unlabelled_unknown_dataset])

    test_dataset = CarsDataset(root=cars_root, train=False, transform=test_transform)

    return {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': None,
        'test': test_dataset,
        'bl_train_unlabelled': None,
    }
