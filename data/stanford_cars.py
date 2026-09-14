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
    """Fallback for jutrera-style dumps (anno_train.csv / anno_test.csv)."""
    import pandas as pd
    csv_path = os.path.join(root, 'anno_train.csv' if train else 'anno_test.csv')
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    fcol = next((cols[c] for c in cols if 'file' in c or 'name' in c or 'image' in c), None)
    ccol = next((cols[c] for c in cols if 'class' in c or 'label' in c or 'target' in c), None)
    if fcol is None or ccol is None:
        raise ValueError(f'{csv_path}: cannot find filename/class columns {list(df.columns)}')
    names_path = os.path.join(root, 'names.csv')
    name_to_id = None
    if os.path.isfile(names_path):
        names = pd.read_csv(names_path, header=None)[0].tolist()
        name_to_id = {n: i for i, n in enumerate(names)}
    raw = []
    for _, row in df.iterrows():
        f, c = str(row[fcol]), row[ccol]
        cid = name_to_id[c] if isinstance(c, str) and name_to_id else int(c)
        raw.append((os.path.basename(f), int(cid)))
    # Auto-detect 0-based (max 195) vs 1-based (max 196) class ids.
    one_based = max(c for _, c in raw) == N_CLASS
    items = [(f, c - 1 if one_based else c) for f, c in raw]
    assert all(0 <= c < N_CLASS for _, c in items), 'class ids out of 0..195'
    items.sort(key=lambda x: x[0])
    return items


class CarsDataset(Dataset):
    def __init__(self, root=cars_root, train=True, transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        img_dir = os.path.join(root, 'cars_train' if train else 'cars_test')
        mat = os.path.join(root, 'devkit',
                           'cars_train_annos.mat' if train else 'cars_test_annos_withlabels.mat')
        if os.path.isfile(mat) and os.path.isdir(img_dir):
            items = _load_devkit_mat(mat)
        elif os.path.isfile(os.path.join(
                root, 'anno_train.csv' if train else 'anno_test.csv')):
            items = _load_kaggle_csv(root, train)
        else:
            try:
                seen = sorted(os.listdir(root))
            except OSError:
                seen = ['<unreadable>']
            raise FileNotFoundError(
                f'No Cars data under {root}. Saw: {seen}. Expected official layout '
                '(cars_train/, cars_test/, devkit/*.mat) or Kaggle dump (anno_*.csv).')
            # Resolve full paths against class folders or flat dirs.
            cand_dirs = [img_dir, root,
                         *[d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d)]]
            resolved = []
            for fname, cls in items:
                hit = next((os.path.join(d, fname) for d in cand_dirs
                            if os.path.isfile(os.path.join(d, fname))), None)
                if hit is None:
                    raise FileNotFoundError(
                        f'{fname} not found under {root}. Expected official layout '
                        '(cars_train/, cars_test/, devkit/*.mat).')
                resolved.append((hit, cls))
            self.samples = resolved
            self.targets = [t for _, t in resolved]
            self.uq_idxs = np.arange(len(resolved))
            return
        self.samples = [(os.path.join(img_dir, f), c) for f, c in items]
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


def _validate_split(idxs, targets, split_dir, split_name):
    idxs = np.asarray(idxs)
    assert idxs.max() < len(targets), f'{split_name} index out of range'
    hist = np.bincount(np.asarray(targets)[idxs], minlength=N_CLASS)
    assert (hist > 0).all(), f'{split_name}: {(hist == 0).sum()} empty classes'
    print(f'[cars] {split_name}: n={len(idxs)} max={hist.max()} min={hist.min()} '
          f'(from {split_dir})')
    return hist


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

    _validate_split(l_k_pos, whole_training_set.targets, split_dir, 'l_k')
    _validate_split(unl_k_pos, whole_training_set.targets, split_dir, 'unl_k')
    _validate_split(unl_unk_pos, whole_training_set.targets, split_dir, 'unl_unk')

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
