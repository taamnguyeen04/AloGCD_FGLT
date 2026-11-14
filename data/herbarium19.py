"""Herbarium 19 dataset loader for BaCon GCD.

EXPECTED LAYOUT under <herbarium_root>/ (default D:/data/Herbarium 19):
    small-train/small-train/<class_id>/*.jpg (683 classes, 34225 images)
    or train/train/<class_id>/*.jpg
    or <herbarium_root>/<class_id>/*.jpg

683 classes total (341 known classes 0..340, 342 novel classes 341..682).
Herbarium 19 is already naturally imbalanced (~46:1 max/min ratio, 14 to 646 images/class).
No artificial split files or pruning needed: all 34,225 images are preserved directly.
Standard GCD protocol evaluates on all unlabeled discovery data (unlabeled known + novel).
"""

from copy import deepcopy
import io
import os
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from config import herbarium_root

N_CLASS = 683
N_KNOWN = 341


def resolve_herbarium_image_dir(root):
    """Auto-resolve the directory containing 683 class subdirectories."""
    root = Path(root).expanduser().resolve()
    candidates = [
        root / "small-train" / "small-train",
        root / "small-train",
        root / "train" / "train",
        root / "train",
        root,
        # Fallback candidates if root is .../herbarium19 but volume has .../herbarium
        root.parent / "herbarium" / "small-train" / "small-train",
        root.parent / "herbarium" / "small-train",
        root.parent / "herbarium" / "train" / "train",
        root.parent / "herbarium" / "train",
        root.parent / "herbarium",
        # Fallback if root is .../herbarium but volume has .../herbarium19
        root.parent / "herbarium19" / "small-train" / "small-train",
        root.parent / "herbarium19" / "small-train",
        root.parent / "herbarium19" / "train" / "train",
        root.parent / "herbarium19" / "train",
        root.parent / "herbarium19",
    ]
    checked = []
    for c in candidates:
        if c in checked:
            continue
        checked.append(c)
        if c.is_dir():
            try:
                subdirs = [x for x in c.iterdir() if x.is_dir()]
                if len(subdirs) == N_CLASS:
                    return c
            except OSError:
                continue
    raise FileNotFoundError(
        f"Could not find {N_CLASS} class directories in Herbarium 19 root: {root}. "
        "Checked: " + ", ".join(str(c) for c in checked)
    )


def _open_image_robust(path, retries=3):
    """Read image handling transient FUSE/volume errors on remote / network mounts."""
    last_err = None
    for attempt in range(retries):
        try:
            with open(path, "rb") as f:
                data = f.read()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            image.load()
            return image
        except OSError as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise OSError(f"Failed to read image after {retries} tries: {path} ({last_err})")


class HerbariumDataset(Dataset):
    def __init__(self, root=None, transform=None):
        if root is None:
            root = herbarium_root
        self.root = Path(root).expanduser().resolve()
        self.img_dir = resolve_herbarium_image_dir(self.root)
        self.transform = transform

        folder_ds = ImageFolder(str(self.img_dir))
        self.samples = folder_ds.samples
        self.targets = folder_ds.targets
        self.classes = folder_ds.classes
        self.uq_idxs = np.arange(len(self.samples), dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = _open_image_robust(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target), int(self.uq_idxs[index])


def subsample_dataset(dataset, idxs):
    idxs = np.asarray(idxs)
    dataset.samples = [dataset.samples[i] for i in idxs]
    dataset.targets = [dataset.targets[i] for i in idxs]
    dataset.uq_idxs = np.asarray(dataset.uq_idxs)[idxs]
    return dataset


def split_herbarium_natural(targets, k=341, anno_ratio=50, seed=416):
    """Split Herbarium 19 keeping its full natural long-tail distribution.

    Herbarium 19 is naturally imbalanced (~46:1 max/min ratio, 14 to 646 samples per class).
    No artificial subsampling or pruning is applied:
      - Known classes (0..k-1): 50% labeled, 50% unlabeled.
      - Novel classes (k..N-1): 100% unlabeled.
    Total samples = 34,225 (all images preserved).
    """
    targets = np.asarray(targets)
    n_total = N_CLASS
    rng = np.random.RandomState(seed)

    l_k, unl_k, unl_unk = [], [], []
    for c in range(n_total):
        cls_idxs = np.where(targets == c)[0]
        if len(cls_idxs) == 0:
            continue
        if c < k:
            n_l = max(1, int(round(len(cls_idxs) * (anno_ratio / 100.0))))
            if len(cls_idxs) >= 2 and n_l == len(cls_idxs):
                n_l = len(cls_idxs) - 1
            perm = rng.permutation(cls_idxs)
            l_k.extend(perm[:n_l])
            unl_k.extend(perm[n_l:])
        else:
            unl_unk.extend(cls_idxs)

    l_k = np.sort(np.asarray(l_k, dtype=np.int64))
    unl_k = np.sort(np.asarray(unl_k, dtype=np.int64))
    unl_unk = np.sort(np.asarray(unl_unk, dtype=np.int64))

    return l_k, unl_k, unl_unk


def get_herbarium_19_datasets(train_transform, test_transform, train_classes=range(341), args=None):
    k = getattr(args, "labeled_classes", 341) if args is not None else 341
    if hasattr(args, "anno_ratio"):
        args.anno_ratio = 50
    anno_ratio = getattr(args, "anno_ratio", 50)
    seed = getattr(args, "seed", 416)
    if seed is None or seed < 0:
        seed = 416

    whole_training_set = HerbariumDataset(root=herbarium_root, transform=train_transform)

    # Herbarium 19 is naturally imbalanced: split directly on full dataset without artificial split files
    l_k_pos, unl_k_pos, unl_unk_pos = split_herbarium_natural(
        whole_training_set.targets, k=k, anno_ratio=anno_ratio, seed=seed
    )

    t = np.asarray(whole_training_set.targets)
    known = set(int(c) for c in t[np.concatenate([l_k_pos, unl_k_pos])])
    novel = set(int(c) for c in t[unl_unk_pos])
    assert known.isdisjoint(novel), f"known/novel overlap: {known & novel}"
    assert len(known) == k, f"known {len(known)} != k={k}"
    assert len(novel) == N_CLASS - k, f"novel {len(novel)} != {N_CLASS - k}"

    print(f"[herbarium19] Natural long-tail distribution: {len(whole_training_set)} total images.")
    print(f"  Known: {k} classes ({len(l_k_pos)} labeled, {len(unl_k_pos)} unlabeled)")
    print(f"  Novel: {N_CLASS - k} classes ({len(unl_unk_pos)} unlabeled)")
    print(f"  Total train (merged): {len(l_k_pos) + len(unl_k_pos) + len(unl_unk_pos)}")
    print(f"  Total unlabelled discovery (test): {len(unl_k_pos) + len(unl_unk_pos)}")

    train_dataset_labelled = subsample_dataset(deepcopy(whole_training_set), l_k_pos)
    unlabelled_known_dataset = subsample_dataset(deepcopy(whole_training_set), unl_k_pos)
    unlabelled_unknown_dataset = subsample_dataset(deepcopy(whole_training_set), unl_unk_pos)
    train_dataset_unlabelled = torch.utils.data.ConcatDataset(
        [unlabelled_known_dataset, unlabelled_unknown_dataset]
    )

    # Standard GCD protocol for Herbarium 19 evaluates on all unlabeled discovery data
    whole_test_set = HerbariumDataset(root=herbarium_root, transform=test_transform)
    test_unlabelled_known = subsample_dataset(deepcopy(whole_test_set), unl_k_pos)
    test_unlabelled_unknown = subsample_dataset(deepcopy(whole_test_set), unl_unk_pos)
    test_dataset = torch.utils.data.ConcatDataset([test_unlabelled_known, test_unlabelled_unknown])

    return {
        "train_labelled": train_dataset_labelled,
        "train_unlabelled": train_dataset_unlabelled,
        "val": None,
        "test": test_dataset,
        "bl_train_unlabelled": None,
    }
