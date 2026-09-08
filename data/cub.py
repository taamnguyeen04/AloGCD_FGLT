import os
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import cub_root


class CUBDataset(Dataset):
    def __init__(self, root=cub_root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.img_path = []
        self.targets = []
        self.uq_idxs = []

        images = self._read_mapping('images.txt', value_type=str)
        labels = self._read_mapping('image_class_labels.txt', value_type=int)
        splits = self._read_mapping('train_test_split.txt', value_type=int)

        for image_id in sorted(images.keys()):
            is_train = splits[image_id] == 1
            if is_train != train:
                continue
            self.img_path.append(os.path.join(root, 'images', images[image_id]))
            self.targets.append(labels[image_id] - 1)
            self.uq_idxs.append(image_id - 1)

        self.uq_idxs = np.array(self.uq_idxs)

    def _read_mapping(self, filename, value_type):
        mapping = {}
        with open(os.path.join(self.root, filename), 'r') as file:
            for line in file:
                key, value = line.strip().split(maxsplit=1)
                mapping[int(key)] = value_type(value)
        return mapping

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        with open(self.img_path[index], 'rb') as file:
            image = Image.open(file).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index], self.uq_idxs[index]


def subsample_dataset(dataset, idxs):
    dataset.img_path = [dataset.img_path[i] for i in idxs]
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]
    return dataset


def subsample_classes(dataset, include_classes):
    cls_idxs = [idx for idx, target in enumerate(dataset.targets) if target in include_classes]
    return subsample_dataset(dataset, cls_idxs)


def split_known_labelled_unlabelled(dataset, args):
    labelled_idxs = []
    unlabelled_idxs = []
    anno_ratio = getattr(args, 'anno_ratio', 50)

    for cls in np.unique(dataset.targets):
        cls_idxs = np.where(np.array(dataset.targets) == cls)[0]
        unlabelled_cls_idxs = np.random.choice(
            cls_idxs,
            replace=False,
            size=(int((1 - anno_ratio / 100) * len(cls_idxs)),),
        )
        labelled_cls_idxs = [idx for idx in cls_idxs if idx not in unlabelled_cls_idxs]
        labelled_idxs.extend(labelled_cls_idxs)
        unlabelled_idxs.extend(unlabelled_cls_idxs)

    return labelled_idxs, unlabelled_idxs


def get_cub_200_datasets(train_transform, test_transform, train_classes=range(100), args=None):
    total_class = 200
    if args is not None:
        train_classes = range(args.labeled_classes)
    args.anno_ratio = 50
    seed = 416
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    use_lt_split = getattr(args, 'use_lt_split', False)
    if hasattr(args, 'imb_ratio') and args.imb_ratio is not None:
        use_lt_split = True

    whole_training_set = CUBDataset(root=cub_root, train=True, transform=train_transform)

    if use_lt_split:
        k = getattr(args, 'labeled_classes', 100)
        imb = getattr(args, 'imb_ratio', 10)
        split_dir = f'data_uq_idxs/cub200_k{k}_imb{imb}'

        # NOTE: make_cub_lt_splits.py saves POSITIONAL indices (0..N_train-1) into the
        # sorted train-set array — exactly what subsample_dataset expects. No mapping needed.
        l_k_pos = torch.load(f'{split_dir}/l_k_uq_idxs.pt', weights_only=False)
        unl_k_pos = torch.load(f'{split_dir}/unl_k_uq_idxs.pt', weights_only=False)
        unl_unk_pos = torch.load(f'{split_dir}/unl_unk_uq_idxs.pt', weights_only=False)

        # Cross-check: no index should exceed dataset size
        n_train = len(whole_training_set)
        assert l_k_pos.max() < n_train, f"l_k index out of range: {l_k_pos.max()} >= {n_train}"

        train_dataset_labelled = subsample_dataset(deepcopy(whole_training_set), l_k_pos)
        unlabelled_known_dataset = subsample_dataset(deepcopy(whole_training_set), unl_k_pos)
        unlabelled_unknown_dataset = subsample_dataset(deepcopy(whole_training_set), unl_unk_pos)
        train_dataset_unlabelled = torch.utils.data.ConcatDataset([unlabelled_known_dataset, unlabelled_unknown_dataset])
    else:
        whole_known_classes_set = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)

        labelled_known_idxs, unlabelled_known_idxs = split_known_labelled_unlabelled(whole_known_classes_set, args)
        train_dataset_labelled = subsample_dataset(deepcopy(whole_known_classes_set), labelled_known_idxs)
        unlabelled_known_dataset = subsample_dataset(deepcopy(whole_known_classes_set), unlabelled_known_idxs)
        unlabelled_unknown_dataset = subsample_classes(
            deepcopy(whole_training_set),
            include_classes=range(len(train_classes), total_class),
        )
        train_dataset_unlabelled = torch.utils.data.ConcatDataset([unlabelled_known_dataset, unlabelled_unknown_dataset])

    test_dataset = CUBDataset(root=cub_root, train=False, transform=test_transform)

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': None,
        'test': test_dataset,
        'bl_train_unlabelled': None,
    }
    return all_datasets
