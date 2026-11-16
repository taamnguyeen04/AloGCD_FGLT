import os
import tempfile
import shutil
from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np
import torch
from data.data_utils import subsample_instances
from config import cifar_100_root, cifar_10_root

def _normalize_cifar_root(root, target_subfolder):
    if not root:
        return root
    clean = str(root).rstrip('/\\')
    if clean.endswith(target_subfolder):
        return os.path.dirname(clean)
    if (os.path.exists(os.path.join(clean, 'train')) or os.path.exists(os.path.join(clean, 'data_batch_1'))) and not os.path.exists(os.path.join(clean, target_subfolder)):
        proxy_dir = os.path.join(tempfile.gettempdir(), f'cifar_proxy_{target_subfolder}')
        sub_dir = os.path.join(proxy_dir, target_subfolder)
        if not os.path.exists(sub_dir):
            os.makedirs(proxy_dir, exist_ok=True)
            try:
                os.symlink(clean, sub_dir, target_is_directory=True)
            except Exception:
                os.makedirs(sub_dir, exist_ok=True)
                for fname in os.listdir(clean):
                    src_f = os.path.join(clean, fname)
                    dst_f = os.path.join(sub_dir, fname)
                    if os.path.isfile(src_f) and not os.path.exists(dst_f):
                        try:
                            os.link(src_f, dst_f)
                        except Exception:
                            shutil.copyfile(src_f, dst_f)
        return proxy_dir
    return clean

class CustomCIFAR10(CIFAR10):

    def __init__(self, root=None, sublist=[], *args, **kwargs):
        root = _normalize_cifar_root(root or cifar_10_root, 'cifar-10-batches-py')
        super(CustomCIFAR10, self).__init__(root=root, *args, **kwargs)

        self.txt = sublist
        self.uq_idxs = np.array(range(len(self)))
        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()
            self.uq_idxs = self.uq_idxs[sublist]

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


class CustomCIFAR100(CIFAR100):

    def __init__(self, root=None, sublist=[], return_false_mask=False, return_true_mask=False, *args, **kwargs):
        root = _normalize_cifar_root(root or cifar_100_root, 'cifar-100-python')
        super(CustomCIFAR100, self).__init__(root=root, *args, **kwargs)
        self.txt = sublist

        self.uq_idxs = np.array(range(len(self)))
        self.return_false_mask = return_false_mask
        self.return_true_mask = return_true_mask

        if len(sublist) > 0:
            self.data = self.data[sublist]
            self.targets = np.array(self.targets)[sublist].tolist()
            self.uq_idxs = self.uq_idxs[sublist]

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:

        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:

        return None


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

def get_lt_dist(cls_num, img_max, imb_factor):
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return torch.tensor(img_num_per_cls)


def get_cifar_100_datasets(train_transform, test_transform, args=None):
    root = getattr(args, 'data_root', None) or cifar_100_root
    splits_root = getattr(args, 'splits_root', None) or 'data_uq_idxs'

    whole_training_set = CustomCIFAR100(root=root, train=True, transform=train_transform, download=True)
    whole_training_set.target_transform = None

    split_dir = os.path.join(splits_root, f'cifar100_k80_imb{args.imb_ratio}')
    l_k_uq_idxs = torch.load(os.path.join(split_dir, 'l_k_uq_idxs.pt'), weights_only=False)
    unl_unk_uq_idxs = torch.load(os.path.join(split_dir, 'unl_unk_uq_idxs.pt'), weights_only=False)
    unl_k_uq_idxs = torch.load(os.path.join(split_dir, 'unl_k_uq_idxs.pt'), weights_only=False)

    train_dataset_labelled = subsample_dataset(deepcopy(whole_training_set), l_k_uq_idxs)
    lt_unlabeled_known_dataset = subsample_dataset(deepcopy(whole_training_set), unl_k_uq_idxs)
    lt_unlabeled_unknown_dataset = subsample_dataset(deepcopy(whole_training_set), unl_unk_uq_idxs)
    train_dataset_unlabelled = torch.utils.data.ConcatDataset([lt_unlabeled_known_dataset, lt_unlabeled_unknown_dataset])

    test_dataset = CustomCIFAR100(root=root, transform=test_transform, train=False)

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'test': test_dataset,
    }

    return all_datasets

def get_cifar_10_datasets(train_transform, test_transform, args=None):
    root = getattr(args, 'data_root', None) or cifar_10_root
    splits_root = getattr(args, 'splits_root', None) or 'data_uq_idxs'

    whole_training_set = CustomCIFAR10(root=root, train=True, transform=train_transform, download=True)
    whole_training_set.target_transform = None

    split_dir = os.path.join(splits_root, f'cifar10_k5_imb{args.imb_ratio}')
    l_k_uq_idxs = torch.load(os.path.join(split_dir, 'l_k_uq_idxs.pt'), weights_only=False)
    unl_unk_uq_idxs = torch.load(os.path.join(split_dir, 'unl_unk_uq_idxs.pt'), weights_only=False)
    unl_k_uq_idxs = torch.load(os.path.join(split_dir, 'unl_k_uq_idxs.pt'), weights_only=False)

    train_dataset_labelled = subsample_dataset(deepcopy(whole_training_set), l_k_uq_idxs)
    lt_unlabeled_known_dataset = subsample_dataset(deepcopy(whole_training_set), unl_k_uq_idxs)
    lt_unlabeled_unknown_dataset = subsample_dataset(deepcopy(whole_training_set), unl_unk_uq_idxs)
    train_dataset_unlabelled = torch.utils.data.ConcatDataset([lt_unlabeled_known_dataset, lt_unlabeled_unknown_dataset])
    
    test_dataset = CustomCIFAR10(root=root, transform=test_transform, train=False)

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'test': test_dataset,
    }

    return all_datasets
