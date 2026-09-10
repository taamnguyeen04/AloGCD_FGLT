"""Reproduce the BaCon-O/BaCon-S rows of Table 4.

The test set is balanced, while the Many/Med/Few groups are defined from the
long-tailed training counts. Clusters are aligned to classes once on the full
test set, then accuracies are reported separately for known and novel groups.
"""

import argparse
import csv
import datetime
import logging
import os
import sys
import traceback

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from config import cifar_10_root, cifar_100_root, cub_root

DATASETS = {
    "cifar10":  (cifar_10_root,  5,   10,  "cifar10_k5"),
    "cifar100": (cifar_100_root, 80,  100, "cifar100_k80"),
    "cub200":   (cub_root,       100, 200, None),
}


class Tee:
    def __init__(self, stream, log_stream):
        self.stream = stream
        self.log_stream = log_stream

    def write(self, message):
        self.stream.write(message)
        self.log_stream.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.log_stream.flush()

    def isatty(self):
        return self.stream.isatty()


def setup_logging(log_file):
    directory = os.path.dirname(log_file)
    if directory:
        os.makedirs(directory, exist_ok=True)
    log_stream = open(log_file, "a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_stream)
    sys.stderr = Tee(sys.__stderr__, log_stream)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_stream


def dataset_class(dataset_name):
    from data.cifar import CustomCIFAR10, CustomCIFAR100
    from data.cub import CUBDataset
    mapping = {
        "cifar10":  CustomCIFAR10,
        "cifar100": CustomCIFAR100,
        "cub200":   CUBDataset,
    }
    return mapping[dataset_name]


def load_features(checkpoint_path, dataset_name, args, device):
    from data.augmentations import get_transform
    root, _, _, _ = DATASETS[dataset_name]
    _, test_transform = get_transform(
        "imagenet", image_size=224,
        args=argparse.Namespace(interpolation=3, crop_pct=0.875),
    )
    # CUB200 does not support the `download` kwarg
    extra_kwargs = {} if dataset_name == "cub200" else {"download": True}
    dataset = dataset_class(dataset_name)(root=root, train=False, transform=test_transform, **extra_kwargs)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    backbone_name = getattr(args, 'backbone', 'dinov2_vitb14')
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model.backbone import load_backbone as load_hub_backbone
    logging.info("Loading %s backbone", backbone_name)
    backbone = load_hub_backbone(backbone_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "cl_backbone" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'cl_backbone': {checkpoint_path}")
    backbone.load_state_dict(checkpoint["cl_backbone"])
    backbone.to(device).eval()

    features = []
    targets = []
    with torch.no_grad():
        for batch_index, (images, labels, _) in enumerate(loader, start=1):
            output = torch.nn.functional.normalize(backbone(images.to(device)), dim=-1)
            features.append(output.cpu().numpy())
            targets.append(labels.numpy())
            logging.info("Feature batch %d complete (%d images)", batch_index, len(labels))
    return np.concatenate(features), np.concatenate(targets).astype(int)


def class_groups(dataset_name, imbalance_ratio):
    root, num_old, num_classes, index_prefix = DATASETS[dataset_name]

    if index_prefix is not None:
        # CIFAR: read split index files to get training counts
        train_dataset = dataset_class(dataset_name)(root=root, train=True, transform=None, download=True)
        index_dir = os.path.join("data_uq_idxs", f"{index_prefix}_imb{imbalance_ratio}")
        split_files = ("l_k_uq_idxs.pt", "unl_k_uq_idxs.pt", "unl_unk_uq_idxs.pt")
        indices = np.concatenate([
            np.asarray(torch.load(os.path.join(index_dir, name), map_location="cpu"))
            for name in split_files
        ])
        counts = np.bincount(np.asarray(train_dataset.targets)[indices], minlength=num_classes)
    else:
        # CUB200: compute directly from the full training set (no index files)
        train_dataset = dataset_class(dataset_name)(root=root, train=True, transform=None)
        counts = np.bincount(np.array(train_dataset.targets), minlength=num_classes)

    groups = {}
    for prefix, classes in (("known", np.arange(num_old)),
                            ("novel", np.arange(num_old, num_classes))):
        ordered = classes[np.argsort(-counts[classes])]
        third = max(1, len(ordered) // 3)
        groups[prefix] = {
            "Many": ordered[:third],
            "Med":  ordered[third:len(ordered) - third],
            "Few":  ordered[len(ordered) - third:],
        }
    return groups



def align_clusters(targets, predictions, num_classes):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (predictions, targets), 1)
    rows, columns = linear_sum_assignment(confusion.max() - confusion)
    mapping = np.full(num_classes, -1, dtype=int)
    mapping[rows] = columns
    return mapping[predictions]


def group_accuracy(mapped, targets, classes):
    mask = np.isin(targets, classes)
    return 100.0 * np.mean(mapped[mask] == targets[mask])


def evaluate(checkpoint_path, label, dataset_name, args, device):
    _, _, num_old, num_classes, _ = (None, *DATASETS[dataset_name])
    features, targets = load_features(checkpoint_path, dataset_name, args, device)
    predictions = KMeans(
        n_clusters=num_classes, random_state=args.seed, n_init=args.n_init,
    ).fit_predict(features)
    mapped = align_clusters(targets, predictions, num_classes)
    try:
        from sklearn.metrics import (adjusted_rand_score,
                                     normalized_mutual_info_score)
        nmi = round(float(normalized_mutual_info_score(targets, predictions)), 4)
        ari = round(float(adjusted_rand_score(targets, predictions)), 4)
    except Exception:
        nmi, ari = None, None
    logging.info("Clustering structure: NMI %s | ARI %s", nmi, ari)
    groups = class_groups(dataset_name, args.imb_ratio)
    result = {"method": label, "dataset": dataset_name,
              "checkpoint": os.path.abspath(checkpoint_path)}

    for split, prefix in (("known", "known"), ("novel", "novel")):
        values = [group_accuracy(mapped, targets, groups[split][name])
                  for name in ("Many", "Med", "Few")]
        result.update({f"{prefix}_{name.lower()}": value
                       for name, value in zip(("Many", "Med", "Few"), values)})
        result[f"{prefix}_std"] = float(np.std(values))
        all_classes = np.concatenate([groups[split][name] for name in ("Many", "Med", "Few")])
        result[f"{prefix}_all"] = group_accuracy(mapped, targets, all_classes)
    result["overall"] = 100.0 * np.mean(mapped == targets)
    result["nmi"] = nmi
    result["ari"] = ari
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cifar10", "cifar100", "cub200", "all"), default="all")
    parser.add_argument("--cifar10_bacon_o_checkpoint")
    parser.add_argument("--cifar10_bacon_s_checkpoint")
    parser.add_argument("--cifar100_bacon_o_checkpoint")
    parser.add_argument("--cifar100_bacon_s_checkpoint")
    parser.add_argument("--cub200_bacon_o_checkpoint")
    parser.add_argument("--cub200_bacon_s_checkpoint")
    parser.add_argument("--output", default="RESULTS/table4_bacons.csv")
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--imb_ratio", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--backbone", default="dinov2_vitb14",
                        choices=["dino_vitb16", "dinov2_vitb14", "dinov2_vitb14_reg"],
                        help="Backbone arch matching the checkpoint")
    args = parser.parse_args()

    if args.log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = os.path.join("logs", f"table4_bacons_{timestamp}.log")
    log_stream = setup_logging(args.log_file)
    logging.info("Starting Table 4 BaCon evaluation")
    logging.info("Command: %s", " ".join(sys.argv))
    logging.info("Arguments: %s", vars(args))

    datasets = ("cifar10", "cifar100") if args.dataset == "all" else (args.dataset,)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Device: %s", device)
        results = []
        for dataset_name in datasets:
            checkpoints = {
                "BaCon-O": getattr(args, f"{dataset_name}_bacon_o_checkpoint"),
                "BaCon-S": getattr(args, f"{dataset_name}_bacon_s_checkpoint"),
            }
            for label, checkpoint in checkpoints.items():
                if checkpoint is None:
                    logging.warning("Skipping %s on %s: checkpoint was not provided", label, dataset_name)
                    continue
                logging.info("Evaluating %s on %s", label, dataset_name)
                results.append(evaluate(checkpoint, label, dataset_name, args, device))

        if not results:
            parser.error("provide at least one checkpoint")

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fields = ["method", "dataset", "known_many", "known_med", "known_few",
                  "known_std", "known_all", "novel_many", "novel_med", "novel_few",
                  "novel_std", "novel_all", "overall", "nmi", "ari", "checkpoint"]
        with open(args.output, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        logging.info("Saved results to %s", os.path.abspath(args.output))
        logging.info("Evaluation completed successfully")
    except Exception:
        logging.error("Evaluation failed")
        traceback.print_exc()
        raise
    finally:
        logging.shutdown()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_stream.close()


if __name__ == "__main__":
    main()