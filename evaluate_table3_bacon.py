"""Reproduce BaCon-O/BaCon-S evaluation with full breakdown.

Metrics reported per method:
  Known: Many | Med | Few | Std | All
  Novel: Many | Med | Few | Std | All
  Overall

Also generates a 2x2 t-SNE visualization PNG (All / Many / Med / Few).
"""

import argparse
import csv
import datetime
import logging
import os
import sys
import traceback

import numpy as np

# ── Dataset registry ──────────────────────────────────────────────────────────
DATASETS = {
    "cifar10":  {"num_old": 5,   "num_classes": 10,  "index_prefix": "cifar10_k5"},
    "cifar100": {"num_old": 80,  "num_classes": 100, "index_prefix": "cifar100_k80"},
    "cub200":   {"num_old": 100, "num_classes": 200, "index_prefix": None},
}


# ── Logging helpers ───────────────────────────────────────────────────────────
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


# ── Dataset helpers ───────────────────────────────────────────────────────────
def get_dataset_class_and_root(dataset_name):
    """Return (DatasetClass, root_path) for the given dataset name."""
    from config import cifar_10_root, cifar_100_root, cub_root
    from data.cifar import CustomCIFAR10, CustomCIFAR100
    from data.cub import CUBDataset

    mapping = {
        "cifar10":  (CustomCIFAR10,  cifar_10_root),
        "cifar100": (CustomCIFAR100, cifar_100_root),
        "cub200":   (CUBDataset,     cub_root),
    }
    if dataset_name not in mapping:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return mapping[dataset_name]


def class_groups(dataset_name, imb_ratio):
    """Return per-split Many/Med/Few class arrays.

    Returns dict:
        {"known": {"Many": [...], "Med": [...], "Few": [...]},
         "novel": {"Many": [...], "Med": [...], "Few": [...]}}
    """
    import torch
    cfg = DATASETS[dataset_name]
    num_old    = cfg["num_old"]
    num_classes= cfg["num_classes"]
    prefix     = cfg["index_prefix"]

    DatasetClass, root = get_dataset_class_and_root(dataset_name)

    if prefix is not None:
        # CIFAR: read split index files to get training counts
        index_dir = os.path.join("data_uq_idxs", f"{prefix}_imb{imb_ratio}")
        split_files = ("l_k_uq_idxs.pt", "unl_k_uq_idxs.pt", "unl_unk_uq_idxs.pt")
        indices = np.concatenate([
            np.asarray(torch.load(os.path.join(index_dir, name), map_location="cpu"))
            for name in split_files
        ])
        train_dataset = DatasetClass(root=root, train=True, transform=None, download=True)
        counts = np.bincount(np.asarray(train_dataset.targets)[indices], minlength=num_classes)
    else:
        # CUB200: compute counts directly from the full training set
        train_dataset = DatasetClass(root=root, train=True, transform=None)
        counts = np.bincount(np.array(train_dataset.targets), minlength=num_classes)

    groups = {}
    for split_name, classes in (("known", np.arange(num_old)),
                                 ("novel", np.arange(num_old, num_classes))):
        ordered = classes[np.argsort(-counts[classes])]
        third   = max(1, len(ordered) // 3)
        groups[split_name] = {
            "Many": ordered[:third],
            "Med":  ordered[third: len(ordered) - third],
            "Few":  ordered[len(ordered) - third:],
        }
    return groups


def group_accuracy(mapped, targets, classes):
    """Accuracy for samples whose true label is in `classes`."""
    mask = np.isin(targets, classes)
    if mask.sum() == 0:
        return 0.0
    return 100.0 * np.mean(mapped[mask] == targets[mask])


# ── Model helpers ─────────────────────────────────────────────────────────────
def load_backbone(checkpoint_path, device):
    import torch
    logging.info("Loading DINO ViT-B/16 backbone")
    backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "cl_backbone" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'cl_backbone': {checkpoint_path}")
    backbone.load_state_dict(checkpoint["cl_backbone"])
    logging.info("Loaded contrastive backbone: %s", checkpoint_path)
    return backbone.to(device).eval()


def extract_features(backbone, loader, device):
    import torch
    features, targets = [], []
    with torch.no_grad():
        for batch_index, (images, labels, _) in enumerate(loader, start=1):
            output = backbone(images.to(device))
            output = torch.nn.functional.normalize(output, dim=-1)
            features.append(output.cpu().numpy())
            targets.append(labels.numpy())
            logging.info("Feature batch %d complete (%d images)", batch_index, len(labels))
    return np.concatenate(features), np.concatenate(targets).astype(int)


def align_clusters(targets, predictions, num_classes):
    """Hungarian-align cluster IDs to class IDs."""
    from scipy.optimize import linear_sum_assignment
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (predictions, targets), 1)
    rows, cols = linear_sum_assignment(confusion.max() - confusion)
    mapping = np.full(num_classes, -1, dtype=int)
    mapping[rows] = cols
    return mapping[predictions]


# ── t-SNE visualization ───────────────────────────────────────────────────────
def plot_tsne(features, targets, groups, dataset_name, out_path):
    """Save a 2x2 t-SNE figure: All / Many / Med / Few."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    logging.info("Running t-SNE (this may take a while)...")
    emb = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=1000).fit_transform(features)
    logging.info("t-SNE done.")

    # Build combined Many/Med/Few arrays (known + novel)
    subplot_groups = {
        "All classes":  np.unique(targets),
        "Many":  np.concatenate([groups["known"]["Many"], groups["novel"]["Many"]]),
        "Median": np.concatenate([groups["known"]["Med"],  groups["novel"]["Med"]]),
        "Few":   np.concatenate([groups["known"]["Few"],  groups["novel"]["Few"]]),
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"t-SNE Visualization of BaCon Features ({dataset_name.upper()})", fontsize=14)
    cmap = plt.cm.get_cmap("tab20", max(targets) + 1)

    for ax, (title, cls_array) in zip(axes.flat, subplot_groups.items()):
        mask = np.isin(targets, cls_array)
        sc = ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=targets[mask], cmap=cmap,
            s=6, alpha=0.7,
        )
        # Legend (show at most 20 classes to avoid clutter)
        unique_cls = np.unique(targets[mask])
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=cmap(c / (max(targets) + 1)), markersize=6,
                       label=str(c))
            for c in unique_cls[:20]
        ]
        ax.legend(handles=handles, title="Class", fontsize=5, title_fontsize=6,
                  loc="upper right", ncol=2)
        ax.set_title(title)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("t-SNE saved to %s", out_path)


# ── Core evaluation ───────────────────────────────────────────────────────────
def evaluate_checkpoint(checkpoint_path, label, dataset, args, device):
    """Evaluate a single checkpoint. Returns a result dict."""
    import torch
    from sklearn.cluster import KMeans
    from torch.utils.data import DataLoader
    from data.augmentations import get_transform

    cfg = DATASETS[dataset]
    num_old    = cfg["num_old"]
    num_classes= cfg["num_classes"]

    DatasetClass, root = get_dataset_class_and_root(dataset)

    _, test_transform = get_transform(
        "imagenet", image_size=224,
        args=argparse.Namespace(interpolation=3, crop_pct=0.875),
    )
    # CUB200 doesn't support `download` kwarg
    extra_kwargs = {} if dataset == "cub200" else {"download": True}
    test_dataset = DatasetClass(root=root, train=False, transform=test_transform, **extra_kwargs)
    loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    logging.info("[%s] Extracting %s test features (%d images)", label, dataset, len(test_dataset))
    backbone = load_backbone(checkpoint_path, device)
    features, targets = extract_features(backbone, loader, device)
    logging.info("[%s] Feature shape: %s", label, features.shape)

    logging.info("[%s] Running K-Means with %d clusters", label, num_classes)
    predictions = KMeans(
        n_clusters=num_classes, random_state=args.seed, n_init=args.n_init,
    ).fit_predict(features)

    mapped = align_clusters(targets, predictions, num_classes)
    groups = class_groups(dataset, args.imb_ratio)

    result = {
        "method":     label,
        "dataset":    dataset,
        "checkpoint": os.path.abspath(checkpoint_path),
    }

    for split in ("known", "novel"):
        values = [group_accuracy(mapped, targets, groups[split][g])
                  for g in ("Many", "Med", "Few")]
        result[f"{split}_many"] = round(values[0], 4)
        result[f"{split}_med"]  = round(values[1], 4)
        result[f"{split}_few"]  = round(values[2], 4)
        result[f"{split}_std"]  = round(float(np.std(values)), 4)
        all_cls = np.concatenate([groups[split][g] for g in ("Many", "Med", "Few")])
        result[f"{split}_all"]  = round(group_accuracy(mapped, targets, all_cls), 4)

    result["overall"] = round(100.0 * float(np.mean(mapped == targets)), 4)

    logging.info(
        "[%s] Known Many %.2f | Med %.2f | Few %.2f | All %.2f",
        label,
        result["known_many"], result["known_med"], result["known_few"], result["known_all"],
    )
    logging.info(
        "[%s] Novel Many %.2f | Med %.2f | Few %.2f | All %.2f",
        label,
        result["novel_many"], result["novel_med"], result["novel_few"], result["novel_all"],
    )
    logging.info("[%s] Overall %.2f", label, result["overall"])

    # Return features and groups alongside result so caller can optionally run t-SNE
    return result, features, targets, groups


# ── Entry point ───────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "method", "dataset",
    "known_many", "known_med", "known_few", "known_std", "known_all",
    "novel_many", "novel_med", "novel_few", "novel_std", "novel_all",
    "overall", "checkpoint",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    parser.add_argument("--bacon_o_checkpoint", default=None)
    parser.add_argument("--bacon_s_checkpoint", default=None)
    parser.add_argument("--output",      default="table3_bacon_results.csv")
    parser.add_argument("--log_file",    default=None)
    parser.add_argument("--batch_size",  type=int,   default=512)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--n_init",      type=int,   default=10)
    parser.add_argument("--imb_ratio",   type=int,   default=100,
                        help="Imbalance ratio (used for CIFAR index files)")
    parser.add_argument("--tsne_output", default=None,
                        help="Path to save t-SNE PNG. If omitted, t-SNE is skipped.")
    args = parser.parse_args()

    if args.bacon_o_checkpoint is None and args.bacon_s_checkpoint is None:
        parser.error("Provide --bacon_o_checkpoint and/or --bacon_s_checkpoint")

    if args.log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = os.path.join("logs", f"table3_{args.dataset}_{timestamp}.log")
    log_stream = setup_logging(args.log_file)

    logging.info("Starting Table 3 BaCon evaluation (with Many/Med/Few breakdown)")
    logging.info("Command: %s", " ".join(sys.argv))
    logging.info("Arguments: %s", vars(args))

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Device: %s", device)

        results = []
        all_features, all_targets, all_groups = None, None, None
        for label, ckpt in (("BaCon-O", args.bacon_o_checkpoint),
                             ("BaCon-S", args.bacon_s_checkpoint)):
            if ckpt is None:
                continue
            result, features, targets, groups = evaluate_checkpoint(ckpt, label, args.dataset, args, device)
            results.append(result)
            # Keep last checkpoint's features/targets/groups for t-SNE
            all_features, all_targets, all_groups = features, targets, groups

        # ── Save CSV first (always, before t-SNE) ────────────────────────────
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(results)
        logging.info("Saved results: %s", os.path.abspath(args.output))
        logging.info("Evaluation completed successfully")

        # ── t-SNE (optional, after CSV is safe) ──────────────────────────────
        if args.tsne_output and all_features is not None:
            try:
                # Subsample to max 5000 points to keep t-SNE fast
                MAX_TSNE = 5000
                if len(all_features) > MAX_TSNE:
                    logging.info("Subsampling %d → %d points for t-SNE", len(all_features), MAX_TSNE)
                    rng = np.random.default_rng(0)
                    idx = rng.choice(len(all_features), MAX_TSNE, replace=False)
                    all_features = all_features[idx]
                    all_targets  = all_targets[idx]
                plot_tsne(all_features, all_targets, all_groups, args.dataset, args.tsne_output)
            except Exception as tsne_err:
                logging.warning("t-SNE failed (non-fatal): %s", tsne_err)
                traceback.print_exc()

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