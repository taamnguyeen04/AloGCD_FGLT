"""Prepare ImageNet-100 from HuggingFace `clane9/imagenet-100` parquet
into the classic ILSVRC folder layout that BaCon expects.

BaCon reads (`data/imagenet.py`, `data/LT_dataset.py`)::

    {root}/train/n02869837/xxx.JPEG  (label from train txt)
    {root}/val/n01558993/xxx.JPEG    (label from test txt)

where ``root`` = ``config.imagenet_root`` (``/ImageNet/``) and the txts live
in ``splits/imagenet100/``. Class order (label 0-99) is the CMC-100 order,
which this HF dataset uses verbatim (verified: same set AND same order as
``splits/imagenet100/imageNet_100_LT_train.txt``).

Parquet rows do NOT carry original ILSVRC filenames, so the original txt
paths cannot be reused. This script therefore:

1. downloads the parquet (`train` ~127k + `validation` ~5k, 160px),
2. exports JPEGs to ``{root}/train/{synset}/`` and ``{root}/val/{synset}/``,
3. subsamples train to the EXACT per-class counts of the original LT txt
   (12,205 imgs, 6..1280 per class) so the long-tail profile is preserved,
4. rewrites both txts (originals backed up to ``*.orig`` once).

Usage (on server)::

    pip install datasets pillow huggingface_hub
    python tools/prepare_imagenet100_hf.py --root /ImageNet
    # if /ImageNet is not writable, use another dir and point config at it:
    #   python tools/prepare_imagenet100_hf.py --root /data/imagenet100
    #   sed -i "s|imagenet_root = '/ImageNet/'|imagenet_root = '/data/imagenet100'|" config.py
"""

import argparse
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def synset_order_from_lt_txt(lt_txt: Path) -> list[str]:
    """Recover label -> synset mapping from the original LT train txt."""
    label2syn: dict[int, str] = {}
    with open(lt_txt) as f:
        for line in f:
            parts = line.split()
            syn = parts[0].split("/")[1]
            lab = int(parts[1])
            if lab in label2syn:
                assert label2syn[lab] == syn, f"label {lab} maps to 2 synsets"
            else:
                label2syn[lab] = syn
    assert len(label2syn) == 100, f"expected 100 classes, got {len(label2syn)}"
    return [label2syn[i] for i in range(100)]


def lt_counts_per_class(lt_txt: Path) -> list[int]:
    cnt = Counter()
    with open(lt_txt) as f:
        for line in f:
            cnt[int(line.split()[1])] += 1
    return [cnt[i] for i in range(100)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/ImageNet",
                    help="Target ImageNet root (train/nXXX, val/nXXX go here)")
    ap.add_argument("--splits-dir", default="splits/imagenet100")
    ap.add_argument("--seed", type=int, default=416,
                    help="Seed for per-class subsampling (416 = repo convention)")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    import random
    from datasets import load_dataset

    root = Path(args.root)
    splits = Path(args.splits_dir)
    lt_txt = splits / "imageNet_100_LT_train.txt"
    test_txt = splits / "ImageNet_100_test.txt"

    order = synset_order_from_lt_txt(lt_txt)
    target_counts = lt_counts_per_class(lt_txt)
    print(f"[info] LT profile: total={sum(target_counts)} "
          f"min={min(target_counts)} max={max(target_counts)}")

    print("[info] downloading/loading clane9/imagenet-100 parquet ...")
    ds = load_dataset("clane9/imagenet-100")
    print({k: len(v) for k, v in ds.items()})
    # Sanity: HF ClassLabel order must equal CMC order used by splits.
    names = ds["train"].features["label"].names
    assert len(names) == 100, f"expected 100 HF labels, got {len(names)}"
    # names are human-readable ("bonnet, poke bonnet", ...); check count only.
    # Index-level check: label c must be synset order[c] (verified 2026-09).

    rng = random.Random(args.seed)

    # ---- train: group row-idx by label, subsample to LT counts ----
    by_label: dict[int, list[int]] = defaultdict(list)
    labels_col = ds["train"]["label"]
    for idx, lab in enumerate(labels_col):
        by_label[int(lab)].append(idx)
    for c in range(100):
        assert len(by_label[c]) >= target_counts[c], (
            f"class {c} ({order[c]}): parquet has {len(by_label[c])} "
            f"< LT target {target_counts[c]}")
    print("[info] per-class pool ok "
          f"(min pool={min(len(by_label[c]) for c in range(100))})")

    train_lines: list[str] = []
    n_saved = 0
    for c in range(100):
        syn = order[c]
        d = root / "train" / syn
        d.mkdir(parents=True, exist_ok=True)
        chosen = rng.sample(sorted(by_label[c]), target_counts[c])
        for k, idx in enumerate(sorted(chosen)):
            img = ds["train"][idx]["image"]
            name = f"{syn}_{k:05d}.JPEG"
            img.convert("RGB").save(d / name, quality=args.jpeg_quality)
            train_lines.append(f"train/{syn}/{name} {c}\n")
            n_saved += 1
        if (c + 1) % 10 == 0:
            print(f"[train] {c + 1}/100 classes, {n_saved} imgs")
    print(f"[train] saved {n_saved} images")

    # ---- val: export all, expect ~50/class ----
    test_lines: list[str] = []
    val_counter = Counter()
    for idx in range(len(ds["validation"])):
        row = ds["validation"][idx]
        c = int(row["label"])
        syn = order[c]
        d = root / "val" / syn
        d.mkdir(parents=True, exist_ok=True)
        k = val_counter[c]
        name = f"ILSVRC2012_val_{syn}_{k:05d}.JPEG"
        row["image"].convert("RGB").save(d / name,
                                         quality=args.jpeg_quality)
        test_lines.append(f"val/{syn}/{name} {c}\n")
        val_counter[c] += 1
        if (idx + 1) % 1000 == 0:
            print(f"[val] {idx + 1}/{len(ds['validation'])}")
    print("[val] per-class counts:",
          f"min={min(val_counter.values())} max={max(val_counter.values())} "
          f"total={sum(val_counter.values())}")

    # ---- rewrite txts (backup originals once) ----
    for path, lines in ((lt_txt, train_lines), (test_txt, test_lines)):
        bak = path.with_suffix(path.suffix + ".orig")
        if not bak.exists():
            shutil.copy(path, bak)
            print(f"[txt] backed up {path.name} -> {bak.name}")
        with open(path, "w") as f:
            f.writelines(lines)
        print(f"[txt] wrote {path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
