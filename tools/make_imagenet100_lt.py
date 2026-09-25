"""Resample ImageNet-100 (exported by prepare_imagenet100_hf.py) to a target
imbalance ratio, default imb10.

Recipe (mirrors tools/make_cifar_lt_splits.py + the shipped LT txt style):
  - Per-class profile = exponential ``count[rank] = int(img_max *
    (1/imb) ** (rank / (C-1)))``, C=100.
  - Classes randomly permuted (seed, default 416); profile rank follows
    permuted position. Known identity stays labels 0..49 (fixed by
    ``data/imagenet.py``) -- counts land randomly on known/novel, exactly
    like the shipped LT txt (head 1280 sits on label 95).
  - ``img_max`` default 700: below the smallest HF class pool (754), so no
    class gets capped. Tail = 70, total ~27k.
  - Images come from the cached ``clane9/imagenet-100`` parquet (no
    re-download); ``{root}/train`` is rebuilt, ``{root}/val`` is (re)built
    if incomplete (already 50/class), LT train txt rewritten (``*.orig``
    backup kept).

Usage (on server, from repo root)::

    python tools/make_imagenet100_lt.py --imb-ratio 10
    # full pools on disk + imb10 txt (slower, ~130k imgs):
    python tools/make_imagenet100_lt.py --imb-ratio 10 --export-full
    # --root defaults to config.imagenet_root; --img-max / --seed optional
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter, defaultdict
from pathlib import Path


def synset_order_from_lt_txt(lt_txt: Path) -> list[str]:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imb-ratio", type=int, default=10)
    ap.add_argument("--img-max", type=int, default=700,
                    help="head-class count (must be <= smallest class pool 754)")
    ap.add_argument("--seed", type=int, default=416)
    ap.add_argument("--root", default=None,
                    help="ImageNet root (default: config.imagenet_root)")
    ap.add_argument("--export-full", action="store_true",
                    help="export ENTIRE pools to {root}/train with stable "
                    "names ({syn}_{parquet-idx}.JPEG), skipping existing "
                    "files; the imb txt then references this full set")
    ap.add_argument("--splits-dir", default="splits/imagenet100")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    import random
    from datasets import load_dataset

    if args.root is None:
        from config import imagenet_root
        args.root = imagenet_root
    root = Path(args.root)
    splits = Path(args.splits_dir)
    lt_txt = splits / "imageNet_100_LT_train.txt"

    order = synset_order_from_lt_txt(lt_txt)

    print("[info] loading clane9/imagenet-100 parquet (cached) ...")
    ds = load_dataset("clane9/imagenet-100")
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(ds["train"]["label"]):
        by_label[int(lab)].append(idx)
    pools = [len(by_label[c]) for c in range(100)]
    print(f"[info] pools: min={min(pools)} max={max(pools)}")
    assert args.img_max <= min(pools), (
        f"img_max {args.img_max} > smallest pool {min(pools)}")

    rng = random.Random(args.seed)
    perm = rng.sample(range(100), 100)
    want = {}
    for rank, c in enumerate(perm):
        n = int(args.img_max * ((1.0 / args.imb_ratio) ** (rank / 99.0)))
        want[c] = min(n, len(by_label[c]))
    total = sum(want.values())
    print(f"[info] imb{args.imb_ratio}: total={total} "
          f"head={max(want.values())} tail={min(want.values())}")

    # ---- val: (re)build if incomplete (deterministic names, 50/class) ----
    val_dir = root / "val"
    n_val = sum(1 for _ in val_dir.rglob("*.JPEG")) if val_dir.exists() else 0
    if n_val != 5000:
        print(f"[val] found {n_val}/5000, (re)exporting ...")
        val_counter = Counter()
        for idx in range(len(ds["validation"])):
            row = ds["validation"][idx]
            c = int(row["label"])
            syn = order[c]
            d = val_dir / syn
            d.mkdir(parents=True, exist_ok=True)
            k = val_counter[c]
            name = f"ILSVRC2012_val_{syn}_{k:05d}.JPEG"
            p = d / name
            if not p.exists():
                row["image"].convert("RGB").save(p,
                                                 quality=args.jpeg_quality)
            val_counter[c] += 1
            if (idx + 1) % 1000 == 0:
                print(f"[val] {idx + 1}/{len(ds['validation'])}")
        print("[val] per-class counts:",
              f"min={min(val_counter.values())} "
              f"max={max(val_counter.values())}")
    else:
        print("[val] already complete (5000), skipped")

    # Rebuild train from scratch (old export would leave strays) --
    # unless --export-full, which keeps the full pools on disk.
    train_dir = root / "train"
    if args.export_full:
        print("[train] exporting FULL pools (skip existing) ...")
        n_full = 0
        for c in range(100):
            syn = order[c]
            d = train_dir / syn
            d.mkdir(parents=True, exist_ok=True)
            for idx in sorted(by_label[c]):
                name = f"{syn}_{idx:06d}.JPEG"
                p = d / name
                if not p.exists():
                    img = ds["train"][idx]["image"]
                    img.convert("RGB").save(p, quality=args.jpeg_quality)
                n_full += 1
            if (c + 1) % 10 == 0:
                print(f"[train-full] {c + 1}/100 classes")
        print(f"[train-full] {n_full} images on disk")
        name_of = lambda syn, idx: f"{syn}_{idx:06d}.JPEG"
    else:
        if train_dir.exists():
            print(f"[info] clearing {train_dir} ...")
            shutil.rmtree(train_dir)
        name_of = None  # subset mode uses k-based names below

    lines: list[str] = []
    n_saved = 0
    for c in range(100):
        syn = order[c]
        d = train_dir / syn
        d.mkdir(parents=True, exist_ok=True)
        chosen = rng.sample(sorted(by_label[c]), want[c])
        for k, idx in enumerate(sorted(chosen)):
            if args.export_full:
                name = name_of(syn, idx)  # already on disk
            else:
                img = ds["train"][idx]["image"]
                name = f"{syn}_{k:05d}.JPEG"
                img.convert("RGB").save(d / name, quality=args.jpeg_quality)
            lines.append(f"train/{syn}/{name} {c}\n")
            n_saved += 1
        if (c + 1) % 10 == 0:
            print(f"[train] {c + 1}/100 classes, {n_saved} imgs")
    print(f"[train] saved {n_saved} images")

    bak = lt_txt.with_suffix(lt_txt.suffix + ".orig")
    if not bak.exists():
        shutil.copy(lt_txt, bak)
        print(f"[txt] backed up {lt_txt.name} -> {bak.name}")
    with open(lt_txt, "w") as f:
        f.writelines(lines)
    print(f"[txt] wrote {lt_txt} ({len(lines)} lines, imb{args.imb_ratio})")


if __name__ == "__main__":
    main()
