"""Subset clustering + prototype figure (viec 5.3) — post-hoc, no training.

Picks 7-8 classes by role (explicit --classes, see plan2 5.3 recipe), embeds
their part-pooled features plus the M part-prototypes per class with t-SNE
(joint fit: sklearn TSNE has no transform for new points), and draws:
  (a) sample scatter (circles=known, triangles=novel),
  (b) prototypes as stars sized by gate a_{c,m} of the TRUE class,
  (c) run twice (base ckpt vs method ckpt) for a before/after panel.

Needs: post-5.3.0 checkpoint (part_bank inside), CUB test data, backbone
weights (torch.hub cache; downloads ~300MB first time).

Usage:
    python tools/plot_subset_prototypes.py --ckpt <model.pt> --classes 5,12,30,86,150,170,180,190 \\
        --cub-root D:/data/CUB_200_2011 --backbone dinov2_vitb14 --out dev_outputs/subset
"""

import argparse
import os
import sys


def resolve_classes(arg):
    """'5,12, 30' -> [5, 12, 30]. Raises on empty/garbage (fail loudly)."""
    classes = []
    for tok in str(arg).split(','):
        tok = tok.strip()
        if not tok:
            continue
        c = int(tok)
        if c not in classes:
            classes.append(c)
    if not classes:
        raise ValueError('empty --classes (e.g. --classes 5,12,30,86,150,170,180,190)')
    return classes


def gate_star_sizes(gates_c, base=60.0, span=220.0):
    """Map gate a in [0,1] -> markersize so open gates pop, closed fade."""
    import numpy as np
    g = np.asarray(list(gates_c), dtype=float)
    return (base + span * g).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--classes', required=True)
    ap.add_argument('--cub-root', default=None)
    ap.add_argument('--backbone', default='dinov2_vitb14')
    ap.add_argument('--num-labeled', type=int, default=100)
    ap.add_argument('--max-per-class', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    import numpy as np
    import torch

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    classes = resolve_classes(a.classes)
    cub_root = a.cub_root or os.environ.get('CUB_ROOT', 'D:/data/CUB_200_2011')

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.part_viz_common import (cub_test_subset, imagenet_test_transform,
                                       load_backbone_and_parts)

    os.makedirs(a.out, exist_ok=True)
    backbone, part_module, part_bank, meta = load_backbone_and_parts(
        a.ckpt, a.backbone, device)
    print(f'[subset] ckpt parts: M={meta["M"]} C={meta["C"]} weights_from={meta["weights_from"]}')
    transform = imagenet_test_transform()
    items = cub_test_subset(cub_root, classes, transform, a.max_per_class, a.seed)
    if not items:
        print('[subset] no images found for requested classes — check --cub-root.')
        return 1
    print(f'[subset] {len(items)} images across {len(classes)} classes.')

    # Sample part features (mean-pooled r_norm = part space) + labels
    from tools.part_viz_common import part_forward
    feats, labels = [], []
    with torch.no_grad():
        for i in range(0, len(items), a.batch_size):
            chunk = items[i:i + a.batch_size]
            imgs = torch.stack([im for im, _, _ in chunk]).to(device)
            r_norm, _ = part_forward(backbone, part_module, imgs)
            pool = r_norm.mean(dim=1)
            feats.append(torch.nn.functional.normalize(pool, dim=-1).cpu())
            labels.extend([t for _, t, _ in chunk])
    feats = torch.cat(feats).numpy()
    labels = np.array(labels)

    # Prototypes (normalized) + gates of TRUE classes
    with torch.no_grad():
        protos = part_bank.get_prototypes().cpu().numpy()  # (C, M, d)
        gates = part_bank.get_gates().cpu().numpy()        # (C, M)
    M = protos.shape[1]
    P, P_class, P_slot, P_size = [], [], [], []
    for c in classes:
        for m in range(M):
            P.append(protos[c, m])
            P_class.append(c)
            P_slot.append(m)
            P_size.append(gates[c, m])
    P = np.array(P)
    P_size = gate_star_sizes(P_size)

    # Joint t-SNE fit (samples + prototypes together)
    from sklearn.manifold import TSNE
    perp = max(5, min(30, len(feats) + len(P) - 1))
    emb = TSNE(n_components=2, random_state=a.seed, perplexity=perp,
               init='random').fit_transform(np.concatenate([feats, P]))
    E, PE = emb[:len(feats)], emb[len(feats):]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = plt.get_cmap('tab10')
    for i, c in enumerate(classes):
        sel = labels == c
        is_old = c < a.num_labeled
        ax.scatter(E[sel, 0], E[sel, 1], s=22, color=cmap(i % 10), alpha=0.65,
                   marker='o' if is_old else '^', linewidths=0, label=f'class {c}')
    for j in range(len(P)):
        c = P_class[j]
        i = classes.index(c)
        ax.scatter([PE[j, 0]], [PE[j, 1]], s=P_size[j], color=cmap(i % 10),
                   marker='*', edgecolors='black', linewidths=0.8, zorder=5)
    ax.scatter([], [], s=160, color='gray', marker='*', edgecolors='black',
               label='prototype (size = gate)')
    ax.set_title(f'Subset part-space t-SNE + prototypes\n'
                 f'{os.path.basename(a.ckpt)} | circles=known triangles=novel')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, ncol=2, loc='best')
    fig.tight_layout()
    path = os.path.join(a.out, 'subset_prototypes.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[subset] saved {path}')
    print('[subset] CAPTION NOTE (paper honesty): t-SNE joint-fit on samples + '
          'prototypes; stars sized by TRUE-class gate a_{c,m}; part space '
          '(mean-pooled r_norm), not CLS space.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
