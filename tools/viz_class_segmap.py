"""Real-image part segmap from a parts checkpoint (Teacher_C3 etc).

For each requested class, takes K test images, runs backbone+LatentPartModule,
saves per-image: original | argmax parts | overlay (which prototype colors which region)
plus per-slot heatmaps.

Usage:
    python tools/viz_class_segmap.py --ckpt ckpts/Teacher_C3_r1.pt --classes 5,150 --max-per-class 4 --backbone dinov2_vitb14 --out dev_outputs/segmap_TeacherC3
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--classes', required=True,
                    help="'5,150' -> 2 classes x K images")
    ap.add_argument('--cub-root', default=None)
    ap.add_argument('--backbone', default='dinov2_vitb14')
    ap.add_argument('--max-per-class', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    import torch
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.part_viz_common import (
        cub_test_subset, imagenet_test_transform,
        load_backbone_and_parts, part_forward,
    )
    from tools.visualize_parts import plot_argmax_segmap, plot_attention_maps

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    classes = [int(t.strip()) for t in str(a.classes).split(',') if t.strip() != '']
    cub_root = a.cub_root or os.environ.get('CUB_ROOT', 'D:/data/CUB_200_2011')

    backbone, part_module, part_bank, meta = load_backbone_and_parts(
        a.ckpt, a.backbone, device)
    print(f"[segmap] M={meta['M']} C={meta['C']} weights_from={meta['weights_from']}")

    transform = imagenet_test_transform()
    items = cub_test_subset(cub_root, classes, transform, a.max_per_class, a.seed)
    if not items:
        print('[segmap] no images — check --cub-root / --classes')
        return 1
    print(f'[segmap] {len(items)} images.')

    os.makedirs(a.out, exist_ok=True)
    with torch.no_grad():
        for i in range(0, len(items), a.batch_size):
            chunk = items[i:i + a.batch_size]
            imgs = torch.stack([im for im, _, _ in chunk]).to(device)
            _, A = part_forward(backbone, part_module, imgs)  # (B,M,N)
            for (img_t, label, path), attn in zip(chunk, A):
                base = f"class{label}_{os.path.splitext(os.path.basename(path))[0]}"
                plot_argmax_segmap(img_t.cpu(), attn.cpu(),
                                   os.path.join(a.out, base + '_segmap.png'))
                plot_attention_maps(img_t.cpu(), attn.cpu(),
                                    os.path.join(a.out, base + '_attn.png'),
                                    M=attn.shape[0])
    print(f'[segmap] saved to {a.out} (each image: _segmap.png + _attn.png)')
    print('[segmap] READ: _segmap = argmax slot per patch (mau nao = prototype do thong tri vung do); _attn = heatmap rieng tung slot.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
