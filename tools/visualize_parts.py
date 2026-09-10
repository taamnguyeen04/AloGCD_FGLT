import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

def plot_attention_maps(image_tensor, attention_map, save_path, M=3):
    """
    image_tensor: (3, H, W) normalized image
    attention_map: (M, 196) attention weights
    """
    # Unnormalize image for visualization (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image_tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    img_uint8 = (img * 255).astype(np.uint8)
    
    fig, axes = plt.subplots(1, M + 1, figsize=(4 * (M + 1), 4))
    
    # Plot original image
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title('Original Image')
    
    # Plot attention for each slot
    # Grid size is dynamic: 196 patches (ViT/16) -> 14x14, 256 (DINOv2/14) -> 16x16
    N = attention_map.shape[1]
    _h = _w = int(round(N ** 0.5))
    assert _h * _w == N, f'non-square patch grid N={N}'
    for m in range(M):
        attn = attention_map[m].reshape(_h, _w).detach().cpu().numpy()
        
        # Resize to 224x224
        attn_resized = cv2.resize(attn, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        # Normalize to 0-1 for colormap
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        
        # Apply colormap (JET)
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(img_uint8, 0.5, heatmap, 0.5, 0)
        
        axes[m + 1].imshow(overlay)
        axes[m + 1].axis('off')
        axes[m + 1].set_title(f'Slot {m+1} Attention')
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_argmax_segmap(image_tensor, attention_map, save_path, alpha=0.45):
    """One-panel part segmentation: each patch colored by winning slot.

    image_tensor: (3, H, W) normalized image.
    attention_map: (M, N) with square N (196->14x14, 256->16x16).
    Clearer than M heatmaps for papers: shows how the bird is partitioned
    into beak/wing/tail regions at a glance.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image_tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    img_uint8 = (img * 255).astype(np.uint8)

    M, N = attention_map.shape
    h = w = int(round(N ** 0.5))
    assert h * w == N, f'non-square patch grid N={N}'
    winner = attention_map.argmax(dim=0).reshape(h, w).detach().cpu().numpy()

    seg_small = np.zeros((h, w, 3), dtype=np.uint8)
    cmap = plt.get_cmap('tab10')
    for m in range(M):
        color = (np.array(cmap(m % 10)[:3]) * 255).astype(np.uint8)
        seg_small[winner == m] = color
    seg = cv2.resize(seg_small, (224, 224), interpolation=cv2.INTER_NEAREST)

    overlay = cv2.addWeighted(img_uint8, 1.0 - alpha, seg, alpha, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(seg)
    axes[1].set_title(f'Argmax parts ({h}x{h} grid)')
    axes[1].axis('off')
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    for m in range(M):
        color = cmap(m % 10)[:3]
        axes[2].plot([], [], marker='s', color=color, linestyle='',
                     label=f'Slot {m + 1}')
    axes[2].legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_retrieval_grid(patches, save_path, title='Top patches per prototype',
                        cols=9):
    """Grid of retrieved patches: what does a prototype 'mean'?

    patches: list of rows; each row is a list of (C, h, w) uint8/float tensors
    (one row = one prototype's top-k). ProtoPNet-style evidence figure.
    """
    rows = len(patches)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(1.6 * cols, 1.6 * rows),
                             squeeze=False)
    fig.suptitle(title, fontsize=11)
    for r, row in enumerate(patches):
        for c in range(cols):
            ax = axes[r][c]
            ax.axis('off')
            if c < len(row):
                p = row[c]
                if torch.is_tensor(p):
                    p = p.detach().cpu().numpy()
                if p.shape[0] in (1, 3):
                    p = np.transpose(p, (1, 2, 0))
                if p.max() <= 1.0:
                    p = (p * 255).astype(np.uint8)
                ax.imshow(p.astype(np.uint8), interpolation='nearest')
                if c == 0:
                    ax.set_ylabel(f'proto {r + 1}', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_gate_heatmap(gate_values, class_frequencies, save_path):
    """
    gate_values: (C, M)
    class_frequencies: (C,)
    """
    # Sort classes by frequency (descending)
    sorted_indices = torch.argsort(class_frequencies, descending=True)
    sorted_gates = gate_values[sorted_indices].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(sorted_gates.T, aspect='auto', cmap='viridis', interpolation='none')
    
    plt.colorbar(label='Gate Activation Value (0 to 1)')
    plt.xlabel('Classes (Sorted by Frequency: Head -> Tail)')
    plt.ylabel('Latent Part Slot')
    plt.title('Distribution-Adaptive Gate Heatmap across Classes')
    plt.yticks(np.arange(sorted_gates.shape[1]), [f'Slot {m+1}' for m in range(sorted_gates.shape[1])])
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def test_visualizations():
    os.makedirs('dev_outputs/visualizations', exist_ok=True)
    
    # 1. Mock Attention Map Visualization (both ViT/16 N=196 and DINOv2/14 N=256)
    print("Generating mock Attention Map...")
    image = torch.randn(3, 224, 224)
    for N, tag in ((196, 'vit16'), (256, 'dinov214')):
        # create a focused attention blob for each slot
        attn_map = torch.rand(3, N) * 0.1
        attn_map[0, N // 4:N // 4 + 10] = 1.0  # Slot 1 focuses on top region
        attn_map[1, N // 2:N // 2 + 10] = 1.0  # Slot 2 focuses on center
        attn_map[2, 3 * N // 4:3 * N // 4 + 10] = 1.0  # Slot 3 focuses on bottom
        plot_attention_maps(image, attn_map,
                            f'dev_outputs/visualizations/mock_attention_{tag}.png')
        plot_argmax_segmap(image, torch.softmax(attn_map * 5.0, dim=0),
                           f'dev_outputs/visualizations/mock_segmap_{tag}.png')

    # 1b. Mock retrieval grid (3 prototypes x 9 patches of noise)
    print("Generating mock retrieval grid...")
    grid = [[torch.rand(3, 32, 32) for _ in range(9)] for _ in range(3)]
    plot_retrieval_grid(grid, 'dev_outputs/visualizations/mock_retrieval.png',
                        title='Mock: top-9 patches per prototype')

    # Non-square N must fail loudly instead of silently misshaping
    # (NB: 100 is square (10x10) — use a truly non-square N like 150.)
    try:
        plot_attention_maps(image, torch.rand(3, 150), 'dev_outputs/visualizations/never.png')
        _guard_ok = False
    except AssertionError as e:
        _guard_ok = 'non-square' in str(e)
    assert _guard_ok, 'non-square guard failed for N=150'
    print('Non-square guard OK.')
    
    # 2. Mock Gate Heatmap
    print("Generating mock Gate Heatmap...")
    C = 200
    M = 3
    # Simulating Head classes have gate ~1.0 for all slots
    # Tail classes have gate ~1.0 for slot 1, but ~0 for slot 2 and 3
    gate_values = torch.zeros(C, M)
    class_freq = torch.exp(-torch.linspace(0, 5, C)) * 1000 # Exponential decay
    
    for c in range(C):
        gate_values[c, 0] = 0.9 + torch.rand(1) * 0.1
        if class_freq[c] > 100:
            gate_values[c, 1] = 0.8 + torch.rand(1) * 0.2
            gate_values[c, 2] = 0.7 + torch.rand(1) * 0.3
        elif class_freq[c] > 20:
            gate_values[c, 1] = 0.5 + torch.rand(1) * 0.3
            gate_values[c, 2] = 0.1 + torch.rand(1) * 0.2
        else:
            gate_values[c, 1] = 0.0 + torch.rand(1) * 0.1
            gate_values[c, 2] = 0.0 + torch.rand(1) * 0.1
            
    plot_gate_heatmap(gate_values, class_freq, 'dev_outputs/visualizations/mock_gate_heatmap.png')
    
    print("Done! Visualizations saved to dev_outputs/visualizations/")

if __name__ == '__main__':
    test_visualizations()
