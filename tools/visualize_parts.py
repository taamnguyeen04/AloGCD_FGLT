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
    for m in range(M):
        # Reshape 196 -> 14x14
        attn = attention_map[m].reshape(14, 14).detach().cpu().numpy()
        
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
    
    # 1. Mock Attention Map Visualization
    print("Generating mock Attention Map...")
    image = torch.randn(3, 224, 224)
    # create a focused attention blob for each slot
    attn_map = torch.rand(3, 196) * 0.1
    attn_map[0, 50:60] = 1.0 # Slot 1 focuses on top region
    attn_map[1, 100:110] = 1.0 # Slot 2 focuses on center
    attn_map[2, 150:160] = 1.0 # Slot 3 focuses on bottom
    
    plot_attention_maps(image, attn_map, 'dev_outputs/visualizations/mock_attention.png')
    
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
