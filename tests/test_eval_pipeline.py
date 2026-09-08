import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.part_modules import LatentPartModule, PartAwareModel

class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(768, 768)])
        
    def prepare_tokens(self, x):
        return x
        
    def norm(self, x):
        return x

def test_eval_pipeline():
    B = 4
    N = 196
    d = 768
    
    # Fake input mimicking backbone output
    images = torch.randn(B, N+1, d)
    
    backbone = DummyBackbone()
    part_module = LatentPartModule(dim=d, num_slots=3)
    
    model = PartAwareModel(backbone, part_module)
    
    # Extract features
    z_eval = model(images)
    
    print("Output feature shape:", z_eval.shape)
    
    # Check shape
    assert z_eval.shape == (B, 2 * d), f"Expected shape {(B, 2 * d)}, got {z_eval.shape}"
    
    # Check normalization
    norms = torch.norm(z_eval, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms)), "Eval features are not properly normalized!"
    
    print("-> Concat Inference pipeline works and features are properly normalized.")

if __name__ == '__main__':
    test_eval_pipeline()
    print("\nAll Phase 4 tests passed!")
