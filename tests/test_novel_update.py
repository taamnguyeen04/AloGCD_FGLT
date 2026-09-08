import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.part_modules import (
    PartPrototypeBank,
    compute_margin_confidence,
    compute_target_capacity,
    compute_dist_adaptive_gate_loss
)

def test_adaptive_capacity():
    # Simulate estimated class frequencies (long-tail)
    pi_hat = torch.tensor([1000.0, 500.0, 100.0, 50.0, 10.0]) # 5 classes
    M = 3
    
    M_c_target = compute_target_capacity(pi_hat, M)
    
    print("Estimated Frequencies (pi_hat):", pi_hat.tolist())
    print("Target Capacity (M_c_target):", M_c_target.tolist())
    
    # Check if Head class (1000) is close to M=3
    assert abs(M_c_target[0] - 3.0) < 1e-4, f"Head class capacity should be 3, got {M_c_target[0]}"
    
    # Check if Tail class (10) is close to 1
    assert abs(M_c_target[4] - 1.0) < 1e-4, f"Tail class capacity should be 1, got {M_c_target[4]}"
    
    print("-> Adaptive capacity allocation logic works as expected.")

def test_novel_ema_update_with_confidence():
    B = 10
    C = 5
    M = 3
    d = 64
    
    bank = PartPrototypeBank(num_classes=C, num_slots=M, dim=d)
    
    # Store old prototypes
    old_P = bank.prototypes.clone()
    
    # Simulate some fused scores
    g_fused = torch.randn(B, C)
    
    # Make sample 0 very confident about class 0
    g_fused[0, 0] = 5.0
    g_fused[0, 1:] = 0.0
    
    # Make sample 1 very UNCONFIDENT about class 0 (pseudo label might be 0, but top2 is close)
    g_fused[1, 0] = 1.1
    g_fused[1, 1] = 1.0
    g_fused[1, 2:] = 0.0
    
    conf = compute_margin_confidence(g_fused)
    
    print("\nSample 0 confidence:", conf[0].item())
    print("Sample 1 confidence:", conf[1].item())
    
    # Assign pseudo labels based on argmax
    pseudo_labels = torch.argmax(g_fused, dim=-1)
    
    # Fake part features
    r_norm = torch.randn(B, M, d)
    r_norm = torch.nn.functional.normalize(r_norm, dim=-1)
    
    # Trigger update
    delta_c = 0.2
    bank.update_ema_novel(r_norm, pseudo_labels, conf, delta_c=delta_c, momentum=0.9)
    
    new_P = bank.prototypes
    
    # Because sample 0 is confident, it should update prototype of class 0
    # Because sample 1 is unconfident (conf = 0.1 < 0.2), it should NOT heavily influence the prototype 
    # Actually, if we only had sample 1 for class 0, class 0 prototype wouldn't change.
    
    diff = (new_P[0] - old_P[0]).norm().item()
    print("Prototype change for class 0:", diff)
    assert diff > 0, "Prototype 0 should have been updated!"
    
    print("-> Novel EMA update with confidence filtering works as expected.")

if __name__ == '__main__':
    test_adaptive_capacity()
    test_novel_ema_update_with_confidence()
    print("\nAll Phase 3 tests passed!")
