import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.part_modules import PartPrototypeBank, compute_fused_ce_loss

def test_gate_dynamics():
    # Test setting
    B = 32
    C = 5
    M = 3
    d = 64
    lambda_part = 0.5
    tau = 0.1
    
    bank = PartPrototypeBank(num_classes=C, num_slots=M, dim=d)
    
    # Initial gates should be 0.5
    initial_a = bank.get_gates()
    assert torch.allclose(initial_a, torch.ones_like(initial_a) * 0.5)
    
    labels = torch.randint(0, C, (B,))
    
    optimizer = optim.SGD(bank.parameters(), lr=5.0) # high learning rate for fast simulation
    
    print("Initial gate for class 0:", bank.get_gates()[0].detach().numpy())
    
    for step in range(100):
        optimizer.zero_grad()
        
        # Simulate extracted part features
        r_norm = torch.randn(B, M, d)
        r_norm = F.normalize(r_norm, dim=-1)
        
        with torch.no_grad():
            P = bank.get_prototypes()
            for i in range(B):
                c = labels[i]
                # Simulating a highly discriminative part: 
                # slot 0 ALWAYS perfectly matches the prototype of the TRUE class
                r_norm[i, 0] = P[c, 0] 
                # slot 1 and 2 are just random noise
        
        # Global branch produces some decent logit for the correct class
        g_global = torch.randn(B, C)
        g_global[torch.arange(B), labels] += 1.0
        
        g_part, s, a = bank(r_norm)
        
        loss_fused, _ = compute_fused_ce_loss(g_global, g_part, labels, lambda_part=lambda_part, tau=tau)
        
        loss_fused.backward()
        optimizer.step()
        
    final_a = bank.get_gates()
    print("\nFinal gate for class 0:", final_a[0].detach().numpy())
    
    print("\nAverage gate values across all classes:")
    print("Slot 0:", final_a[:, 0].mean().item())
    print("Slot 1:", final_a[:, 1].mean().item())
    print("Slot 2:", final_a[:, 2].mean().item())
    
    # If the mathematical derivation in W1 response is correct, 
    # slot 0 gate should be driven towards 1.0, while slots 1 and 2 may decrease or stay low
    assert final_a[:, 0].mean() > 0.85, "Gate collapse happened! Slot 0 failed to open."
    print("\nSUCCESS: Gate dynamics test passed. Discriminative parts cause gates to open, resolving W1.")

if __name__ == "__main__":
    torch.manual_seed(416)
    test_gate_dynamics()
