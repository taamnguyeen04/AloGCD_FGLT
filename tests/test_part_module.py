import sys
import os
import torch
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.part_modules import LatentPartModule, compute_spatial_loss

class TestLatentPartModule(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.N = 196
        self.d = 768
        self.M = 3
        
        self.module = LatentPartModule(dim=self.d, num_slots=self.M)
        self.patch_tokens = torch.randn(self.B, self.N, self.d, requires_grad=True)

    def test_forward_shapes(self):
        r_norm, A = self.module(self.patch_tokens)
        
        self.assertEqual(r_norm.shape, (self.B, self.M, self.d))
        self.assertEqual(A.shape, (self.B, self.M, self.N))
        
        # Check normalization
        norms = torch.norm(r_norm, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms)))

    def test_spatial_loss(self):
        r_norm, A = self.module(self.patch_tokens)
        loss_total, loss_overlap, loss_ent = compute_spatial_loss(A)
        
        self.assertFalse(torch.isnan(loss_total))
        self.assertFalse(torch.isinf(loss_total))
        self.assertTrue(loss_overlap.item() >= 0.0)
        self.assertTrue(loss_ent.item() >= 0.0)
        
    def test_gradient_flow(self):
        r_norm, A = self.module(self.patch_tokens)
        loss, _, _ = compute_spatial_loss(A)
        
        # Add a dummy loss on r_norm to also get gradients from it
        loss_total = loss + r_norm.sum()
        loss_total.backward()
        
        # Check if gradients exist
        self.assertIsNotNone(self.module.queries.grad)
        self.assertIsNotNone(self.patch_tokens.grad)
        
        # Check no NaN/Inf in gradients
        self.assertFalse(torch.isnan(self.module.queries.grad).any())
        self.assertFalse(torch.isinf(self.module.queries.grad).any())
        self.assertFalse(torch.isnan(self.patch_tokens.grad).any())
        self.assertFalse(torch.isinf(self.patch_tokens.grad).any())

if __name__ == '__main__':
    unittest.main()
