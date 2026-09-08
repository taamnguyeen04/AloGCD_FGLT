import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentPartModule(nn.Module):
    def __init__(self, dim, num_slots=3, temperature=0.07):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.temperature = temperature
        
        # Queries for latent parts, initialized orthogonally
        self.queries = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.orthogonal_(self.queries)
        
    def forward(self, patch_tokens):
        """
        patch_tokens: (B, N, d) where N is number of patches (e.g. 196)
        Returns:
            r_norm: (B, M, d) L2-normalized part features
            A: (B, M, N) Attention maps
        """
        B, N, d = patch_tokens.shape
        
        # Dot product attention: (M, d) x (B, d, N) -> (B, M, N)
        # Scale by sqrt(d) for stable gradients; temperature sharpens the distribution
        logits = torch.einsum('md,bnd->bmn', self.queries, patch_tokens) / (math.sqrt(d) * self.temperature)
        
        # Softmax over patches
        A = F.softmax(logits, dim=-1)
        
        # Weighted pooling: (B, M, N) x (B, N, d) -> (B, M, d)
        r = torch.einsum('bmn,bnd->bmd', A, patch_tokens)
        
        # Normalize
        r_norm = F.normalize(r, dim=-1)
        
        return r_norm, A

class PartAwareModel(nn.Module):
    """
    Wrapper for evaluation. Combines CLS token from backbone
    with GATE-WEIGHTED part features from LatentPartModule.
    Cùng công thức với concat eval trong bacon.test() (single source of truth):
      pred = argmax(g_part); r_pool = Σ_m a[pred,m] r_m / Σ_m a[pred,m].
    part_bank=None -> fallback mean-pool (giữ compat với test cũ).
    grad_from_block: matches BaCon's training convention (frozen layers before this block).
    """
    def __init__(self, backbone, part_module, part_bank=None, grad_from_block=11):
        super().__init__()
        self.backbone = backbone
        self.part_module = part_module
        self.part_bank = part_bank
        self.grad_from_block = grad_from_block

    def forward(self, images):
        x = self.backbone.prepare_tokens(images)
        for i, blk in enumerate(self.backbone.blocks):
            if i < self.grad_from_block:
                with torch.no_grad():
                    x = blk(x)
            else:
                x = blk(x)
        x = self.backbone.norm(x)  # (B, N+1, d)

        z_cls = x[:, 0]          # (B, d)
        patch_tokens = x[:, 1:]  # (B, N, d)

        r_norm, _ = self.part_module(patch_tokens)  # (B, M, d)
        if self.part_bank is not None:
            with torch.no_grad():
                g_part, _, a = self.part_bank(r_norm)
                pred = g_part.argmax(dim=-1)  # (B,)
                a_pred = a[pred]              # (B, M)
            r_pool = (r_norm * a_pred.unsqueeze(-1)).sum(dim=1) / (a_pred.sum(dim=1, keepdim=True) + 1e-6)
        else:
            r_pool = r_norm.mean(dim=1)       # (B, d) fallback

        z_eval = torch.cat([z_cls, r_pool], dim=-1)  # (B, 2d)
        z_eval = F.normalize(z_eval, p=2, dim=-1)

        return z_eval

def compute_spatial_loss(A, gamma_ent=0.1):
    """
    A: (B, M, N)
    Computes spatial diversity and concentration loss.
    """
    B, M, N = A.shape
    
    # 1. Diversity loss: penalize overlap
    if M > 1:
        # (B, M, N) x (B, N, M) -> (B, M, M)
        overlap = torch.einsum('bmn,bkn->bmk', A, A)
        mask = 1.0 - torch.eye(M, device=A.device).unsqueeze(0)
        # Average over B and M*(M-1)
        loss_overlap = (overlap * mask).sum() / (B * M * (M - 1))
    else:
        loss_overlap = torch.tensor(0.0, device=A.device)
        
    # 2. Entropy / Concentration penalty
    # Want to minimize entropy so attention is focused
    entropy = - (A * torch.log(A + 1e-8)).sum(dim=-1) # (B, M)
    loss_ent = entropy.mean()
    
    loss_total = loss_overlap + gamma_ent * loss_ent
    return loss_total, loss_overlap, loss_ent

class PartPrototypeBank(nn.Module):
    def __init__(self, num_classes, num_slots, dim):
        super().__init__()
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.dim = dim
        
        # Prototype bank P: (C, M, d) — registered as buffer (EMA only, no grad)
        proto = torch.empty(num_classes * num_slots, dim)
        nn.init.orthogonal_(proto)
        self.register_buffer('prototypes', proto.view(num_classes, num_slots, dim))
        
        # Adaptive gates u: (C, M)
        # Initialize to 0.0, so sigmoid(0.0) = 0.5
        self.gate_logits = nn.Parameter(torch.zeros(num_classes, num_slots))
        
    def get_gates(self):
        # a_{c,m} = sigmoid(u_{c,m})
        return torch.sigmoid(self.gate_logits)
        
    def get_prototypes(self):
        # L2 normalize on d dimension
        return F.normalize(self.prototypes, dim=-1)

    def forward(self, r_norm):
        """
        r_norm: (B, M, d)
        Returns:
            g_part: (B, C) normalized part score
            s: (B, C, M) cosine similarities
            a: (C, M) gates
        """
        P = self.get_prototypes() # (C, M, d)
        a = self.get_gates()      # (C, M)
        
        # s_{i,c,m} = <r_{i,m}, p_{c,m}>
        # r_norm: (B, M, d), P: (C, M, d) -> s: (B, C, M)
        s = torch.einsum('bmd,cmd->bcm', r_norm, P)
        
        # g_{i,c}^{part} = \sum_m a_{c,m} s_{i,c,m} / (\sum_m a_{c,m} + \epsilon)
        weighted_s = s * a.unsqueeze(0) # (B, C, M)
        numerator = weighted_s.sum(dim=-1) # (B, C)
        denominator = a.sum(dim=-1).unsqueeze(0) + 1e-6 # (1, C)
        
        g_part = numerator / denominator # (B, C)
        
        return g_part, s, a
    @torch.no_grad()
    def update_ema(self, r_norm, labels, momentum=0.99):
        """
        Update known class prototypes using EMA.
        r_norm: (B, M, d)
        labels: (B,)
        """
        P = self.prototypes
        for c in torch.unique(labels):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            # Centroid of samples in this batch for class c
            r_c = r_norm[mask].mean(dim=0) # (M, d)
            # Update prototype
            P[c] = F.normalize(momentum * P[c] + (1 - momentum) * r_c, dim=-1)
        self.prototypes.copy_(P)

    @torch.no_grad()
    def update_ema_novel(self, r_norm, pseudo_labels, w, threshold=0.7, momentum=0.99):
        """
        Update novel class prototypes using EMA with confidence filtering.
        r_norm: (B, M, d)
        pseudo_labels: (B,)
        w: (B,) probability confidence [0,1]
        """
        P = self.prototypes
        for c in torch.unique(pseudo_labels):
            mask = (pseudo_labels == c) & (w > threshold)
            if mask.sum() == 0:
                continue
            
            w_c = w[mask].view(-1, 1, 1)  # (N_c, 1, 1)
            r_c = (r_norm[mask] * w_c).sum(dim=0) / (w_c.sum(dim=0) + 1e-6)  # (M, d)
            
            P[c] = F.normalize(momentum * P[c] + (1 - momentum) * r_c, dim=-1)
        self.prototypes.copy_(P)

def compute_fused_ce_loss(g_global, g_part, labels, lambda_part=0.5, tau=0.1):
    """
    Fused CE loss on labeled data.
    """
    g_fused = g_global + lambda_part * g_part
    loss_fused = F.cross_entropy(g_fused / tau, labels)
    return loss_fused, g_fused

def compute_margin_confidence(g_fused):
    """
    g_fused: (B, C)
    Returns margin confidence conf(x) = top1 - top2
    """
    top2_vals, _ = torch.topk(g_fused, k=2, dim=-1)
    conf = top2_vals[:, 0] - top2_vals[:, 1]
    return conf

def compute_target_capacity(pi_hat, M):
    """
    pi_hat: (C,) estimated class frequencies
    M: int, maximum number of slots
    Returns M_c_target: (C,)
    """
    pi_min = pi_hat.min()
    pi_max = pi_hat.max()
    
    if pi_max <= pi_min:
        return torch.ones_like(pi_hat) * M
    
    log_pi = torch.log(pi_hat + 1e-8)
    log_min = torch.log(pi_min + 1e-8)
    log_max = torch.log(pi_max + 1e-8)
    
    ratio = (log_pi - log_min) / (log_max - log_min + 1e-8)
    M_c = 1.0 + (M - 1.0) * ratio
    return torch.clamp(M_c, 1.0, float(M))

def compute_dist_adaptive_gate_loss(a, M_c_target):
    """
    a: (C, M) gates
    M_c_target: (C,)
    """
    gate_sum = a.sum(dim=-1) # (C,)
    loss_dist_gate = F.mse_loss(gate_sum, M_c_target)
    return loss_dist_gate

def compute_gate_reg_loss(a, M_min=1.0, lambda_bound=0.05):
    """
    Regularize gates to be binary and maintain lower bound.
    """
    # 1. Binarization loss: sum a(1-a)
    loss_bin = (a * (1 - a)).mean()
    
    # 2. Lower-bound penalty: max(0, M_min - sum a)
    gate_sum = a.sum(dim=-1) # (C,)
    loss_bound = F.relu(M_min - gate_sum).mean()
    
    loss_gate = loss_bin + lambda_bound * loss_bound
    return loss_gate, loss_bin, loss_bound
