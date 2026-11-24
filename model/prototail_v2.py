"""ProtoTail v2: evidence ledger, calibrated selection and pure helpers.

This module is intentionally independent from ``model.bacon`` so the v1
training loop remains runnable.  The training script only calls the pieces it
enables with feature flags.

Core ideas implemented here
----------------------------
* Time-decayed evidence ledger (``EvidenceLedger``) with raw and shrunk
  prototypes.  Raw prototypes are used for classification logits; shrunk
  prototypes are used only for conservative selection/verification.
* Per-class score normalisation against non-members.  A shared (global)
  centring would be a softmax no-op; non-member normalisation is not.
* Sample-level credibility ``lambda_i = lambda * min_{c in top-k} Z_c``.
* Beta-Binomial lower confidence bounds, BH/FDR selection and a distinct
  ``n_c,eff`` counter (one sample id contributes at most once to n_eff).
* Helpers for holdout calibration, novel-group purity and split/merge guards.

All tensors in the ledger are registered buffers, so saving it together with a
checkpoint keeps the prototype state.  Python-only history (prediction votes,
per-class member sets) is intentionally cheap to rebuild.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, dim=dim, eps=eps)


def _to_numpy(x: Any, dtype=np.float64) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype)
    return np.asarray(x, dtype=dtype)


def _sid(kind: str, value: int) -> Tuple[str, int]:
    return (str(kind), int(value))


def _safe_quantile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return float(default)
    return float(np.quantile(np.asarray(vals, dtype=np.float64), q))


# ---------------------------------------------------------------------------
# Beta / IMDB utilities
# ---------------------------------------------------------------------------

def _beta_ppf(q: float, a: float, b: float) -> float:
    """Inverse Beta CDF with a dependency-free fallback."""
    a = max(float(a), 1e-6)
    b = max(float(b), 1e-6)
    try:
        from scipy.stats import beta as _beta_dist  # type: ignore
        return float(_beta_dist.ppf(float(q), a, b))
    except Exception:
        # Normal approximation is only used in minimal environments (tests).
        mean = a / (a + b)
        var = a * b / ((a + b) ** 2 * (a + b + 1.0))
        z = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(float(q))))
        return float(np.clip(mean + z * math.sqrt(max(var, 1e-12)), 0.0, 1.0))


def _beta_cdf(x: float, a: float, b: float) -> float:
    """Beta CDF with a dependency-free fallback."""
    a = max(float(a), 1e-6)
    b = max(float(b), 1e-6)
    try:
        from scipy.stats import beta as _beta_dist  # type: ignore
        return float(_beta_dist.cdf(float(x), a, b))
    except Exception:
        # Crude but monotone approximation around the mean.
        mean = a / (a + b)
        var = a * b / ((a + b) ** 2 * (a + b + 1.0))
        sd = math.sqrt(max(var, 1e-12))
        return float(np.clip(0.5 * (1.0 + math.erf((float(x) - mean) / (sd * math.sqrt(2.0)))), 0.0, 1.0))


def fit_beta_prior(
    rates: Sequence[float],
    counts: Optional[Sequence[float]] = None,
    default_p: float = 0.85,
    min_m: float = 2.0,
    max_m: float = 200.0,
) -> Tuple[float, float]:
    """Fit a Beta prior by moments.

    Returns ``(p_bar, m)`` where ``m`` is the IMDB/Bühlmann prior strength.
    Empty input returns the conservative default used for a class without
    calibration samples.
    """
    vals = np.asarray([float(v) for v in rates if v is not None and np.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return float(default_p), float(min_m)
    w = None
    if counts is not None:
        cand = np.asarray([float(v) for v in counts], dtype=np.float64)
        if cand.size == vals.size and np.all(np.isfinite(cand)) and cand.sum() > 0:
            w = np.clip(cand, 1e-6, None)
    p_bar = float(np.average(vals, weights=w)) if w is not None else float(vals.mean())
    p_bar = float(np.clip(p_bar, 0.02, 0.98))
    if w is not None:
        var = float(np.average((vals - p_bar) ** 2, weights=w))
    else:
        var = float(vals.var())
    if not np.isfinite(var) or var <= 1e-8:
        return p_bar, float(max_m)
    m = p_bar * (1.0 - p_bar) / var - 1.0
    if not np.isfinite(m):
        m = min_m
    return p_bar, float(np.clip(m, min_m, max_m))


def beta_posterior(
    success: float,
    n: float,
    p_bar: float,
    m: float,
) -> Tuple[float, float]:
    """Posterior parameters for a Beta-Binomial precision/agreement rate."""
    n = max(float(n), 0.0)
    success = float(np.clip(float(success), 0.0, n)) if n > 0 else 0.0
    a = success + max(float(m), 1e-6) * float(np.clip(p_bar, 1e-6, 1.0 - 1e-6))
    b = (n - success) + max(float(m), 1e-6) * (1.0 - float(np.clip(p_bar, 1e-6, 1.0 - 1e-6)))
    return float(a), float(b)


def beta_lcb(
    success: float,
    n: float,
    p_bar: float,
    m: float,
    quantile: float = 0.10,
) -> float:
    a, b = beta_posterior(success, n, p_bar, m)
    return _beta_ppf(float(quantile), a, b)


def beta_upper_tail_pvalue(
    success: float,
    n: float,
    p_bar: float,
    m: float,
    threshold: float,
) -> float:
    """Posterior probability that the true rate is *not* above threshold.

    A small value is evidence that precision/agreement exceeds ``threshold``;
    this is the p-value used by the BH procedure below.
    """
    a, b = beta_posterior(success, n, p_bar, m)
    return _beta_cdf(float(threshold), a, b)


def benjamini_hochberg(pvalues: Sequence[float], q: float = 0.10) -> np.ndarray:
    """BH step-up procedure.  Returns a boolean selection mask."""
    p = np.asarray([float(v) if v is not None and np.isfinite(float(v)) else 1.0
                    for v in pvalues], dtype=np.float64)
    m = int(p.size)
    out = np.zeros(m, dtype=bool)
    if m == 0:
        return out
    order = np.argsort(p)
    ranked = p[order]
    passed = ranked <= (float(q) * (np.arange(m) + 1) / float(m))
    if not np.any(passed):
        return out
    k = int(np.where(passed)[0].max())
    out[order[: k + 1]] = True
    return out


def gate_records(
    records: Sequence[Dict[str, Any]],
    *,
    threshold: float,
    p_bar: float,
    m: float,
    quantile: float = 0.10,
    use_fdr: bool = True,
    fdr_q: float = 0.10,
) -> List[Dict[str, Any]]:
    """Apply LCB + optional BH-FDR to class/group records.

    Each record must contain ``success`` and ``n``.  Returned records are copies
    with ``lcb``, ``pvalue`` and ``selected`` fields added.
    """
    enriched: List[Dict[str, Any]] = []
    pvals: List[float] = []
    for rec in records:
        item = dict(rec)
        s = float(item.get('success', 0.0) or 0.0)
        n = float(item.get('n', 0.0) or 0.0)
        item['lcb'] = beta_lcb(s, n, p_bar, m, quantile=quantile)
        item['pvalue'] = beta_upper_tail_pvalue(s, n, p_bar, m, threshold)
        item['passes_lcb'] = bool(item['lcb'] >= float(threshold))
        enriched.append(item)
        pvals.append(float(item['pvalue']))
    if use_fdr and enriched:
        bh = benjamini_hochberg(pvals, q=fdr_q)
        for item, passed in zip(enriched, bh):
            item['passes_fdr'] = bool(passed)
            item['selected'] = bool(item['passes_lcb'] and passed)
    else:
        for item in enriched:
            item['passes_fdr'] = True
            item['selected'] = bool(item['passes_lcb'])
    return enriched


# ---------------------------------------------------------------------------
# Holdout / dataset helpers
# ---------------------------------------------------------------------------

def dataset_labels(dataset: Any) -> List[int]:
    """Best-effort recursive label extraction for the datasets in this repo."""
    if dataset is None:
        return []
    if isinstance(dataset, torch.utils.data.Subset):
        parent = dataset_labels(dataset.dataset)
        return [int(parent[int(i)]) for i in dataset.indices]
    for attr in ('targets', 'labels'):
        vals = getattr(dataset, attr, None)
        if vals is not None:
            try:
                return [int(v) for v in vals]
            except Exception:
                pass
    samples = getattr(dataset, 'samples', None)
    if samples is not None:
        try:
            return [int(s[1]) for s in samples]
        except Exception:
            pass
    children = getattr(dataset, 'datasets', None)
    if children is not None:
        return [v for child in children for v in dataset_labels(child)]
    if hasattr(dataset, 'labelled_dataset') and hasattr(dataset, 'unlabelled_dataset'):
        return dataset_labels(dataset.labelled_dataset) + dataset_labels(dataset.unlabelled_dataset)
    return []


def dataset_uq_ids(dataset: Any) -> List[int]:
    """Best-effort recursive uq-id extraction."""
    if dataset is None:
        return []
    if isinstance(dataset, torch.utils.data.Subset):
        parent = dataset_uq_ids(dataset.dataset)
        return [int(parent[int(i)]) for i in dataset.indices]
    vals = getattr(dataset, 'uq_idxs', None)
    if vals is not None:
        try:
            return [int(v) for v in vals]
        except Exception:
            pass
    children = getattr(dataset, 'datasets', None)
    if children is not None:
        return [v for child in children for v in dataset_uq_ids(child)]
    return []


def set_dataset_transform(dataset: Any, transform: Any) -> None:
    """Set a transform recursively (supports Subset/ConcatDataset wrappers)."""
    if dataset is None:
        return
    if isinstance(dataset, torch.utils.data.Subset):
        set_dataset_transform(dataset.dataset, transform)
        return
    children = getattr(dataset, 'datasets', None)
    if children is not None:
        for child in children:
            set_dataset_transform(child, transform)
        return
    if hasattr(dataset, 'transform'):
        try:
            dataset.transform = transform
        except Exception:
            pass


def make_holdout_split(
    train_dataset: Any,
    ratio: float = 0.15,
    min_labels: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """Move a stratified fraction of labeled samples into the unlabeled pool.

    The labels of the moved samples are returned in ``uq_to_label`` and must be
    used for calibration only.  The function mutates ``train_dataset`` in place
    (the caller already owns a deep-copied MergedDataset) and returns metadata.
    """
    ratio = float(ratio)
    if ratio <= 0.0:
        return {'enabled': False, 'uq_to_label': {}, 'n_holdout': 0}

    labelled = train_dataset.labelled_dataset
    labels = dataset_labels(labelled)
    if not labels:
        return {'enabled': False, 'uq_to_label': {}, 'n_holdout': 0,
                'reason': 'no-labels'}
    uq_ids = dataset_uq_ids(labelled)
    if len(uq_ids) != len(labels):
        uq_ids = list(range(len(labels)))

    rng = np.random.RandomState(int(seed))
    by_class: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(idx)

    keep_idx: List[int] = []
    hold_idx: List[int] = []
    for label, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        if n < int(min_labels):
            keep_idx.extend(idxs)
            continue
        n_hold = int(round(ratio * n))
        if n_hold <= 0 and ratio > 0:
            n_hold = 1
        n_hold = min(n_hold, max(0, n - 1))
        hold_idx.extend(idxs[:n_hold])
        keep_idx.extend(idxs[n_hold:])

    if not hold_idx:
        return {'enabled': False, 'uq_to_label': {}, 'n_holdout': 0,
                'reason': 'ratio-too-small'}
    if not keep_idx:
        return {'enabled': False, 'uq_to_label': {}, 'n_holdout': 0,
                'reason': 'would-empty-labeled-set'}

    # Preserve a deterministic order for both halves.
    keep_idx = sorted(keep_idx)
    hold_idx = sorted(hold_idx)
    hold_subset = torch.utils.data.Subset(labelled, hold_idx)
    keep_subset = torch.utils.data.Subset(labelled, keep_idx)
    uq_to_label = {int(uq_ids[i]): int(labels[i]) for i in hold_idx}

    # Holdout samples are treated exactly like the rest of D_U during training.
    old_unlabelled = train_dataset.unlabelled_dataset
    train_dataset.labelled_dataset = keep_subset
    train_dataset.unlabelled_dataset = torch.utils.data.ConcatDataset(
        [hold_subset, old_unlabelled]
    )
    return {
        'enabled': True,
        'uq_to_label': uq_to_label,
        'n_holdout': len(hold_idx),
        'n_kept': len(keep_idx),
        'dataset': hold_subset,
    }


# ---------------------------------------------------------------------------
# Evidence ledger
# ---------------------------------------------------------------------------

class EvidenceLedger(nn.Module):
    """Time-decayed per-class evidence with raw/shrunk prototypes.

    Parameters
    ----------
    num_classes, num_slots, dim:
        Number of head dimensions, latent slots and feature width.
    decay:
        Per-epoch decay ``beta`` (default 0.5).
    history_epochs:
        Number of epochs kept for sample stability (default 3).
    shrink_k:
        Initial Bühlmann/IMDB prior strength.  With ``shrink_k_mode='buhlmann'``
        it is re-estimated at the end of each epoch from within/between class
        variance; otherwise it stays fixed.
    """

    def __init__(
        self,
        num_classes: int,
        num_slots: int,
        dim: int,
        *,
        decay: float = 0.5,
        history_epochs: int = 3,
        shrink_k: float = 10.0,
        shrink_k_mode: str = 'fixed',
        num_known: Optional[int] = None,
        min_stat_count: float = 10.0,
        min_sigma: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_slots = int(num_slots)
        self.dim = int(dim)
        self.decay = float(decay)
        self.history_epochs = max(1, int(history_epochs))
        self.shrink_k_mode = str(shrink_k_mode)
        self.num_known = int(num_known) if num_known is not None else int(num_classes)
        self.min_stat_count = float(min_stat_count)
        self.min_sigma = float(min_sigma)
        self.eps = float(eps)

        proto = _normalize(torch.randn(self.num_classes, self.num_slots, self.dim), dim=-1)
        self.register_buffer('S', torch.zeros(self.num_classes, self.num_slots, self.dim))
        self.register_buffer('W', torch.zeros(self.num_classes))
        self.register_buffer('raw_proto', proto.clone())
        self.register_buffer('shrunk_proto', proto.clone())
        self.register_buffer('prior', _normalize(proto.mean(dim=0), dim=-1))
        self.register_buffer('n_eff', torch.zeros(self.num_classes))
        self.register_buffer('Z', torch.zeros(self.num_classes))
        self.register_buffer('support', torch.zeros(self.num_classes))
        self.register_buffer('shrink_k', torch.tensor(float(shrink_k)))
        self.register_buffer('nu', torch.zeros(self.num_classes))
        self.register_buffer('sigma', torch.ones(self.num_classes))
        self.register_buffer('stat_sum', torch.zeros(self.num_classes))
        self.register_buffer('stat_sq', torch.zeros(self.num_classes))
        self.register_buffer('stat_count', torch.zeros(self.num_classes))
        self.register_buffer('within_sum', torch.zeros(self.num_classes))
        self.register_buffer('within_count', torch.zeros(self.num_classes))
        self.register_buffer('core_sum', torch.zeros(self.num_classes, self.dim))
        self.register_buffer('core_count', torch.zeros(self.num_classes))
        self.register_buffer('history_agreement_bar', torch.tensor(0.5))

        self._epoch_S = torch.zeros_like(self.S)
        self._epoch_W = torch.zeros_like(self.W)
        self._epoch_stat_sum = torch.zeros_like(self.stat_sum)
        self._epoch_stat_sq = torch.zeros_like(self.stat_sq)
        self._epoch_stat_count = torch.zeros_like(self.stat_count)
        self._epoch_within_sum = torch.zeros_like(self.within_sum)
        self._epoch_within_count = torch.zeros_like(self.within_count)
        self._epoch_members: List[Dict[Tuple[str, int], float]] = [
            {} for _ in range(self.num_classes)
        ]
        self._epoch_preds: Dict[Tuple[str, int], Tuple[int, int, int]] = {}
        self._member_epoch: List[Dict[Tuple[str, int], Tuple[float, int]]] = [
            {} for _ in range(self.num_classes)
        ]
        self._history: Dict[Tuple[str, int], List[Tuple[int, int, int]]] = {}
        self.promoted_ids = set()
        self.current_epoch = -1

    # -- bookkeeping -----------------------------------------------------
    def set_support_counts(self, counts: Dict[int, int]) -> None:
        with torch.no_grad():
            for c, n in counts.items():
                c = int(c)
                if 0 <= c < self.num_classes:
                    self.support[c] = float(n)

    def mark_promoted(self, uq_ids: Iterable[int]) -> None:
        for uq in uq_ids:
            try:
                self.promoted_ids.add(int(uq))
            except Exception:
                pass

    def begin_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self._epoch_S.zero_()
        self._epoch_W.zero_()
        self._epoch_stat_sum.zero_()
        self._epoch_stat_sq.zero_()
        self._epoch_stat_count.zero_()
        self._epoch_within_sum.zero_()
        self._epoch_within_count.zero_()
        self._epoch_members = [{} for _ in range(self.num_classes)]
        self._epoch_preds = {}

    # -- online update ---------------------------------------------------
    @torch.no_grad()
    def observe_batch(
        self,
        r_norm: torch.Tensor,
        labels: torch.Tensor,
        mask_lab: torch.Tensor,
        uq_ids: Sequence[int],
        pred_global: Optional[torch.Tensor] = None,
        pred_global_v1: Optional[torch.Tensor] = None,
        pred_part: Optional[torch.Tensor] = None,
        probs_global: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate one training batch.

        ``r_norm`` has shape ``(B, M, d)``.  Labeled/promoted samples contribute
        with weight 1; unlabeled samples contribute to their global-argmax class
        with weight ``rho_i * p(c|x_i)``, where ``rho_i`` is the sample
        stability from the last ``history_epochs`` epochs.
        """
        if r_norm.numel() == 0:
            return
        device = self.S.device
        r = r_norm.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device).long()
        mask_lab = mask_lab.to(device=device).bool()
        bsz = r.shape[0]
        if len(uq_ids) != bsz:
            # Defensive: truncate / pad with unique placeholders.
            uq_list = [int(x) for x in list(uq_ids)[:bsz]]
            uq_list.extend(range(10 ** 9, 10 ** 9 + bsz - len(uq_list)))
        else:
            uq_list = [int(x) for x in uq_ids]

        if pred_global is None:
            pred_global = labels.clone()
        pred_global = pred_global.to(device=device).long()
        pred_global_v1 = (pred_global if pred_global_v1 is None
                          else pred_global_v1.to(device=device).long())
        pred_part = (pred_global if pred_part is None
                     else pred_part.to(device=device).long())

        weight = torch.zeros(bsz, device=device, dtype=torch.float32)
        target = labels.clone()

        # Labeled and promoted samples: one certain vote.
        for i in range(bsz):
            if bool(mask_lab[i]):
                target[i] = labels[i]
                weight[i] = 1.0
            else:
                c = int(pred_global[i].item())
                if c < 0 or c >= self.num_classes:
                    weight[i] = 0.0
                    continue
                target[i] = c
                rho = self.stability(uq_list[i], c)
                conf = 1.0
                if probs_global is not None:
                    try:
                        conf = float(probs_global[i, c].item())
                    except Exception:
                        conf = 1.0
                weight[i] = float(np.clip(rho * conf, 0.0, 1.0))

        valid = weight > 0
        if not bool(valid.any()):
            return
        one_hot = F.one_hot(target.clamp(0, self.num_classes - 1),
                            num_classes=self.num_classes).float()
        weighted = one_hot * weight.unsqueeze(-1)
        self._epoch_S.add_(torch.einsum('bc,bmd->cmd', weighted, r))
        self._epoch_W.add_(weighted.sum(dim=0))

        # Count distinct sample ids, not repeated observations.
        for i in range(bsz):
            w = float(weight[i].item())
            if w <= 0.0:
                continue
            c = int(target[i].item())
            if not (0 <= c < self.num_classes):
                continue
            kind = 'U' if (not bool(mask_lab[i])) or (uq_list[i] in self.promoted_ids) else 'L'
            sid = _sid(kind, uq_list[i])
            old = self._epoch_members[c].get(sid, 0.0)
            if w > old:
                self._epoch_members[c][sid] = w
            if not bool(mask_lab[i]):
                self._epoch_preds[sid] = (
                    int(pred_global[i].item()),
                    int(pred_global_v1[i].item()),
                    int(pred_part[i].item()),
                )

        # Non-member score statistics: s_{i,c} with c != ref(i).
        s = torch.einsum('bmd,cmd->bcm', r, self.raw_proto)
        s_mean = s.mean(dim=-1)
        ref = torch.where(mask_lab, labels, pred_global)
        nonmember = (ref.unsqueeze(1) != torch.arange(
            self.num_classes, device=device).unsqueeze(0)).float()
        self._epoch_stat_sum.add_((s_mean * nonmember).sum(dim=0))
        self._epoch_stat_sq.add_(((s_mean ** 2) * nonmember).sum(dim=0))
        self._epoch_stat_count.add_(nonmember.sum(dim=0))

        # Within-class dispersion estimate for the Bühlmann k.
        if bool(valid.any()):
            for i in torch.nonzero(valid, as_tuple=False).flatten().tolist():
                c = int(target[i].item())
                if not (0 <= c < self.num_classes):
                    continue
                cos = float(s[i, c].mean().item())
                w = float(weight[i].item())
                self._epoch_within_sum[c] += w * max(0.0, 1.0 - cos)
                self._epoch_within_count[c] += w

    # -- epoch close -----------------------------------------------------
    @torch.no_grad()
    def end_epoch(self, epoch: Optional[int] = None) -> Dict[str, float]:
        if epoch is None:
            epoch = self.current_epoch + 1
        epoch = int(epoch)
        self.current_epoch = epoch

        self.S.mul_(self.decay).add_(self._epoch_S)
        self.W.mul_(self.decay).add_(self._epoch_W)
        self.stat_sum.mul_(self.decay).add_(self._epoch_stat_sum)
        self.stat_sq.mul_(self.decay).add_(self._epoch_stat_sq)
        self.stat_count.mul_(self.decay).add_(self._epoch_stat_count)
        self.within_sum.mul_(self.decay).add_(self._epoch_within_sum)
        self.within_count.mul_(self.decay).add_(self._epoch_within_count)

        # Merge distinct-member sets and prune samples older than the window.
        cutoff = epoch - self.history_epochs + 1
        for c in range(self.num_classes):
            store = self._member_epoch[c]
            for sid, w in self._epoch_members[c].items():
                old = store.get(sid)
                if old is None or w >= old[0]:
                    store[sid] = (float(w), epoch)
            for sid, (w, ep) in list(store.items()):
                if ep < cutoff:
                    del store[sid]
            self.n_eff[c] = float(sum(min(1.0, w) for w, _ in store.values()))

        # Raw prototype = normalized decayed evidence sum.
        norm = self.S.norm(dim=-1, keepdim=True)
        has_evidence = norm.squeeze(-1) > self.eps
        new_raw = _normalize(self.S, dim=-1)
        self.raw_proto.copy_(torch.where(has_evidence.unsqueeze(-1), new_raw, self.raw_proto))

        # Global prior over classes, weighted by their evidence mass.
        w = self.W.clamp_min(0.0)
        if float(w.sum().item()) > self.eps:
            prior = (self.raw_proto * w[:, None, None]).sum(dim=0) / w.sum().clamp_min(self.eps)
        else:
            prior = self.raw_proto.mean(dim=0)
        self.prior.copy_(_normalize(prior, dim=-1))

        # Optional Bühlmann k from within/between dispersion.
        if self.shrink_k_mode == 'buhlmann':
            valid = ((self.n_eff >= 5) & (self.within_count > 0)).nonzero(as_tuple=False).flatten()
            if valid.numel() >= 4:
                within = (self.within_sum[valid] / self.within_count[valid].clamp_min(self.eps))
                between = 1.0 - (self.raw_proto[valid] * self.prior.unsqueeze(0)).sum(dim=-1)
                between = between.mean(dim=-1)
                between_adj = between - (within / self.n_eff[valid].clamp_min(1.0))
                k = float((within.mean() / between_adj.clamp_min(1e-4)).item())
                if np.isfinite(k):
                    self.shrink_k.fill_(float(np.clip(k, 1.0, 100.0)))

        # Shrunk prototypes: conservative, for selection only.
        z = self.n_eff / (self.n_eff + self.shrink_k.clamp_min(1e-3))
        self.Z.copy_(z)
        shrunk = _normalize(
            z[:, None, None] * self.raw_proto + (1.0 - z[:, None, None]) * self.prior.unsqueeze(0),
            dim=-1,
        )
        self.shrunk_proto.copy_(shrunk)

        # Per-class non-member mean / std.  Classes with too little evidence
        # fall back to the identity transform (nu=0, sigma=1).
        count = self.stat_count.clamp_min(self.eps)
        nu = self.stat_sum / count
        var = (self.stat_sq / count) - nu ** 2
        sigma = torch.sqrt(var.clamp_min(self.eps))
        enough = self.stat_count >= self.min_stat_count
        self.nu.copy_(torch.where(enough, nu, torch.zeros_like(nu)))
        self.sigma.copy_(torch.where(enough, sigma.clamp_min(self.min_sigma),
                                     torch.ones_like(sigma)))

        # Prediction history / stability prior.
        for sid, votes in self._epoch_preds.items():
            hist = self._history.setdefault(sid, [])
            hist.append(votes)
            if len(hist) > self.history_epochs:
                del hist[:-self.history_epochs]
        bars = []
        for hist in self._history.values():
            if not hist:
                continue
            flat = [v for triple in hist for v in triple]
            if flat:
                counts = np.bincount(np.asarray(flat, dtype=np.int64), minlength=self.num_classes)
                bars.append(float(counts.max() / max(1, len(flat))))
        if bars:
            self.history_agreement_bar.fill_(float(np.clip(np.mean(bars), 0.1, 0.95)))

        info = {
            'Z_mean': float(self.Z.mean().item()),
            'Z_known_mean': (float(self.Z[: max(1, self.num_known)].mean().item())
                             if self.num_known > 0 else float(self.Z.mean().item())),
            'n_eff_mean': float(self.n_eff.mean().item()),
            'shrink_k': float(self.shrink_k.item()),
            'history_bar': float(self.history_agreement_bar.item()),
            'sigma_mean': float(self.sigma.mean().item()),
        }
        return info

    # -- stability -------------------------------------------------------
    def stability(self, uq_id: int, target: Optional[int] = None,
                  prior_strength: float = 3.0) -> float:
        """IMDB-smoothed agreement of one sample over its recent history."""
        sid = _sid('U', int(uq_id))
        hist = self._history.get(sid, [])
        flat = [v for triple in hist for v in triple]
        if not flat:
            return float(self.history_agreement_bar.item())
        v = len(flat)
        if target is None:
            counts = np.bincount(np.asarray(flat, dtype=np.int64), minlength=self.num_classes)
            a = int(counts.max())
        else:
            a = int(sum(1 for x in flat if int(x) == int(target)))
        bar = float(self.history_agreement_bar.item())
        return float((a + prior_strength * bar) / (v + prior_strength))

    def stability_for_ids(self, uq_ids: Sequence[int],
                          targets: Optional[Sequence[int]] = None) -> np.ndarray:
        if targets is None:
            return np.asarray([self.stability(u) for u in uq_ids], dtype=np.float64)
        return np.asarray([self.stability(u, int(t)) for u, t in zip(uq_ids, targets)],
                          dtype=np.float64)

    # -- scores / logits -------------------------------------------------
    def forward(
        self,
        r_norm: torch.Tensor,
        use_shrunk: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Part-only score, cosine tensor and uniform slot weights.

        The signature mirrors ``PartPrototypeBank`` so existing eval code can
        use the ledger as a drop-in ``part_bank`` reference.
        """
        p = self.shrunk_proto if use_shrunk else self.raw_proto
        s = torch.einsum('bmd,cmd->bcm', r_norm, p)
        g_part = s.mean(dim=-1)
        a = torch.ones(self.num_classes, self.num_slots,
                       device=r_norm.device, dtype=r_norm.dtype) / float(self.num_slots)
        return g_part, s, a

    def raw_scores(self, r_norm: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bmd,cmd->bc', r_norm, self.raw_proto)

    def shrunk_scores(self, r_norm: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bmd,cmd->bc', r_norm, self.shrunk_proto)

    def normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        return (scores - self.nu.unsqueeze(0)) / self.sigma.unsqueeze(0).clamp_min(self.min_sigma)

    def fused_logits(
        self,
        g_global: torch.Tensor,
        r_norm: torch.Tensor,
        lambda_part: float = 0.5,
        topk: int = 5,
        use_raw: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Non-no-op v2 fusion: g_global + lambda_i * class-normalised evidence.

        ``lambda_i`` is shared across all classes of a sample and is the minimum
        credibility among the global top-k classes (stop-grad).
        """
        scores = self.raw_scores(r_norm) if use_raw else self.shrunk_scores(r_norm)
        ell = self.normalize_scores(scores)
        k = min(max(1, int(topk)), int(g_global.shape[1]))
        top = g_global.detach().topk(k=k, dim=-1).indices
        z_min = self.Z.detach()[top].min(dim=-1).values
        lam = float(lambda_part) * z_min
        fused = g_global + lam.unsqueeze(-1) * ell
        return fused, ell, lam

    # -- identity core ---------------------------------------------------
    @torch.no_grad()
    def add_core_samples(self, labels: Sequence[int], features: torch.Tensor,
                         weights: Optional[Sequence[float]] = None) -> None:
        if features.numel() == 0:
            return
        f = features.to(device=self.core_sum.device, dtype=torch.float32)
        if f.dim() == 3:
            f = f.mean(dim=1)
        f = _normalize(f, dim=-1)
        if weights is None:
            w = torch.ones(f.shape[0], device=f.device)
        else:
            w = torch.as_tensor(weights, device=f.device, dtype=torch.float32)
        for i, c in enumerate(labels):
            c = int(c)
            if not (0 <= c < self.num_classes):
                continue
            wi = float(w[i].item())
            if wi <= 0:
                continue
            self.core_sum[c].add_(f[i] * wi)
            self.core_count[c].add_(wi)

    def core_feature(self, c: int) -> Optional[torch.Tensor]:
        c = int(c)
        if not (0 <= c < self.num_classes):
            return None
        n = float(self.core_count[c].item())
        if n <= 0:
            return None
        return _normalize(self.core_sum[c], dim=-1)


# ---------------------------------------------------------------------------
# Selection guards (pure helpers)
# ---------------------------------------------------------------------------

def class_model_record(success: int, n: int, **extra: Any) -> Dict[str, Any]:
    rec = {'success': int(success), 'n': int(n)}
    rec.update(extra)
    return rec


def build_known_records(
    holdout_preds: Sequence[int],
    holdout_labels: Sequence[int],
    pass_mask: Sequence[bool],
    pred_class: int,
) -> Tuple[int, int]:
    """Return ``(correct, n_calibrated)`` for one known class."""
    correct = 0
    n = 0
    for pred, true, passed in zip(holdout_preds, holdout_labels, pass_mask):
        if not passed or int(pred) != int(pred_class):
            continue
        n += 1
        if int(true) == int(pred_class):
            correct += 1
    return correct, n


def build_novel_record(
    preds: Sequence[int],
    pass_mask: Sequence[bool],
    dim: int,
) -> Tuple[int, int]:
    """Return ``(agree, n_members)`` for one novel head dimension."""
    n = 0
    agree = 0
    for pred, passed in zip(preds, pass_mask):
        if int(pred) != int(dim):
            continue
        n += 1
        if passed:
            agree += 1
    return agree, n


def split_merge_guard(
    features: np.ndarray,
    *,
    other_centroids: Optional[np.ndarray] = None,
    split_silhouette: float = 0.25,
    merge_cos: float = 0.95,
    min_cluster_size: int = 3,
) -> Dict[str, Any]:
    """Detect a likely split (one head dim = two classes) or merge.

    ``features`` is ``(n, d)`` (slot-averaged local features).  The split test
    runs 2-means + silhouette on the candidate group.  The merge test compares
    the group centroid with centroids of other accepted novel groups.
    """
    f = np.asarray(features, dtype=np.float64)
    if f.ndim != 2 or f.shape[0] < max(4, 2 * min_cluster_size):
        return {'split': False, 'merge': False, 'reason': 'small'}
    # Normalise for cosine geometry.
    norm = np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
    fn = f / norm
    centroid = fn.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    merge = False
    merge_cos_value = -1.0
    if other_centroids is not None and len(other_centroids) > 0:
        oc = np.asarray(other_centroids, dtype=np.float64)
        oc = oc / (np.linalg.norm(oc, axis=1, keepdims=True) + 1e-12)
        cos = oc @ centroid
        merge_cos_value = float(np.max(cos))
        merge = bool(merge_cos_value >= float(merge_cos))

    split = False
    silhouette = -1.0
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        km = KMeans(n_clusters=2, n_init=3, random_state=0).fit(fn)
        labels = km.labels_
        sizes = np.bincount(labels, minlength=2)
        if int(sizes.min()) >= int(min_cluster_size):
            silhouette = float(silhouette_score(fn, labels))
            split = bool(silhouette >= float(split_silhouette))
    except Exception:
        # If sklearn is unavailable, use a simple cohesion test instead.
        cos_to_centroid = fn @ centroid
        split = bool(float(np.mean(1.0 - cos_to_centroid)) >= 0.35)

    return {
        'split': split,
        'merge': merge,
        'reason': 'split' if split else ('merge' if merge else 'ok'),
        'silhouette': silhouette,
        'merge_cos': merge_cos_value,
        'centroid': centroid,
    }


def novel_purity_guard(
    leak_count: int,
    group_count: int,
    max_leak_rate: float = 0.05,
) -> Dict[str, Any]:
    """Reject a novel dimension with too many known holdout samples in it."""
    group_count = max(int(group_count), 0)
    leak_count = max(int(leak_count), 0)
    rate = float(leak_count / group_count) if group_count > 0 else 0.0
    return {
        'leak_count': leak_count,
        'group_count': group_count,
        'leak_rate': rate,
        'pass': bool(rate <= float(max_leak_rate)),
    }


def select_by_budget(
    candidates: Sequence[Dict[str, Any]],
    budget: int,
    *,
    used_uq: Optional[set] = None,
    exclude_uq: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Take the highest-confidence candidates within a class/dimension budget."""
    used_uq = used_uq or set()
    exclude_uq = exclude_uq or set()
    ordered = sorted(candidates, key=lambda x: float(x.get('confidence', 0.0)), reverse=True)
    out: List[Dict[str, Any]] = []
    for item in ordered:
        if len(out) >= max(0, int(budget)):
            break
        uq = int(item.get('uq_idx', -1))
        if uq in used_uq or uq in exclude_uq:
            continue
        out.append(item)
    return out


def inverse_credibility_budget(
    z: float,
    *,
    budget_min: int = 20,
    budget_max: int = 500,
) -> int:
    z = float(np.clip(z, 0.0, 1.0))
    return int(round(float(budget_min) + (float(budget_max) - float(budget_min)) * (1.0 - z)))


def selection_schedule(
    epoch: int,
    epochs: int,
    *,
    start: float = 0.25,
    end: float = 0.75,
    every: float = 0.05,
    last_bucket: Optional[int] = None,
) -> Tuple[bool, int]:
    """Progress-based selection schedule.

    Returns ``(should_run, bucket)``.  ``last_bucket`` lets the caller suppress
    repeated runs inside one progress bucket.
    """
    epochs = max(1, int(epochs))
    progress = float(epoch) / float(max(1, epochs - 1))
    if progress < float(start) or progress > float(end):
        return False, -1
    every = max(float(every), 1e-6)
    bucket = int(math.floor((progress - float(start)) / every + 1e-8))
    if last_bucket is not None and bucket == int(last_bucket):
        return False, bucket
    return True, bucket


__all__ = [
    'EvidenceLedger',
    'beta_lcb',
    'beta_posterior',
    'beta_upper_tail_pvalue',
    'benjamini_hochberg',
    'gate_records',
    'fit_beta_prior',
    'make_holdout_split',
    'dataset_labels',
    'dataset_uq_ids',
    'set_dataset_transform',
    'build_known_records',
    'build_novel_record',
    'split_merge_guard',
    'novel_purity_guard',
    'select_by_budget',
    'inverse_credibility_budget',
    'selection_schedule',
]
