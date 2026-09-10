"""Post-hoc pseudo-labeling evidence figures (viec 5.2) — no training needed.

Reads an experiment dir containing (new runs) pseudo_events.json +
test_history.json, or (old runs) log.txt as fallback, and writes:
  1. stacked_contamination.png — per-iter stacked bars of audit outcomes
     (D1: correct / wrong-old / swallowed; D2: novel-correct / leak / confused)
  2. novel_tax.png — cumulative pseudo injected vs new-acc (and all-acc) by epoch
  3. reliability.png — predicted-conf vs empirical-correct, known vs novel doors
     (needs conf lists: only in new pseudo_events.json)
  4. funnel.png — novel gate funnel per iter (needs novel_gate: only in new JSON)

Usage:
    python tools/plot_pseudo_evidence.py <exp_dir> [--out DIR] [--which all|stacked|tax|reliability|funnel]

<exp_dir> may be an exp root (final_report.csv / log.txt / *.json alongside)
or a dev_outputs-style dir. Never crashes: missing inputs -> figure skipped
with a warning.
"""

import argparse
import json
import os
import re
import sys


def _load_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _parse_log_test_history(log_path):
    """Fallback: test trajectory from 'Test Accuracies' lines."""
    hist = []
    try:
        with open(log_path, encoding='utf-8', errors='replace') as f:
            text = f.read()
    except Exception:
        return hist
    rx = re.compile(
        r"Epoch\s+(\d+),\s+\S+\s+ACC_v2:\s+All\s+([\d.\-]+)\s+\|\s+Old\s+([\d.\-]+)"
        r"\s+\|\s+New\s+([\d.\-]+)")
    for m in rx.finditer(text):
        try:
            hist.append({'epoch': int(m.group(1)), 'all': float(m.group(2)),
                         'old': float(m.group(3)), 'new': float(m.group(4))})
        except ValueError:
            continue
    # pseudo totals over time (for the tax x-axis when JSON missing)
    totals = []
    for m in re.finditer(r"Total pseudo samples added:\s+(\d+)", text):
        totals.append(int(m.group(1)))
    return hist, totals


def _parse_log_audits(log_path):
    """Fallback: per-iter audit outcomes from legacy/new audit blocks."""
    iters = []
    try:
        with open(log_path, encoding='utf-8', errors='replace') as f:
            text = f.read()
    except Exception:
        return iters
    blocks = re.split(r"\[PSEUDO AUDIT iter (\d+)\]", text)[1:]
    for i in range(0, len(blocks) - 1, 2):
        it_no, body = int(blocks[i]), blocks[i + 1]
        # new 2-door format first
        d1 = re.search(r"\[D1 known door\] n=(\d+)", body)
        d2 = re.search(r"\[D2 novel door\] n=(\d+)", body)
        rec = {'iteration': it_no}
        if d1 or d2:
            rec['n_selected_d1'] = int(d1.group(1)) if d1 else 0
            rec['n_selected_d2'] = int(d2.group(1)) if d2 else 0

            def _grab(pat):
                m = re.search(pat, body)
                return int(m.group(1)) if m else 0

            rec['n_true_correct'] = _grab(r"Correct \(old==old\):\s+(\d+)/")
            rec['n_wrong_old'] = _grab(r"WRONG OLD class:\s+(\d+)/")
            rec['n_novel_contamination'] = _grab(r"NOVEL class swallowed:\s+(\d+)/")
            rec['n_novel_correct'] = _grab(r"Correct \(novel==novel\):\s+(\d+)/")
            rec['n_known_leakage'] = _grab(r"KNOWN leaked as novel:\s+(\d+)/")
            rec['n_novel_confused'] = _grab(r"NOVEL-vs-NOVEL confused:\s+(\d+)/")
        else:
            # legacy format
            def _grab(pat):
                m = re.search(pat, body)
                return int(m.group(1)) if m else 0

            c = _grab(r"Correct \(old==old\):\s+(\d+)/")
            w = _grab(r"WRONG OLD class:\s+(\d+)/")
            s = _grab(r"NOVEL class swallowed:\s+(\d+)/")
            rec.update({'n_selected_d1': c + w + s, 'n_selected_d2': 0,
                        'n_true_correct': c, 'n_wrong_old': w,
                        'n_novel_contamination': s, 'n_novel_correct': 0,
                        'n_known_leakage': 0, 'n_novel_confused': 0})
        iters.append(rec)
    return iters


def _resolve_inputs(exp_dir):
    """Return (events, history, pseudo_totals_fallback, source_tag)."""
    ev = _load_json(os.path.join(exp_dir, 'pseudo_events.json'))
    hist = _load_json(os.path.join(exp_dir, 'test_history.json'))
    if ev or hist:
        return ev, hist, [], 'json'
    log = os.path.join(exp_dir, 'log.txt')
    if os.path.isfile(log):
        ev2 = _parse_log_audits(log)
        hist2, totals = _parse_log_test_history(log)
        return ev2, hist2, totals, 'log-fallback'
    return [], [], [], 'none'


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_stacked(events, out_path):
    """Stacked bars per pseudo iteration: D1 correct/wrong/swallowed + D2."""
    plt = _mpl()
    import numpy as np
    iters = [e.get('iteration', i + 1) for i, e in enumerate(events)]
    g = lambda k: np.array([int(e.get(k, 0) or 0) for e in events], dtype=float)
    d1_ok, d1_wo, d1_sw = g('n_true_correct'), g('n_wrong_old'), g('n_novel_contamination')
    d2_ok, d2_lk, d2_cf = g('n_novel_correct'), g('n_known_leakage'), g('n_novel_confused')
    x = np.arange(len(events))
    fig, ax = plt.subplots(figsize=(max(8, len(events) * 1.2), 5))
    ax.bar(x, d1_ok, label='D1 correct (old==old)', color='#2e7d32')
    ax.bar(x, d1_wo, bottom=d1_ok, label='D1 wrong old', color='#f9a825')
    ax.bar(x, d1_sw, bottom=d1_ok + d1_wo, label='D1 novel swallowed', color='#c62828')
    b = d1_ok + d1_wo + d1_sw
    ax.bar(x, d2_ok, bottom=b, label='D2 novel correct', color='#1565c0')
    ax.bar(x, d2_lk, bottom=b + d2_ok, label='D2 known leaked', color='#ef6c00')
    ax.bar(x, d2_cf, bottom=b + d2_ok + d2_lk, label='D2 novel confused', color='#6a1b9a')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in iters])
    ax.set_xlabel('Pseudo iteration')
    ax.set_ylabel('Samples selected')
    ax.set_title('Pseudo-label audit per iteration (D1 known door + D2 novel door)')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_novel_tax(events, history, pseudo_totals_fallback, out_path):
    """Twin axes: cumulative pseudo injected vs new-acc (and all-acc) by epoch."""
    plt = _mpl()
    import numpy as np
    # cumulative injected per iteration (+ epoch if known)
    cum, iters, ep_of_iter = [], [], []
    run = 0
    for i, e in enumerate(events):
        run += int(e.get('n_selected', e.get('n_selected_gt', 0)) or 0)
        cum.append(run)
        iters.append(e.get('iteration', i + 1))
        ep_of_iter.append(e.get('epoch', None))
    fig, ax1 = plt.subplots(figsize=(10, 5))
    if cum:
        ax1.step(iters, cum, where='mid', color='#c62828', linewidth=2,
                 label='cumulative pseudo injected')
        ax1.set_xlabel('Pseudo iteration')
        ax1.set_ylabel('Cumulative pseudo samples', color='#c62828')
    ax1.tick_params(axis='y', labelcolor='#c62828')
    ax2 = ax1.twinx()
    drew = False
    if history:
        ep = [h.get('epoch') for h in history
              if h.get('epoch') is not None and h.get('new', -1) >= 0]
        new = [h.get('new') for h in history
               if h.get('epoch') is not None and h.get('new', -1) >= 0]
        allv = [h.get('all') for h in history
                if h.get('epoch') is not None and h.get('new', -1) >= 0]
        if ep:
            ax2.plot(ep, new, 'o-', color='#1565c0', markersize=3, label='new-acc')
            ax2.plot(ep, allv, 's-', color='gray', markersize=2, alpha=0.6, label='all-acc')
            drew = True
    # mark injection epochs on the acc axis
    for it, epv in zip(iters, ep_of_iter):
        if epv is not None and drew:
            ax2.axvline(epv, color='#c62828', alpha=0.25, linestyle='--')
    ax2.set_ylabel('Test accuracy', color='#1565c0')
    ax2.tick_params(axis='y', labelcolor='#1565c0')
    ax2.set_xlabel('Epoch (dashed red = pseudo injection)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')
    ax2.set_title('Novel-tax view: pseudo volume vs new/all accuracy')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _reliability_points(confs, labels, n_bins=5):
    """Bin predicted-confs -> (bin_center, empirical_correct, count)."""
    import numpy as np
    confs = np.asarray(confs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    edges = np.linspace(0, 1, n_bins + 1)
    pts = []
    for b in range(n_bins):
        m = (confs >= edges[b]) & (confs <= edges[b + 1] if b == n_bins - 1
                                   else confs < edges[b + 1])
        if m.sum() == 0:
            continue
        pts.append(((edges[b] + edges[b + 1]) / 2, labels[m].mean(), int(m.sum())))
    return pts


def plot_reliability(events, out_path):
    """Known vs novel reliability: predicted conf vs empirical correct.

    Needs conf lists (new pseudo_events.json only). Returns None when absent.
    """
    plt = _mpl()
    ck = [c for e in events for c in (e.get('conf_correct') or [])]
    wk = ([c for e in events for c in (e.get('conf_wrong_old') or [])]
          + [c for e in events for c in (e.get('conf_swallowed') or [])])
    cn = [c for e in events for c in (e.get('conf_novel_correct') or [])]
    wn = ([c for e in events for c in (e.get('conf_leak') or [])]
          + [c for e in events for c in (e.get('conf_confused') or [])])
    if not ck + wk and not cn + wn:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, ok_c, bad_c, title in (
            (axes[0], ck, wk, 'D1 known door'),
            (axes[1], cn, wn, 'D2 novel door')):
        confs = list(ok_c) + list(bad_c)
        labels = [1] * len(ok_c) + [0] * len(bad_c)
        if not confs:
            ax.set_title(title + ' (no data)')
            continue
        pts = _reliability_points(confs, labels)
        if pts:
            xs, ys, ns = zip(*pts)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='perfect calibration')
            ax.plot(xs, ys, 'o-', label='empirical')
            for x, y, n in pts:
                ax.annotate(str(n), (x, y), fontsize=7, ha='center', va='bottom')
        ax.set_xlabel('Predicted confidence (bin center)')
        ax.set_title(title + f' (n={len(confs)})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Fraction correct')
    fig.suptitle('Reliability: does confidence mean correctness? (n per bin annotated)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_funnel(events, out_path):
    """Novel gate funnel per iter: leftover -> kept + drop reasons.

    Needs novel_gate summaries (new pseudo_events.json only). Returns None when absent.
    """
    plt = _mpl()
    import numpy as np
    rows = [(e.get('iteration', i + 1), e.get('novel_gate') or {})
            for i, e in enumerate(events)]
    rows = [(it, g) for it, g in rows if isinstance(g, dict) and g.get('status') == 'kept']
    if not rows:
        return None
    # union of drop reasons across iters for stable colors
    reasons = []
    for _, g in rows:
        for r in (g.get('reason_hist') or {}):
            if r != 'keep' and r not in reasons:
                reasons.append(r)
    iters = [it for it, _ in rows]
    kept = np.array([int(g.get('n_selected', 0) or 0) for _, g in rows], dtype=float)
    lo = np.array([int(g.get('n_leftover', 0) or 0) for _, g in rows], dtype=float)
    x = np.arange(len(rows))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(rows) * 1.4), 8),
                                   sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    ax1.bar(x, lo, color='lightgray', label='leftover clusters')
    ax1.bar(x, kept, color='#1565c0', label='clusters kept')
    ax1.set_ylabel('# clusters')
    ax1.legend(fontsize=8)
    ax1.set_title('Novel gate funnel: leftover clusters -> kept + drop reasons')
    bottom = np.zeros(len(rows))
    cmap = plt.get_cmap('tab10')
    for i, r in enumerate(reasons):
        v = np.array([int(g.get('reason_hist', {}).get(r, 0) or 0) for _, g in rows],
                     dtype=float)
        ax2.bar(x, v, bottom=bottom, color=cmap(i % 10), label=r)
        bottom = bottom + v
    ax2.bar(x, kept, bottom=bottom, color='#1565c0', label='keep')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i) for i in iters])
    ax2.set_xlabel('Pseudo iteration (novel iters only)')
    ax2.set_ylabel('# clusters')
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp_dir')
    ap.add_argument('--out', default=None)
    ap.add_argument('--which', default='all',
                    choices=['all', 'stacked', 'tax', 'reliability', 'funnel'])
    a = ap.parse_args()
    out = a.out or a.exp_dir
    os.makedirs(out, exist_ok=True)
    events, history, totals, src = _resolve_inputs(a.exp_dir)
    print(f'[plot] source={src} iters={len(events)} test_rounds={len(history)}')
    if not events and not history:
        print('[plot] nothing to draw (need pseudo_events.json/test_history.json or log.txt).')
        return 1
    jobs = {'stacked': lambda: plot_stacked(
                events, os.path.join(out, 'stacked_contamination.png')) if events else None,
            'tax': lambda: plot_novel_tax(
                events, history, totals, os.path.join(out, 'novel_tax.png')),
            'reliability': lambda: plot_reliability(
                events, os.path.join(out, 'reliability.png')),
            'funnel': lambda: plot_funnel(
                events, os.path.join(out, 'gate_funnel.png'))}
    want = list(jobs) if a.which == 'all' else [a.which]
    for w in want:
        try:
            p = jobs[w]()
            print(f'[plot] {w}: {p if p else "skipped (no input data)"}')
        except Exception as e:
            print(f'[plot] {w} FAILED (non-fatal): {e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
