"""
Visualization utilities for GCD experiments.

Generates PCA / t-SNE scatter plots and confusion matrices from the
features, ground-truth labels and K-Means cluster predictions that are
already computed inside ``model.bacon.test``.

All functions are defensive: visualization must never crash training,
so every failure is logged and swallowed.
"""

import os

import numpy as np

import matplotlib
matplotlib.use('Agg')  # headless backend (no display on Modal)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vis_dir(args, subdir):
    """Resolve <exp_root>/visualizations/<subdir> from args.model_dir."""
    vis_root = os.path.abspath(
        os.path.join(args.model_dir, os.pardir, 'visualizations'))
    path = os.path.join(vis_root, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def _cmap(n):
    """Distinct color list for n classes (tab20 cycles past 20)."""
    base = plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors
    return [base[i % len(base)] for i in range(max(n, 1))]


def _save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scatter plots (PCA / t-SNE)
# ---------------------------------------------------------------------------

def plot_embedding_scatter(embedding, targets, mask, title, save_path,
                           max_points=5000, seed=0):
    """
    Scatter plot of a 2-D embedding, colored by ground-truth class.

    Old (labeled) classes get solid markers, new (unlabeled) ones open
    markers. Points are subsampled for speed / legibility.

    :param embedding: np.array [n_samples, 2]
    :param targets: np.array [n_samples] ground-truth labels
    :param mask: bool np.array [n_samples] True = old/labeled class
    """
    rng = np.random.RandomState(seed)
    n = embedding.shape[0]
    idx = np.arange(n)
    if n > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)

    emb = embedding[idx]
    tgt = targets[idx]
    msk = mask[idx]

    classes = np.unique(tgt)
    colors = _cmap(len(classes))
    # Map arbitrary class ids -> palette positions so old/new share style split
    class_to_color = {c: colors[i % len(colors)] for i, c in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for c in classes:
        sel = tgt == c
        is_old = msk[sel][0] if sel.any() else True
        ax.scatter(emb[sel, 0], emb[sel, 1],
                   s=8 if len(classes) > 30 else 18,
                   color=class_to_color[c],
                   alpha=0.6,
                   marker='o' if is_old else '^',
                   linewidths=0,
                   label=str(c))

    n_old = int(msk.sum())
    n_new = int((~msk).sum())
    ax.set_title(f'{title}\nold/labeled: {n_old} pts | new/unlabeled: {n_new} pts '
                 f'(circles = old, triangles = new)')
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend only when class count is small enough to stay readable
    if len(classes) <= 20:
        ax.legend(markerscale=2, fontsize=7, loc='best', ncol=2)

    _save_fig(fig, save_path)


def run_pca(features, targets, mask, epoch, args, save_name='Test ACC',
            max_points=5000, seed=0):
    """Compute 2-component PCA on features and save the scatter plot."""
    try:
        from sklearn.decomposition import PCA

        emb = PCA(n_components=2, random_state=seed).fit_transform(features)
        path = os.path.join(_vis_dir(args, 'pca'),
                            f'pca_epoch{epoch}.png')
        plot_embedding_scatter(emb, targets, mask,
                               f'PCA — {save_name} (epoch {epoch})',
                               path, max_points=max_points, seed=seed)
        args.logger.info(f'[VISUALIZATION] PCA saved to {path}')
        return path
    except Exception as e:
        args.logger.warning(f'[VISUALIZATION] PCA failed: {e}')
        return None


def run_tsne(features, targets, mask, epoch, args, save_name='Test ACC',
             max_points=3000, perplexity=30, seed=0):
    """
    Compute t-SNE (on a PCA-50 pre-reduction, standard practice) and save
    the scatter plot. Subsamples to ``max_points`` first — t-SNE is slow.
    """
    try:
        from sklearn.decomposition import PCA as _PCA
        from sklearn.manifold import TSNE

        rng = np.random.RandomState(seed)
        n = features.shape[0]
        feats, tgt, msk = features, targets, mask
        if n > max_points:
            idx = rng.choice(n, size=max_points, replace=False)
            feats, tgt, msk = features[idx], targets[idx], mask[idx]

        pca_dims = min(50, feats.shape[1])
        feats_pca = _PCA(n_components=pca_dims,
                         random_state=seed).fit_transform(feats)

        perpl = float(min(perplexity, max(5.0, (len(feats) - 1) / 3.0)))
        emb = TSNE(n_components=2,
                   init='pca',
                   learning_rate='auto',
                   perplexity=perpl,
                   random_state=seed).fit_transform(feats_pca)

        path = os.path.join(_vis_dir(args, 'tsne'),
                            f'tsne_epoch{epoch}.png')
        plot_embedding_scatter(emb, tgt, msk,
                               f't-SNE — {save_name} (epoch {epoch})',
                               path, max_points=len(feats), seed=seed)
        args.logger.info(f'[VISUALIZATION] t-SNE saved to {path}')
        return path
    except Exception as e:
        args.logger.warning(f'[VISUALIZATION] t-SNE failed: {e}')
        return None


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def build_confusion(y_true, y_pred, ind_map=None):
    """
    Row-normalized confusion matrix between ground truth and predictions.

    If ``ind_map`` ({gt_class: cluster_id} from Hungarian matching) is given,
    cluster ids are mapped back to gt-class space first, so diagonal ≈ correct.

    :return: (matrix [n_gt, n_pred], used_map or None)
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    if ind_map is not None:
        mapper = np.vectorize(lambda c: ind_map.get(c, -1))
        y_pred_mapped = mapper(y_pred)
    else:
        y_pred_mapped = y_pred

    gt_classes = sorted(set(y_true.tolist()))
    pred_classes = sorted(set(y_pred_mapped.tolist()))
    gt_idx = {c: i for i, c in enumerate(gt_classes)}
    pred_idx = {c: i for i, c in enumerate(pred_classes)}

    mat = np.zeros((len(gt_classes), len(pred_classes)), dtype=np.float64)
    for t, p in zip(y_true, y_pred_mapped):
        if p == -1:
            continue
        mat[gt_idx[t], pred_idx[p]] += 1

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat_norm = mat / row_sums

    labels_true = [f'GT {c}' for c in gt_classes]
    labels_pred = [f'P {c}' for c in pred_classes]
    return mat_norm, (labels_true, labels_pred)


def run_confusion_matrix(y_true, y_pred, epoch, args, ind_map=None,
                         save_name='Test ACC', annotate_max=25):
    """Save row-normalized confusion matrix heatmap."""
    try:
        mat, (labels_true, labels_pred) = build_confusion(y_true, y_pred,
                                                          ind_map=ind_map)

        fig_w = max(8, min(24, 0.45 * mat.shape[1]))
        fig_h = max(6, min(22, 0.35 * mat.shape[0]))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        cmap = LinearSegmentedColormap.from_list(
            'blue_scale', ['#f7fbff', '#08306b'])
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect='auto')

        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        ax.set_xticklabels(labels_pred, rotation=90, fontsize=7)
        ax.set_yticklabels(labels_true, fontsize=7)
        ax.set_xlabel('Predicted cluster')
        ax.set_ylabel('Ground-truth class')
        ax.set_title(f'Confusion Matrix — {save_name} (epoch {epoch})')

        # Annotate cells only for small matrices
        if mat.shape[0] <= annotate_max and mat.shape[1] <= annotate_max:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat[i, j]
                    if v > 0.005:
                        color = 'white' if v > 0.5 else 'black'
                        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                                fontsize=6, color=color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Row recall')
        plt.tight_layout()

        path = os.path.join(_vis_dir(args, 'confusion_matrix'),
                            f'confusion_epoch{epoch}.png')
        _save_fig(fig, path)
        args.logger.info(f'[VISUALIZATION] Confusion matrix saved to {path}')
        return path
    except Exception as e:
        args.logger.warning(f'[VISUALIZATION] Confusion matrix failed: {e}')
        return None


# ---------------------------------------------------------------------------
# Entry point used by model.bacon.test()
# ---------------------------------------------------------------------------

def generate_eval_visualizations(features, targets, mask, preds, ind_map,
                                 epoch, args, save_name='Test ACC'):
    """
    One call generates all three visualizations for a test round.

    Called from ``test()`` where features/targets/preds already exist, so no
    extra forward passes are needed.
    """
    results = {}
    results['pca'] = run_pca(features, targets, mask.astype(bool), epoch,
                             args, save_name)
    results['tsne'] = run_tsne(features, targets, mask.astype(bool), epoch,
                               args, save_name)
    results['confusion'] = run_confusion_matrix(targets, preds, epoch, args,
                                                ind_map=ind_map,
                                                save_name=save_name)

    done = [k for k, v in results.items() if v]
    args.logger.info(f'[VISUALIZATION] Generated: {done or "none"}')
    return results


# ---------------------------------------------------------------------------
# Final evaluation report (after training finishes)
# ---------------------------------------------------------------------------

def _fmt(v):
    """Format accuracy value; -1 means undefined (empty subset)."""
    return f'{v:.1f}' if v is not None and v >= 0 else 'n/a'


def save_final_report(accs, args, extra_info=None):
    """
    Write the final Old/New/All + Many/Med/Few table to
    <exp_root>/final_report.txt (human-readable) and .csv (machine-readable).

    :param accs: dict with keys
        'all', 'old', 'new'                       — headline accuracies
        'k_many', 'k_med', 'k_few', 'k_std'       — known-class subsets
        'u_many', 'u_med', 'u_few', 'u_std'       — novel-class subsets
    :param extra_info: optional dict of scalar metadata (dataset, epochs, ...)
    :return: path to the txt report
    """
    try:
        report_path = os.path.abspath(
            os.path.join(args.model_dir, os.pardir, 'final_report.txt'))
        csv_path = os.path.abspath(
            os.path.join(args.model_dir, os.pardir, 'final_report.csv'))

        std_k = (accs.get('k_std') if accs.get('k_std') is not None
                 else np.std([accs['k_many'], accs['k_med'], accs['k_few']]))
        std_u = (accs.get('u_std') if accs.get('u_std') is not None
                 else np.std([accs['u_many'], accs['u_med'], accs['u_few']]))

        sep = '-' * 100
        lines = []
        lines.append('=' * 100)
        lines.append('FINAL EVALUATION REPORT (best checkpoint, test set)')
        lines.append('=' * 100)
        if extra_info:
            for k, v in extra_info.items():
                lines.append(f'{k}: {v}')
        lines.append(sep)
        lines.append('Old / New / All')
        lines.append(f'  Old: {_fmt(accs["old"])}   New: {_fmt(accs["new"])}   All: {_fmt(accs["all"])}')
        lines.append(sep)
        # Header matching the standard GCD paper table
        lines.append('Known                       | Novel')
        lines.append('Many(up)  Med(up)  Few(up)  Std(dn)| Many(up)  Med(up)  Few(up)  Std(dn)')
        lines.append(sep)
        lines.append(f'{_fmt(accs["k_many"]):<9} {_fmt(accs["k_med"]):<8} '
                     f'{_fmt(accs["k_few"]):<8} {_fmt(std_k):<5} | '
                     f'{_fmt(accs["u_many"]):<9} {_fmt(accs["u_med"]):<8} '
                     f'{_fmt(accs["u_few"]):<8} {_fmt(std_u):<5}')
        lines.append(sep)

        text = '\n'.join(lines)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(text + '\n')

        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            import csv as _csv
            w = _csv.writer(f)
            w.writerow(['metric', 'value'])
            if extra_info:
                for k, v in extra_info.items():
                    w.writerow([k, v])
            for key in ['old', 'new', 'all',
                        'k_many', 'k_med', 'k_few', 'k_std',
                        'u_many', 'u_med', 'u_few', 'u_std']:
                v = std_k if key == 'k_std' else (std_u if key == 'u_std'
                                                  else accs.get(key))
                w.writerow([key, f'{v:.2f}' if isinstance(v, (int, float)) else v])

        args.logger.info('\n' + text)
        args.logger.info(f'[FINAL REPORT] Saved to {report_path} and {csv_path}')
        return report_path
    except Exception as e:
        args.logger.warning(f'[FINAL REPORT] Failed to write report: {e}')
        return None
