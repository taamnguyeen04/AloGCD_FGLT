import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('./')
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from util.visualize_utils import generate_eval_visualizations
from config import exp_root
from model.loss import info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from copy import deepcopy
from sklearn.cluster import KMeans
# from cuml.cluster import KMeans as cuKMeans
from data.cifar import CustomCIFAR100, cifar_100_root
import random
from model.dist_est import dist_est
from model.reg_loss import compute_reg_loss
from model.softconloss import compute_softconloss
import os
import sys

class CE_Head(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = nn.functional.normalize(x, dim=-1, p=2)
        logits = self.last_layer(x)
        return logits


def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()




def train_dual(ce_backbone, ce_head, cl_backbone, cl_head, train_loader, test_loader, args):
    def set_model(train=False, eval=False):
        assert (train ^ eval)
        if train:
            student_ce.train()
            student_cl.train()
        elif eval:
            student_ce.eval()
            student_cl.eval()

    from util.cluster_and_log_utils import set_args_mmf
    set_args_mmf(args, train_loader)

    save_path = args.model_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    student_ce = nn.Sequential(ce_backbone, ce_head).to(device)
    student_cl = nn.Sequential(cl_backbone, cl_head).to(device)

    params_groups_cl = list(cl_head.parameters()) + list(cl_backbone.parameters())
    params_groups_ce = get_params_groups(student_ce)
    optimizer_ce = SGD(params_groups_ce, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler_ce = lr_scheduler.CosineAnnealingLR(
        optimizer_ce,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )

    optimizer_cl = SGD(params_groups_cl, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler_cl = lr_scheduler.CosineAnnealingLR(
        optimizer_cl,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )
    args.current_epoch = 0
    best_test_acc_all_cl = -1
    best_epoch = -1
    
    # Pseudo labeling state (Fine-Grained GCD)
    if args.enable_pseudo_labeling:
        pseudo_iteration = 0
        pseudo_samples_added = 0
        used_pseudo_uq_idxs = set()   # uq_idxs already promoted to labeled, avoid duplicates
        args.pseudo_events = []       # one entry per pseudo update (bias evidence)
        # Ground-truth maps used ONLY for diagnostics (contamination audit),
        # never fed back into training.
        args.uq2true = build_uq_to_true_label(train_loader.dataset.unlabelled_dataset)
        orig_labeled_counts = {}
        count_labeled_per_class(train_loader.dataset.labelled_dataset, orig_labeled_counts)
        labeled_class_counts = dict(orig_labeled_counts)
        pseudo_added_per_class = {}
        args.logger.info("\n[PSEUDO LABELING] Enabled - Mode {mode}".format(mode=args.pseudo_mode))
        args.logger.info("[PSEUDO LABELING] Original labeled distribution: {}".format(
            dict(sorted(orig_labeled_counts.items()))))
    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args=args,
    )

    set_model(eval=True)
    est_count = dist_est(ce_backbone, cl_backbone, ce_head, train_all_test_trans_loader, args)
    set_model(train=True)


    for epoch in range(args.epochs):
        args.current_epoch = epoch
        loss_record_ce = AverageMeter()
        loss_record_cl = AverageMeter()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            images_, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images_, dim=0).cuda(non_blocking=True)

            x = ce_backbone.prepare_tokens(images)

            for i, blk in enumerate(ce_backbone.blocks):
                if i < args.grad_from_block:
                    x = blk(x)   # get fixed feature

            for i, blk in enumerate(ce_backbone.blocks):
                if i >= args.grad_from_block:
                    ce_backbone_feature = blk(x)
            ce_backbone_feature = ce_backbone.norm(ce_backbone_feature)
            ce_backbone_feature = ce_backbone_feature[:, 0]
            student_out = ce_head(ce_backbone_feature)

            for i, blk in enumerate(cl_backbone.blocks):
                if i >= args.grad_from_block:
                    cl_backbone_feature = blk(x)
            cl_backbone_feature = cl_backbone.norm(cl_backbone_feature)
            cl_backbone_feature = cl_backbone_feature[:, 0]
            cl_proj_feature = cl_head(cl_backbone_feature)

            ####################### COMPUTE LOSS #######################
            pstr = ''
            teacher_out = student_out.detach()

            # clustering, unsup
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)

            # clustering, sup
            sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
            sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)

            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            me_max_loss = compute_reg_loss(student_out, est_count, args)

            pstr += f'cls_loss: {cls_loss.item():.2f} '
            pstr += f'cps_loss: {cluster_loss.item():.2f} '
            pstr += f'reg_loss: {me_max_loss.item():.2f} '
            cluster_loss += args.memax_weight * me_max_loss
            loss_ce = (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss

            loss_record_ce.update(loss_ce.item(), class_labels.size(0))
            optimizer_ce.zero_grad()
            loss_ce.backward()
            optimizer_ce.step()

            loss_cl = 0
            # for CL part
            # represent learning, unsup
            cl_proj_feature = torch.nn.functional.normalize(cl_proj_feature, dim=-1)

            contrastive_logits, contrastive_labels = info_nce_logits(features=cl_proj_feature)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # representation learning, sup
            sup_cl_proj_feature = torch.cat([f[mask_lab].unsqueeze(1) for f in cl_proj_feature.chunk(2)], dim=1)
            sup_con_labels = class_labels[mask_lab]

            if epoch >= args.ce_warmup:
                sup_con_loss = SupConLoss()(sup_cl_proj_feature, labels=sup_con_labels)
                soft_con_loss = compute_softconloss(student_out, cl_proj_feature, sup_con_labels, sup_cl_proj_feature, mask_lab, args)
                loss_cl += ((1 - args.sup_weight) * contrastive_loss + (args.sup_weight / 2) * sup_con_loss)
                loss_cl += (args.sup_weight / 2) * soft_con_loss
            else:
                sup_con_loss = SupConLoss()(sup_cl_proj_feature, labels=sup_con_labels)
                loss_cl += ((1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss)

            pstr += f'sup_con_loss: {sup_con_loss.item():.2f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.2f} '

            loss_record_cl.update(loss_cl.item(), class_labels.size(0))
            optimizer_cl.zero_grad()
            loss_cl.backward()
            optimizer_cl.step()

            if batch_idx % args.print_freq == 0:
                pbar.set_postfix({
                    'loss_ce': f"{loss_ce.item():.4f}", 
                    'loss_cl': f"{loss_cl.item():.4f}", 
                    'metrics': pstr.strip()
                })

        args.logger.info('Train Epoch: {} Avg Loss_ce: {:.2f} Avg Loss_cl: {:.2f}'.format(epoch, loss_record_ce.avg, loss_record_cl.avg))

        if (epoch+1) % args.est_freq == 0:
            set_model(eval=True)
            est_count = dist_est(ce_backbone, cl_backbone, ce_head, train_all_test_trans_loader, args)

        if epoch % args.test_freq == 0:
            args.logger.info('Testing on disjoint test set...')
            with torch.no_grad():
                all_acc_test_cl, old_acc_test_cl, new_acc_test_cl, acc_list_cl, cl_ind_map = test(
                    student_cl,
                    test_loader,
                    epoch=epoch,
                    save_name='Test ACC',
                    args=args,
                    train_loader=train_loader)
            
            args.logger.info(
                    'Test Accuracies CL: All {:.1f} | Old {:.1f} | New {:.1f}'.format(all_acc_test_cl,
                                                                                      old_acc_test_cl,
                                                                                      new_acc_test_cl))

            # Evaluate and log per-class accuracy
            student_cl.eval()
            per_class_stats, pred_dist = evaluate_per_class_accuracy(
                student_cl, test_loader, args.train_classes, args.num_classes
            )
            log_per_class_stats(per_class_stats, epoch, args)

            # Chart every 10 epochs so bias evolution is visible over time
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                save_per_class_accuracy_chart(per_class_stats, epoch, args)

            student_cl.train()
        
        # Pseudo Labeling Update Logic (Fine-Grained GCD)
        if args.enable_pseudo_labeling and args.pseudo_update_freq > 0:
            should_update = (epoch > 0) and (epoch % args.pseudo_update_freq == 0)
            
            if should_update and pseudo_iteration < args.max_pseudo_iterations:
                pseudo_iteration += 1
                args.logger.info("\n" + "="*60)
                args.logger.info(f"[PSEUDO LABELING] Iteration {pseudo_iteration}")
                args.logger.info("="*60)
                
                # Get unlabeled loader
                unlab_loader = DataLoader(
                    train_loader.dataset.unlabelled_dataset,
                    batch_size=256, shuffle=False, num_workers=0
                )
                
                if args.pseudo_mode == 1:
                    # Find best class
                    best_class = max(per_class_stats.keys(), 
                                   key=lambda c: per_class_stats[c]['acc'])
                    best_acc = per_class_stats[best_class]['acc']
                    
                    args.logger.info(f"[MODE 1] Best class: {best_class} (Acc: {best_acc:.3f})")
                    target_class = best_class
                else:
                    target_class = None
                    args.logger.info(f"[MODE 2] High confidence threshold across all classes: {args.confidence_threshold}")
                
                # Collect pseudo labels
                new_pseudo, newly_used = collect_pseudo_labels_from_unlabeled(
                    student_ce, unlab_loader,
                    mode=args.pseudo_mode,
                    target_class=target_class,
                    max_samples=args.max_samples_per_class,
                    threshold=getattr(args, 'confidence_threshold', 0.9),
                    used_uq_idxs=used_pseudo_uq_idxs,
                    top_ratio=getattr(args, 'pseudo_top_ratio', 0.8),
                    max_label=args.num_labeled_classes if args.pseudo_mode == 2 else None
                )
                used_pseudo_uq_idxs |= newly_used

                if args.pseudo_mode == 1:
                    args.logger.info(f"Collected {len(new_pseudo)} pseudo samples for class {target_class}")
                else:
                    args.logger.info(f"Collected {len(new_pseudo)} pseudo samples across classes (th>={getattr(args, 'confidence_threshold', 0.9)})")

                # ---------------------------------------------------------
                # GROUND-TRUTH AUDIT of the selected pseudo samples.
                # Diagnostics only — never touches training. Shows how many
                # promoted samples are actually NOVEL-class images (contam-
                # ination) or the WRONG old class (confirmation-bias fuel).
                # ---------------------------------------------------------
                audit = audit_pseudo_samples(new_pseudo, args.uq2true,
                                             num_labeled=args.num_labeled_classes)
                log_pseudo_audit(audit, pseudo_iteration, target_class, args)

                if new_pseudo:
                    # Update train loader
                    train_loader = update_train_loader(
                        train_loader, train_loader.dataset, new_pseudo
                    )
                    pseudo_samples_added += len(new_pseudo)
                    for s in new_pseudo:
                        pseudo_added_per_class[s['label']] = pseudo_added_per_class.get(s['label'], 0) + 1

                    # Labeled-set distribution after injection
                    count_pseudo_into(labeled_class_counts, new_pseudo)
                    log_labeled_distribution(orig_labeled_counts, labeled_class_counts,
                                             pseudo_added_per_class, args)

                    args.pseudo_events.append({
                        'iteration': pseudo_iteration,
                        'epoch': epoch,
                        'n_selected': len(new_pseudo),
                        **audit,   # n_true_correct / n_wrong_old / n_novel_contamination
                        'pseudo_samples_total': pseudo_samples_added,
                        **{f'labeled_c{c}': labeled_class_counts.get(c, 0)
                           for c in sorted(set(orig_labeled_counts) | set(labeled_class_counts))},
                    })
                    save_bias_evidence_chart(args.pseudo_events, pseudo_iteration, args)
                    args.logger.info(f"Total pseudo samples added: {pseudo_samples_added}")
                    args.logger.info("="*60 + "\n")
        
        # Step schedule
        exp_lr_scheduler_ce.step()
        exp_lr_scheduler_cl.step()

        if epoch % args.test_freq == 0 and all_acc_test_cl > best_test_acc_all_cl:

            best_test_acc_new_cl = new_acc_test_cl
            best_test_acc_old_cl = old_acc_test_cl
            best_test_acc_all_cl = all_acc_test_cl
            best_epoch = epoch

            save_dict_cl = {
                'ce_backbone': ce_backbone.state_dict(),
                'ce_head': ce_head.state_dict(),
                'cl_backbone': cl_backbone.state_dict(),
                'cl_head': cl_head.state_dict(),
            }

            torch.save(save_dict_cl, save_path + f'/model_epoch{epoch}.pt')
            args.logger.info("model saved to {}.".format(save_path))

        if epoch >= args.stop_epoch:
            break

    args.logger.info(
        f'Metrics with best model on test set: All: {best_test_acc_all_cl:.1f} Old: {best_test_acc_old_cl:.1f} New: {best_test_acc_new_cl:.1f} ')

    # ----------------------
    # FINAL EVALUATION with best checkpoint → final_report.txt / .csv
    # ----------------------
    try:
        ckpt_path = os.path.join(save_path, f'model_epoch{best_epoch}.pt')
        if best_epoch >= 0 and os.path.exists(ckpt_path):
            args.logger.info(f'\n[FINAL EVAL] Reloading best checkpoint (epoch {best_epoch})...')
            ckpt = torch.load(ckpt_path, map_location='cuda')
            ce_backbone.load_state_dict(ckpt['ce_backbone'])
            ce_head.load_state_dict(ckpt['ce_head'])
            cl_backbone.load_state_dict(ckpt['cl_backbone'])
            cl_head.load_state_dict(ckpt['cl_head'])
            del ckpt

            student_cl = nn.Sequential(cl_backbone, cl_head)
            set_model(eval=True)
            # Always visualize the final best model regardless of vis_freq phase
            saved_vis_freq = args.vis_freq
            args.vis_freq = 1
            all_acc, old_acc, new_acc, acc_list, ind_map = test(
                student_cl, test_loader, epoch=best_epoch,
                save_name='Final ACC', args=args, train_loader=train_loader)
            args.vis_freq = saved_vis_freq

            accs = {
                'all': all_acc, 'old': old_acc, 'new': new_acc,
                'k_many': acc_list[0], 'k_med': acc_list[1], 'k_few': acc_list[2],
                'k_std': np.array(acc_list[:3]).std(),
                'u_many': acc_list[3], 'u_med': acc_list[4], 'u_few': acc_list[5],
                'u_std': np.array(acc_list[3:6]).std(),
            }
            from util.visualize_utils import save_final_report
            save_final_report(accs, args, extra_info={
                'dataset': args.dataset_name,
                'imb_ratio': args.imb_ratio,
                'epochs': args.epochs,
                'best_epoch': best_epoch,
                'pseudo_labeling': args.enable_pseudo_labeling,
                'pseudo_mode': getattr(args, 'pseudo_mode', 0),
            })
        else:
            args.logger.warning('[FINAL EVAL] No best checkpoint found — skipping final report.')
    except Exception as e:
        args.logger.warning(f'[FINAL EVAL] Final evaluation failed: {e}')


def test(model, test_loader, epoch, save_name, args, train_loader):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])
    print('Collating features...')
    # First extract all features
    for batch_idx, batch in enumerate(test_loader):
        batch = batch[:3]
        (images, label, _) = batch
        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model[0](images)  # follow GCD: clustering on normalized backbone feature

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.detach().cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    all_acc, old_acc, new_acc, acc_list, ind_map = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args, train_loader=train_loader)

    # PCA / t-SNE / confusion matrix — reuse features & preds already computed
    vis_freq = getattr(args, 'vis_freq', 1)
    if vis_freq > 0 and epoch % vis_freq == 0:
        generate_eval_visualizations(
            features=all_feats,
            targets=targets,
            mask=mask,
            preds=preds,
            ind_map=ind_map,
            epoch=epoch,
            args=args,
            save_name=save_name)

    return all_acc, old_acc, new_acc, acc_list, ind_map


# ============================================================================
# PSEUDO LABELING FUNCTIONS (Fine-Grained GCD)
# ============================================================================

def evaluate_per_class_accuracy(model, test_loader, known_classes, num_classes, device='cuda'):
    """Evaluate per-class accuracy using K-Means clustering (same as test())."""
    from sklearn.cluster import KMeans
    from scipy.optimize import linear_sum_assignment as linear_assignment

    model.eval()

    # Extract features
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0], batch[1]
            images = images.to(device)

            feats = model[0](images)
            feats = F.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    all_feats = np.concatenate(all_feats)
    all_labels = np.concatenate(all_labels).astype(int)

    # K-Means clustering
    kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    # Hungarian match cluster id -> gt label id (same alignment as split_cluster_acc_v2);
    # raw K-Means cluster ids are arbitrary, comparing them to labels directly is meaningless
    D = max(preds.max(), all_labels.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for p_, t_ in zip(preds, all_labels):
        w[p_, t_] += 1
    ind = np.vstack(linear_assignment(w.max() - w)).T
    gt_to_pred = {int(gt): int(cl) for cl, gt in ind}

    # Calculate per-class accuracy
    per_class_stats = {c: {'correct': 0, 'total': 0} for c in range(num_classes)}
    known_set = set(known_classes)

    for pred, label in zip(preds, all_labels):
        if label in known_set:
            per_class_stats[label]['total'] += 1
            if pred == gt_to_pred[int(label)]:
                per_class_stats[label]['correct'] += 1

    for c in per_class_stats:
        stats = per_class_stats[c]
        stats['acc'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
    
    # Debug: prediction distribution
    pred_dist = {}
    for p in preds:
        pred_dist[p] = pred_dist.get(p, 0) + 1
    
    return per_class_stats, pred_dist

def log_per_class_stats(per_class_stats, epoch=None, args=None):
    """Log per-class accuracy statistics."""
    header = "PER-CLASS ACCURACY (for Pseudo Labeling)"
    if epoch is not None:
        header += f" — epoch {epoch}"
    args.logger.info("\n" + "="*60)
    args.logger.info(header)
    args.logger.info("="*60)

    accuracies = []
    for c in sorted(per_class_stats.keys()):
        stats = per_class_stats[c]
        acc = stats['acc']
        accuracies.append(acc)
        marker = "★★★" if acc >= 0.8 else ("★★" if acc >= 0.6 else "")
        args.logger.info(f"  Class {c}: {stats['correct']:3d}/{stats['total']:3d} = {acc:.3f} {marker}")

    if accuracies:
        mean_acc = sum(accuracies) / len(accuracies)
        std_acc = (sum((a - mean_acc)**2 for a in accuracies) / len(accuracies)) ** 0.5
        best_class = max(per_class_stats.keys(), key=lambda c: per_class_stats[c]['acc'])
        worst_class = min(per_class_stats.keys(), key=lambda c: per_class_stats[c]['acc'])
        gap = per_class_stats[best_class]['acc'] - per_class_stats[worst_class]['acc']

        args.logger.info(f"\n  Mean Acc: {mean_acc:.3f} | Std: {std_acc:.3f}")
        args.logger.info(f"  Best Class: {best_class} ({per_class_stats[best_class]['acc']:.3f}) | "
                         f"Worst Class: {worst_class} ({per_class_stats[worst_class]['acc']:.3f}) | "
                         f"Bias Gap: {gap:.3f}")

    args.logger.info("="*60 + "\n")



# ============================================================================
# PSEUDO LABELING UPDATE LOGIC
# ============================================================================

def collect_pseudo_labels_from_unlabeled(model, unlabeled_loader, mode, target_class, max_samples, threshold,
                                         used_uq_idxs=None, top_ratio=0.8, max_label=None, device='cuda'):
    """Collect high-confidence samples based on the selected pseudo labeling mode.

    mode 1: samples predicted as target_class, keep the top_ratio fraction with
            highest confidence (capped at max_samples).
    mode 2: any class with confidence >= threshold, capped at max_samples per
            predicted class; if max_label is set, only classes < max_label are
            eligible (their head dims are trained with real CE labels).

    used_uq_idxs: set of uq_idxs already taken in earlier iterations; those are
            skipped so the same image is never added to the labeled set twice.
    """
    import torch.nn.functional as F

    model.eval()
    if used_uq_idxs is None:
        used_uq_idxs = set()

    candidates = []   # (confidence, uq_idx, image tensor, pred)
    seen_uq = []

    with torch.no_grad():
        for batch in unlabeled_loader:
            images = batch[0]
            uq_idxs = batch[2]
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)

            feats = model[0](images)
            feats = F.normalize(feats, dim=-1)
            logits = model[1](feats)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            for img, pred, conf, uq in zip(images, preds, confs, uq_idxs):
                uq = int(uq)
                seen_uq.append(uq)
                if uq in used_uq_idxs:
                    continue
                p = pred.item()
                c = conf.item()

                if mode == 1:
                    if p == target_class:
                        candidates.append({'image': img.cpu(), 'label': p,
                                           'confidence': c, 'uq_idx': uq})
                elif mode == 2:
                    if c >= threshold and (max_label is None or p < max_label):
                        candidates.append({'image': img.cpu(), 'label': p,
                                           'confidence': c, 'uq_idx': uq})

    # Mode 1: keep top_ratio fraction by confidence (user request: e.g. best 80%)
    if mode == 1:
        candidates.sort(key=lambda s: s['confidence'], reverse=True)
        n_keep = int(len(candidates) * top_ratio)
        n_keep = min(n_keep, max_samples)
        selected = candidates[:n_keep]
    else:
        # Mode 2: per-class cap, prefer highest confidence within each class
        per_class = {}
        for s in candidates:
            per_class.setdefault(s['label'], []).append(s)
        selected = []
        for lbl, items in per_class.items():
            items.sort(key=lambda s: s['confidence'], reverse=True)
            selected.extend(items[:max_samples])
        selected.sort(key=lambda s: s['confidence'], reverse=True)

    new_used = {s['uq_idx'] for s in selected}
    return selected, new_used


def create_pseudo_dataset(pseudo_samples):
    """Create dataset from pseudo samples."""
    import torch
    
    class PseudoDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.images = [s['image'] for s in samples]
            self.labels = [s['label'] for s in samples]
            # keep the real uq_idxs from the unlabeled pool so exclusion logic works
            self.uq_idxs = [s['uq_idx'] for s in samples]
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            # Training loop expects 2 contrastive views per sample
            # (ContrastiveLearningViewGenerator); emit duplicated views so
            # collate/chunk(2) work like real dataset items.
            img = self.images[idx]
            return [img, img.clone()], self.labels[idx], self.uq_idxs[idx]
    
    return PseudoDataset(pseudo_samples)


def update_train_loader(train_loader, train_dataset, new_pseudo_samples):
    """Create new train loader with pseudo samples added."""
    import torch
    
    if not new_pseudo_samples:
        return train_loader
    
    pseudo_dataset = create_pseudo_dataset(new_pseudo_samples)
    
    # Get original labeled and unlabeled
    original_labeled = train_dataset.labelled_dataset
    original_unlabeled = train_dataset.unlabelled_dataset
    
    # Create combined labeled dataset
    class CombinedLabeledDataset(torch.utils.data.Dataset):
        def __init__(self, ds1, ds2):
            self.ds1 = ds1
            self.ds2 = ds2
        
        def __len__(self):
            return len(self.ds1) + len(self.ds2)
        
        def __getitem__(self, idx):
            if idx < len(self.ds1):
                return self.ds1[idx]
            return self.ds2[idx - len(self.ds1)]
    
    combined_labeled = CombinedLabeledDataset(original_labeled, pseudo_dataset)
    
    # Create new merged dataset
    from data.data_utils import MergedDataset
    new_train_dataset = MergedDataset(
        labelled_dataset=combined_labeled,
        unlabelled_dataset=original_unlabeled
    )
    
    # Create new sampler
    label_len = len(combined_labeled)
    unlab_len = len(original_unlabeled)
    
    sample_weights = [1 if i < label_len else label_len / unlab_len 
                     for i in range(len(new_train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(new_train_dataset)
    )
    
    # Create new loader
    new_loader = DataLoader(
        new_train_dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
        num_workers=train_loader.num_workers
    )
    
    return new_loader



# ============================================================================
# PSEUDO LABELING UPDATE LOGIC
# ============================================================================

# ============================================================================
# PSEUDO-LABEL BIAS EVIDENCE (charts + ground-truth audit for diagnostics)
# All outputs are diagnostics only — nothing here feeds back into training.
# ============================================================================

def _pseudo_vis_dir(args):
    path = os.path.abspath(os.path.join(args.model_dir, os.pardir, 'visualizations'))
    os.makedirs(path, exist_ok=True)
    return path


def build_uq_to_true_label(unlabelled_dataset):
    """Map uq_idx -> true label over the whole unlabeled pool.

    Works for ConcatDataset (recursing) and plain datasets exposing .targets.
    Used ONLY by the contamination audit.
    """
    mapping = {}

    def _walk(ds):
        # ConcatDataset of LT subsets (the cifar pipeline builds 2 sub-datasets)
        inner = getattr(ds, 'datasets', None)
        if inner is not None:
            for d in inner:
                _walk(d)
            return
        targets = getattr(ds, 'targets', None)
        uq_idxs = getattr(ds, 'uq_idxs', None)
        if targets is not None and uq_idxs is not None:
            for t, u in zip(targets, uq_idxs):
                mapping[int(u)] = int(t)

    _walk(unlabelled_dataset)
    return mapping


def count_labeled_per_class(labeled_dataset, out):
    """Accumulate per-class counts of the original labeled set into `out`."""
    targets = getattr(labeled_dataset, 'targets', None)
    if targets is not None:
        for t in targets:
            c = int(t)
            out[c] = out.get(c, 0) + 1
    else:
        # ConcatDataset-style labeled set without .targets
        inner = getattr(labeled_dataset, 'datasets', None)
        if inner is not None:
            for d in inner:
                count_labeled_per_class(d, out)


def count_pseudo_into(class_counts, pseudo_samples):
    """Add pseudo samples into a running per-class count dict."""
    for s in pseudo_samples:
        c = int(s['label'])
        class_counts[c] = class_counts.get(c, 0) + 1


def audit_pseudo_samples(pseudo_samples, uq2true, num_labeled):
    """Classify each selected pseudo sample against its true label.

    Returns counts:
      n_true_correct         — pseudo label == true old-class label
      n_wrong_old            — truly another OLD class (confirmation-bias fuel)
      n_novel_contamination  — truly a NOVEL class mislabeled as old (novel swallowing)
      novel_true_labels      — Counter {true novel class: n} (which classes got eaten)
    """
    from collections import Counter
    n_true_correct = n_wrong_old = n_novel = 0
    novel_true_labels = Counter()
    for s in pseudo_samples:
        true_lbl = uq2true.get(int(s['uq_idx']))
        if true_lbl is None:
            continue
        if true_lbl < num_labeled:
            if true_lbl == int(s['label']):
                n_true_correct += 1
            else:
                n_wrong_old += 1
        else:
            n_novel += 1
            novel_true_labels[true_lbl] += 1
    return {
        'n_selected_gt': len(pseudo_samples),
        'n_true_correct': n_true_correct,
        'n_wrong_old': n_wrong_old,
        'n_novel_contamination': n_novel,
        'novel_true_labels': dict(novel_true_labels),
    }


def log_pseudo_audit(audit, pseudo_iteration, target_class, args):
    """Print the audit verdict for one pseudo-labeling iteration."""
    sel = audit['n_selected_gt']
    if sel == 0:
        args.logger.info(f"[PSEUDO AUDIT iter {pseudo_iteration}] No samples selected — nothing to audit.")
        return

    pct_ok = 100.0 * audit['n_true_correct'] / sel
    pct_wrong_old = 100.0 * audit['n_wrong_old'] / sel
    pct_novel = 100.0 * audit['n_novel_contamination'] / sel

    args.logger.info("\n" + "-" * 60)
    args.logger.info(f"[PSEUDO AUDIT iter {pseudo_iteration}] Ground-truth check of selected pseudo labels"
                     + (f" (target class {target_class})" if target_class is not None else ""))
    args.logger.info(f"  Correct (old==old):        {audit['n_true_correct']:4d}/{sel} = {pct_ok:5.1f}%")
    args.logger.info(f"  WRONG OLD class:           {audit['n_wrong_old']:4d}/{sel} = {pct_wrong_old:5.1f}%")
    args.logger.info(f"  NOVEL class swallowed:     {audit['n_novel_contamination']:4d}/{sel} = {pct_novel:5.1f}%")
    if audit['novel_true_labels']:
        args.logger.info(f"  True identities of swallowed novel samples: "
                         f"{dict(sorted(audit['novel_true_labels'].items()))}")
    verdict = ("OK (<10% wrong)" if pct_ok >= 90 else
               "BIASED (10-30% wrong)" if pct_ok >= 70 else
               "HEAVILY BIASED (>30% wrong)")
    args.logger.info(f"  VERDICT: {verdict}")
    args.logger.info("-" * 60 + "\n")


def log_labeled_distribution(orig_counts, cur_counts, pseudo_added_per_class, args):
    """Log how the labeled-set distribution drifts as pseudo samples pile up."""
    all_classes = sorted(set(orig_counts) | set(cur_counts))
    orig_vals = np.array([orig_counts.get(c, 0) for c in all_classes], dtype=float)
    cur_vals = np.array([cur_counts.get(c, 0) for c in all_classes], dtype=float)

    orig_max = orig_vals.max() if orig_vals.max() > 0 else 1.0
    cur_max = cur_vals.max() if cur_vals.max() > 0 else 1.0
    imb_orig = orig_max / max(orig_vals.min(), 1)
    imb_cur = cur_max / max(cur_vals.min(), 1)

    args.logger.info("[PSEUDO DISTRIBUTION] Labeled-set composition after injection:")
    for c, o, v in zip(all_classes, orig_vals, cur_vals):
        delta = int(v - o)
        args.logger.info(f"  Class {c}: {int(o):5d} -> {int(v):5d}  (+{delta})")
    args.logger.info(f"  Imbalance ratio (max/min): {imb_orig:.2f} -> {imb_cur:.2f}"
                     f"   {'(WORSE)' if imb_cur > imb_orig * 1.05 else '(~same/better)'}")


def save_per_class_accuracy_chart(per_class_stats, epoch, args):
    """Bar chart of per-class accuracy on known classes; saved every 10 epochs."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        classes = sorted(per_class_stats.keys())
        accuracies = [per_class_stats[c]['acc'] for c in classes]

        fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.7), 4.5))
        colors = ['green' if a >= 0.8 else 'orange' if a >= 0.5 else 'red' for a in accuracies]
        ax.bar(classes, accuracies, color=colors)
        mean_acc = float(np.mean(accuracies)) if len(accuracies) else 0.0
        ax.axhline(mean_acc, color='blue', linestyle='--',
                   label=f'Mean {mean_acc:.3f}')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Known class')
        ax.set_ylabel('Cluster-matched accuracy')
        ax.set_title(f'Per-Class Accuracy — epoch {epoch}')
        ax.legend()
        plt.tight_layout()

        path = os.path.join(_pseudo_vis_dir(args), f'per_class_acc_epoch{epoch}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        args.logger.info(f'[VISUALIZATION] Per-class accuracy chart: {path}')
        return path
    except Exception as e:
        args.logger.warning(f'[VISUALIZATION] Per-class chart failed: {e}')
        return None


def save_bias_evidence_chart(events, pseudo_iteration, args):
    """One figure per pseudo iteration: contamination pie + labeled drift bars."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ev = events[-1]
        sel = max(ev['n_selected_gt'], 1)
        ok, wrong, novel = ev['n_true_correct'], ev['n_wrong_old'], ev['n_novel_contamination']

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

        # Panel 1: what did the model actually promote?
        sizes = [ok, wrong, novel]
        labels = [f'true correct\n{ok}', f'wrong old class\n{wrong}', f'novel swallowed\n{novel}']
        colors = ['#2e7d32', '#f9a825', '#c62828']
        total = sum(sizes)
        if total > 0:
            axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                        startangle=90, textprops={'fontsize': 9})
        axes[0].set_title(f'Select quality — iter {pseudo_iteration}\n'
                          f'{ev["n_selected_gt"]} samples promoted')

        # Panel 2: cumulative contamination across iterations
        iters = list(range(1, len(events) + 1))
        cum_sel = np.cumsum([e['n_selected_gt'] for e in events])
        cum_novel = np.cumsum([e['n_novel_contamination'] for e in events])
        cum_wrong = np.cumsum([e['n_wrong_old'] for e in events])
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_bad = np.where(cum_sel > 0, 100.0 * (cum_novel + cum_wrong) / cum_sel, 0.0)
        axes[1].plot(iters, pct_bad, 'r-o')
        axes[1].set_xlabel('Pseudo iteration')
        axes[1].set_ylabel('% selected that are wrong/novel (cumulative)')
        axes[1].set_title('Cumulative pseudo-label error rate')
        axes[1].grid(True, alpha=0.3)

        # Panel 3: labeled-set drift (orig vs current, incl. pseudo)
        class_keys = sorted(k for k in events[-1] if k.startswith('labeled_c'))
        classes = [int(k[len('labeled_c'):]) for k in class_keys]
        vals = [events[-1][k] for k in class_keys]
        axes[2].bar([str(c) for c in classes], vals,
                    color=['steelblue'] * len(classes))
        axes[2].set_xlabel('Labeled class')
        axes[2].set_ylabel('Count (original + pseudo)')
        axes[2].set_title('Labeled-set distribution after injection')

        plt.tight_layout()
        path = os.path.join(_pseudo_vis_dir(args),
                            f'bias_evidence_iter{pseudo_iteration}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        args.logger.info(f'[VISUALIZATION] Bias evidence chart: {path}')
        return path
    except Exception as e:
        args.logger.warning(f'[VISUALIZATION] Bias evidence chart failed: {e}')
        return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--eval-funcs', type=list, default=['v2'])
    parser.add_argument('--dataset-name', type=str, default='cifar100')
    parser.add_argument('--prop-train-labels', type=float, default=0.5)
    parser.add_argument('--grad-from-block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp-root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup-weight', type=float, default=0.35)
    parser.add_argument('--n-views', default=2, type=int)
    parser.add_argument('--memax-weight', type=float, default=1)
    parser.add_argument('--warmup-teacher-temp', default=0.07, type=float)
    parser.add_argument('--teacher-temp', default=0.04, type=float)
    parser.add_argument('--warmup-teacher-temp-epochs', default=30, type=int)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print-freq', default=10, type=int)
    parser.add_argument('--exp-name', default='cifar100', type=str)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--p', default=1.1, type=float)
    parser.add_argument('--test-freq', default=1, type=int)
    parser.add_argument('--ce-warmup', default=1, type=int)
    parser.add_argument('--est-freq', default=10, type=int)
    parser.add_argument('--labeled-classes', default=80, type=int)
    parser.add_argument('--config-file', default='', type=str)
    parser.add_argument('--alpha', default=0.8, type=float)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--tro', default=0.5, type=float)
    parser.add_argument('--stop-epoch', default=200, type=int)
    parser.add_argument('--imb-ratio', default=100, type=int)
    parser.add_argument("--enable-pseudo-labeling", action="store_true", default=False)
    parser.add_argument("--pseudo-mode", type=int, default=1, choices=[1, 2])
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--pseudo-top-ratio", type=float, default=0.8,
                        help="Mode 1: fraction of highest-confidence target-class samples to keep")
    parser.add_argument("--max-samples-per-class", type=int, default=500)
    parser.add_argument("--pseudo-update-freq", type=int, default=1)
    parser.add_argument("--max-pseudo-iterations", type=int, default=3)
    parser.add_argument("--use-exact-exp-root", action="store_true", default=False)
    parser.add_argument("--vis-freq", type=int, default=10,
                        help="Generate PCA/t-SNE/confusion every N test epochs (0 = off)")

    # ----------------------
    # INIT
    # ----------------------

    args = parser.parse_args()
    pid = os.getpid()
    print('MY PID:', pid)

    if args.config_file != '':
        with open('configs/' + args.config_file, 'r') as f:
            args = parser.parse_args(f.read().split())


    device = torch.device('cuda:0')
    args = get_class_splits(args)
    if args.dataset_name == 'cifar10':
        total_class = 10
    elif args.dataset_name == 'cub200':
        total_class = 200
    else:
        total_class = 100

    args.train_classes = range(args.labeled_classes)
    args.unlabeled_classes = range(args.labeled_classes, total_class)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    init_experiment(args, runner_name=['BaCon'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    seed = torch.randint(0, 100000, (1,)).item()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)

    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    train_all_test_trans = deepcopy(train_dataset)
    train_labelled_test_trans = deepcopy(train_dataset.labelled_dataset)
    train_unlabelled_test_trans = deepcopy(train_dataset.unlabelled_dataset)

    train_all_test_trans.labelled_dataset.transform = test_transform

    train_all_test_trans.unlabelled_dataset.datasets[0].transform = test_transform
    train_all_test_trans.unlabelled_dataset.datasets[1].transform = test_transform
    train_unlabelled_test_trans.datasets[0].transform = test_transform
    train_unlabelled_test_trans.datasets[1].transform = test_transform
    train_labelled_test_trans.transform = test_transform


    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    train_all_test_trans_loader = DataLoader(train_all_test_trans, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)
    train_labelled_test_trans_loader = DataLoader(train_labelled_test_trans, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)
    train_unlabelled_test_trans_loader = DataLoader(train_unlabelled_test_trans, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------

    from model import vision_transformer as vits


    cl_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim, out_dim=65536, nlayers=args.num_mlp_layers)
    ce_head = CE_Head(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)

    cl_backbone = deepcopy(backbone)
    ce_backbone = deepcopy(backbone)

    # ----------------------
    # TRAIN
    # ----------------------

    ce_backbone = ce_backbone.to(device)
    ce_head = ce_head.to(device)
    cl_backbone = cl_backbone.to(device)
    cl_head = cl_head.to(device)
    
    train_dual(ce_backbone, ce_head, cl_backbone, cl_head, train_loader, test_loader_labelled, args)







