import sys
import math
import time
import torch
import numpy as np
from typing import Iterable
from timm.utils import accuracy 
from collections import defaultdict
import torch.nn.functional as F
from torchvision import transforms as T

torch.autograd.set_detect_anomaly(True)

import utils.utils as utils


def get_norm_tfm(args):
    return T.Normalize(
                mean=torch.tensor(args.image_mean),
                std=torch.tensor(args.image_std))

def compute_composite_crop_features(args, global_feature, crop_features):
    # Compute relevance scores
    crop_relevance = (crop_features @ global_feature.unsqueeze(-1)).squeeze(-1)  # (batch_size, n_crops)
    # Compute composite feature
    composite_feat = (crop_features * F.softmax(crop_relevance, dim=-1).unsqueeze(-1)).sum(dim=1)
    composite_feat = F.normalize(composite_feat, p=2, dim=-1)
    
    return crop_relevance, composite_feat

def compute_attn_pooled_features(model, query, patch_feats, mask=None, use_unr_token=False):
    local_feats = patch_feats
    if mask is not None:
        local_feats = local_feats * mask.unsqueeze(-1)  # (B, N, D)
    # Append empty token
    if use_unr_token:
        local_feats = torch.cat([local_feats, model.unr_token.expand(local_feats.size(0), 1, local_feats.size(-1))], dim=1)  # (B, N+1, D)
    else:
        local_feats = torch.cat([local_feats, torch.zeros(local_feats.size(0), 1, local_feats.size(-1)).to(local_feats.device)], dim=1)  # (B, N+1, D)
    # Compute attention weights
    attn_weights = torch.softmax(model.query_proj(query).unsqueeze(1) @ model.key_proj(local_feats).permute(0, 2, 1) / math.sqrt(patch_feats.size(-1)), dim=-1)
    attn_weights = attn_weights.squeeze(1)
    # Attention pool the local features
    local_feat = torch.sum(model.value_proj(local_feats) * attn_weights.unsqueeze(-1), dim=1)
    composite_feat = F.normalize(local_feat @ model.clip_model.visual.proj, p=2, dim=-1)
    return attn_weights, composite_feat

def get_foreground_mask_ncut(patch_keys, grid_size, tau=0.2, eps=1e-6):
    A = torch.matmul(patch_keys, patch_keys.transpose(1, 2))
    A = (A > tau).float()  # Convert to binary adjacency matrix
    A = torch.where(A == 0, torch.full_like(A, eps), A)  # Replace 0s with eps
    # Compute Degree Matrix D (B, num_patches, num_patches)
    D = torch.diag_embed(torch.sum(A, dim=2))
    
    # Compute normalized Laplacian L_sym = D⁻¹/² * (D - A) * D⁻¹/²
    D_inv_sqrt = torch.diag_embed(1.0 / (torch.sqrt(torch.sum(A, dim=2)) + eps))
    L_sym = torch.matmul(torch.matmul(D_inv_sqrt, (D - A)), D_inv_sqrt)
    # Solve eigenproblem: L_sym * x = λ * x
    _, eigenvectors = torch.linalg.eigh(L_sym)  # Output shape: (B, num_patches, num_patches)
    
    fiedler_vector = eigenvectors[:, :, 1]
    # Compute bipartition threshold based on mean value
    avg = fiedler_vector.std(dim=1, keepdim=True)
    bipartition = fiedler_vector > avg  # Boolean mask, shape: (B, num_patches)
    
    # Find the seed (index of max absolute value in eigenvector)
    seed_indices = torch.argmax(torch.abs(fiedler_vector), dim=1)  # Shape: (B,)

    # Ensure consistent sign across batch
    for i in range(patch_keys.size(0)):
        if bipartition[i, seed_indices[i]] != 1:
            fiedler_vector[i] *= -1
            bipartition[i] = ~bipartition[i]  # Logical NOT for batch item i

    # Reshape to (B, grid_size, grid_size)
    bipartition = bipartition.view(-1, grid_size, grid_size)
    
    return bipartition, fiedler_vector

def get_local_global_features(args, img, model, normalize=False):
    norm_tfm = get_norm_tfm(args)
    
    if normalize:
        img_normalized = norm_tfm(img)
    else:
        img_normalized = img
    global_feat, local_feat, global_cls_token = model(img_normalized, return_patch_tokens=True, return_cls_token=True)

    grid_size = int(np.sqrt(local_feat.size(1)))
    fg_mask, fiedler_vector = get_foreground_mask_ncut(local_feat, grid_size)
            
    if args.use_naive_token_avg: # No Attention Pooling
        local_pooled_feats = local_feat.mean(dim=1)
        pooled_local_feats = F.normalize(local_pooled_feats @ model.clip_model.visual.proj, p=2, dim=-1)
        attn_weights = None
    if args.use_ncut_token_avg: # No Attention Pooling
        local_pooled_feats = (local_feat) * fg_mask.flatten(1, 2).unsqueeze(-1)
        mask_sum = fg_mask.flatten(1, 2).sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        local_pooled_feats = local_pooled_feats.sum(dim=1) / mask_sum
        pooled_local_feats = F.normalize(local_pooled_feats @ model.clip_model.visual.proj, p=2, dim=-1)
        attn_weights = None
    else:
        if args.use_global_feature_for_query: # No Normalized Cut
            query_feats = global_cls_token
        elif args.use_token_avg_for_query: # No Normalized Cut
            query_feats = local_feat.mean(dim=1)
        elif args.use_random_selection_for_query: # No Normalized Cut
            # Select a random index from the second dimension for all batches
            random_indices = torch.randint(0, local_feat.size(1), (local_feat.size(0), 1), device=local_feat.device)
            query_feats = local_feat.gather(1, random_indices.unsqueeze(-1).expand(-1, -1, local_feat.size(-1))).squeeze(1)  # (B, D)
        else: # Use Normalized Cut
            masked_feat = (local_feat) * fg_mask.flatten(1, 2).unsqueeze(-1)  # (B, N, D)
            mask_sum = fg_mask.flatten(1, 2).sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            query_feats = masked_feat.sum(dim=1) / mask_sum  # (B, D)
        
        # Attention pool the local features
        attn_weights, pooled_local_feats = compute_attn_pooled_features(model, query_feats, local_feat, use_unr_token=args.use_unr_token)
    
    return global_feat, pooled_local_feats, attn_weights

def train_one_epoch(args,
                    model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    amp_autocast,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    lr_schedule_values,
                    train_config,
                    start_steps: int = 0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch + 1}/{args.epochs}] Iter:'
    print_freq = 10
    start_time = time.time()
    norm_tfm = get_norm_tfm(args)
    
    for data_iter_step, (inputs, true_labels, index) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        img_train_weak = inputs[:, 0].to(device, non_blocking=True)
        img_train_strong = inputs[:, 1].to(device, non_blocking=True)
        crops = inputs[:, 2:].to(device, non_blocking=True)
        true_labels = true_labels.to(device, non_blocking=True)
        
        it = start_steps + data_iter_step  # global training iteration
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None: 
                    param_group["lr"] = lr_schedule_values[it]
                    
        gamma = args.dataset_params['gamma']
        PL_classifier = model.get_fixed_classifier() if not train_config['use_learnable_classifier'] else model.get_classifier()
               
        # Generate pseudo-labels.
        if not args.fully_supervised:
            with torch.no_grad():
                if args.baseline:
                    weak_feats = model(norm_tfm(img_train_weak))
                    pseudo_logits = weak_feats @ PL_classifier.T
                elif args.wca_baseline:
                    global_feats = model(norm_tfm(img_train_weak))
                    crop_feats = model(norm_tfm(crops).flatten(0, 1)) # (n_crops * batch_size, feat_dim)
                    crop_feats = crop_feats.view(-1, args.dataset_params['n_crops'], crop_feats.size(-1)) # (batch_size, n_crops, feat_dim)
                    crop_relevance, composite_feat = compute_composite_crop_features(args, global_feats, crop_feats)
                    pseudo_logits = composite_feat @ PL_classifier.T
                else:
                    global_feats, local_feats, attn_map = get_local_global_features(args, img_train_weak, model, normalize=True)
                    new_feat = F.normalize(0.5 * global_feats + 0.5 * local_feats, dim=-1) # The feature that's being trained on and used for inference
                    
                    crop_feats = model(norm_tfm(crops).flatten(0, 1)) # (n_crops * batch_size, feat_dim)
                    crop_feats = crop_feats.view(-1, args.dataset_params['n_crops'], crop_feats.size(-1)) # (batch_size, n_crops, feat_dim)
                    crop_relevance, composite_feat = compute_composite_crop_features(args, global_feats, crop_feats) # Localized visual prompting
                    
                    global_logits = composite_feat @ model.get_fixed_classifier().T
                    new_logits = new_feat @ model.get_classifier().T
                    pseudo_logits = gamma * global_logits + (1 - gamma) * new_logits
                
                conf, pseudo_labels = torch.max(pseudo_logits, dim=1)
                metric_logger.update(PL_conf=conf.mean().item())
        else:
            weak_feats = model(norm_tfm(img_train_weak))
            pseudo_labels = true_labels
        
        # Forward pass.
        with amp_autocast():
            if args.baseline or args.wca_baseline:
                strong_feats = model(norm_tfm(img_train_strong))
                output = 100 * strong_feats @ model.get_classifier().T
                loss = F.cross_entropy(output, pseudo_labels)
            else:
                strong_global_feats, strong_local_feats, _ = get_local_global_features(args, img_train_strong, model, normalize=True)
                global_logits = 100 * strong_global_feats @ model.get_classifier().T
                local_logits = 100 * strong_local_feats @ model.get_classifier().T
                output = args.fusion_ratio * global_logits + (1 - args.fusion_ratio) * local_logits
                loss = args.dataset_params['ce_weight'] * F.cross_entropy(output, pseudo_labels)
                
            if not args.fully_supervised:
                probs = F.softmax(output, dim=-1)
                probs_batch_avg = probs.mean(0)    
                if data_iter_step == 0:
                    probs_avg = probs_batch_avg
                else:    
                    probs_avg = 0.5 * (probs_avg.detach() + probs_batch_avg)
                loss_fair = -(torch.log(probs_avg)).mean() / 0.5
                loss += args.dataset_params['fairness_weight'] * loss_fair
        
        metric_logger.update(loss=loss.item())
        if not args.wca_baseline and not args.baseline:
            metric_logger.update(acc_local=(torch.argmax(local_logits, dim=1) == true_labels).float().mean().item() * 100)
            metric_logger.update(acc_global=(torch.argmax(global_logits, dim=1) == true_labels).float().mean().item() * 100)
        metric_logger.update(acc=(torch.argmax(output, dim=1) == true_labels).float().mean().item() * 100)
        metric_logger.update(acc_PL=(pseudo_labels == true_labels).float().mean().item() * 100)
        
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)
    
        optimizer.zero_grad()
        if loss.requires_grad:
            if loss_scaler is not None:
                grad_norm = loss_scaler(loss, optimizer, clip_grad=1.0, parameters=model.parameters(), create_graph=False)
                metric_logger.update(grad_norm=grad_norm)
            else:                   
                loss.backward(create_graph=True)       
                optimizer.step()
                
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    epoch_time = time.time() - start_time
    
    print('-----------------------------------------------------------------------')
    print(f"Averaged stats: {epoch} : {metric_logger}, Time: {epoch_time:.2f}s")
    print('-----------------------------------------------------------------------')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
def zs_eval(args, model, inputs):
    norm_tfm = get_norm_tfm(args)
    feat_test = model(norm_tfm(inputs))
    output = 100. * feat_test @ model.classname_embeddings.t()
    return output
    
def ours_eval(args, model, inputs):
    global_feats, local_feats, attn_map = get_local_global_features(args, inputs, model, normalize=True)
    global_logits = 100 * global_feats @ model.get_classifier().T
    local_logits = 100 * local_feats @ model.get_classifier().T
    return global_logits, local_logits
    
def cupl_eval(args, model, inputs):
    norm_tfm = get_norm_tfm(args)
    feat_test = model(norm_tfm(inputs))
    output = 100. * feat_test @ model.get_classifier().t()
    return output
    
@torch.no_grad()
def evaluate(args,
             data_loader, 
             model, 
             device, 
             eval_func=ours_eval, 
             classnames=None, 
             show_per_class=False, 
             show_harmonic_mean=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # Dictionary to store per-class metrics
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    # Switch to evaluation mode
    model.eval()
    norm_tfm = get_norm_tfm(args)
        
    for i, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        inputs = batch[0].to(device, non_blocking=True)
        target = batch[1].to(device, non_blocking=True)
        
        # Compute output
        if args.baseline or args.wca_baseline:
            img_feats = model(norm_tfm(inputs))
            output = 100 * img_feats @ model.get_classifier().T
        else:
            output = eval_func(args, model, inputs)
            if isinstance(output, tuple):
                local_logits, global_logits = output
                output = args.fusion_ratio * global_logits + (1 - args.fusion_ratio) * local_logits
                local_acc = accuracy(local_logits, target)[0]
                global_acc = accuracy(global_logits, target)[0]
                metric_logger.update(acc_local=local_acc.item(), acc_global=global_acc.item())
        
        # Compute predictions and accumulate per-class metrics
        _, preds = torch.max(output, 1)
        correct = preds.eq(target)
        
        for t, p in zip(target, correct):
            class_correct[t.item()] += p.item()
            class_total[t.item()] += 1
        
        # Overall accuracy
        acc = accuracy(output, target)[0]
        metric_logger.meters['acc'].update(acc.item(), n=inputs.shape[0])
    
    return_results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    # Calculate and display per-class accuracy
    if show_per_class and classnames is not None:
        print("\nPer-Class Accuracy Results:")
        accuracies = []
        for label, total in class_total.items():
            classname = classnames[label]  # Use the classnames argument provided
            correct = class_correct[label]
            if total > 0:
                ACC = 100.0 * correct / total
                if ACC > 0:  # Only include non-zero accuracies for harmonic mean
                    accuracies.append(ACC / 100.0)  # Convert percentage to decimal
                print(
                    "* class: {} ({})\t"
                    "total: {:,}\t"
                    "correct: {:,}\t"
                    "acc: {:.2f}%".format(
                        label, classname, total, correct, ACC 
                    )
                )
        
        # Calculate Mean Accuracy
        mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        print(f"\n* Mean Accuracy: {mean_accuracy * 100:.2f}%")

        # Calculate Harmonic Mean Accuracy (if enabled and non-zero accuracies exist)
        if show_harmonic_mean and len(accuracies) > 0:
            harmonic_mean_accuracy = len(accuracies) / sum(1.0 / acc for acc in accuracies)
            return_results['hm_acc'] = harmonic_mean_accuracy * 100
            print(f"* Harmonic Mean Accuracy: {harmonic_mean_accuracy * 100:.2f}%")

    print()
    
    return return_results