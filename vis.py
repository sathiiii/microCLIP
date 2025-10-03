import re
import os
import cv2
import json
import math
import copy
import torch
import pickle
import random
import denseCRF
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import utils.utils as utils
from PIL import Image, ImageOps
from timm.utils import accuracy 
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from distinctipy import distinctipy
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from torchvision import transforms as T
from engine_self_training import evaluate
from collections import Counter, defaultdict
from utils.build_dataset import build_dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
from utils.model import CLIPClassifier, tokenize
from skimage.measure import label as measure_label
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from torchvision.transforms.functional import InterpolationMode 


torch.manual_seed(0)
np.random.seed(0)
torch.set_grad_enabled(False)
torch.cuda.set_device('cuda:3')

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
args = AttrDict(
    batch_size=64,
    template='templates.json',
    classname='classes.json',
    clip_model='ViT-B/32',
    image_mean=(0.48145466, 0.4578275, 0.40821073),
    image_std=(0.26862954, 0.26130258, 0.27577711),
    input_size=224,
    train_config='configs/train_configs/ours_vit_b_32_cupl_proto.json',
    train_crop_min=0.3,
    color_jitter=0,
    aa='rand-m9-mstd0.5-inc1',
    train_interpolation='bicubic',
    nb_classes=0,
    dataset='pets',
    num_workers=10,
    device='cuda',
    resume='output/pets/2025-05-10_16:12:44_ViT-B_32_8_crops_epoch15_lr0.0001/checkpoint-last.pth',
    vis=True,
    use_gpt3_prompts=True,
    text_finetune_only=False,
    text_descriptions_path='all_prompts/train_prompts/',
    ctx_checkpoint=None,
    alpha=0.5,
    beta=0.9,
    n_samples=16,
    n_crops=8,
    eval=False,
    baseline=False,
    use_crops_for_test=False,
    text_scale=1.0,
    use_naive_token_avg=False,
    use_global_feature_for_query=False,
    use_unr_token=False
)

train_config_path = args.train_config
with open(train_config_path, 'r') as train_config_file:
    train_config = json.load(train_config_file)
dataset_config_path = os.path.join("configs/dataset_configs/", args.dataset + ".json")
with open(dataset_config_path, 'r') as dataset_config_file:
    dataset_params = json.load(dataset_config_file)
device = torch.device(args.device)
args.train_config = train_config
args.dataset_params = dataset_params

batch_size = args.batch_size

dataset_val, _ = build_dataset(is_train=False, args=args)  
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
data_loader_val = torch.utils.data.DataLoader(
    dataset_val, 
    sampler     = sampler_val,
    batch_size  = 4 * batch_size,
    num_workers = 2,
    pin_memory  = True,
    drop_last   = False
)

model = CLIPClassifier(args)
args.nb_classes = len(model.classnames)
checkpoint = torch.load(args.resume, map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)

classnames = model.classnames

def get_norm_tfm(args):
    return T.Normalize(
                mean=torch.tensor(args.image_mean),
                std=torch.tensor(args.image_std))
    
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

def get_label_colors():
    # base colors
    label_colors = {
            0: [255, 0, 0], 
            1: [0, 255, 0],     
            2: [0, 0, 255],     
            3: [255, 255, 0],   
            4: [255, 165, 0],
            5: [255, 192, 203],
            6: [160, 32, 240],
            7: [0, 255, 255], 
            8: [128, 0, 0],
            9: [128, 128, 0],
            10: [128, 0, 128],
            11: [255, 105, 180],
            12: [75, 0, 130],
            13: [0, 128, 0],
            14: [0, 128, 128],
            15: [70, 130, 180],
            16: [255, 69, 0],
            17: [139, 69, 19],
            18: [0, 0, 128], 
            19: [255, 20, 147], 
            20: [255, 140, 0]}
    
    # add 30 other random colors
#     start_from = list(label_colors.keys())[-1] + 1
#     for c in range(start_from, 100):
#         label_colors[c] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
    
    return label_colors

def compute_attn_pooled_features(model, query, patch_feats, mask=None):
    local_feats = patch_feats
    if mask is not None:
        local_feats = local_feats * mask.unsqueeze(-1)  # (B, N, D)
    # Append empty token
    local_feats = torch.cat([local_feats, torch.zeros(local_feats.size(0), 1, local_feats.size(-1)).to(local_feats.device)], dim=1)  # (B, N+1, D)
    # Compute attention weights
    attn_weights = torch.softmax(model.query_proj(query).unsqueeze(1) @ model.key_proj(local_feats).permute(0, 2, 1) / math.sqrt(patch_feats.size(-1)), dim=-1)
    attn_weights = attn_weights.squeeze(1)
    # Attention pool the local features
    local_feat = torch.sum(model.value_proj(local_feats) * attn_weights.unsqueeze(-1), dim=1)
    composite_feat = F.normalize(local_feat @ model.clip_model.visual.proj, p=2, dim=-1)
    return attn_weights, composite_feat

def get_local_global_features(args, img, model, dino, visualize=False, normalize=False, iter_name=''):
    norm_tfm = get_norm_tfm(args)
    
    if normalize:
        img_normalized = norm_tfm(img)
    else:
        img_normalized = img
    global_feat, local_feat, global_cls_token = model(img_normalized, return_patch_tokens=True, return_cls_token=True)

    _, _, key_feats = model.extract_last_layer_key_feats(img_normalized)
    model.clear_layer_activations()
    key_feats = F.normalize(key_feats[:, 1:], p=2, dim=-1)

    grid_size = int(np.sqrt(local_feat.size(1)))
    fg_mask, fiedler_vector = get_foreground_mask_ncut(key_feats, grid_size)
            
    if args.use_naive_token_avg: # No Attention Pooling
        local_pooled_feats = local_feat.mean(dim=1)
        pooled_local_feats = F.normalize(local_pooled_feats @ model.clip_model.visual.proj, p=2, dim=-1)
        attn_weights = None
    else:
        if args.use_global_feature_for_query: # No Normalized Cut
            query_feats = global_cls_token
        else: # Use Normalized Cut
            masked_feat = (local_feat) * fg_mask.flatten(1, 2).unsqueeze(-1)  # (B, N, D)
            mask_sum = fg_mask.flatten(1, 2).sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            query_feats = masked_feat.sum(dim=1) / mask_sum  # (B, D)
        
        # Attention pool the local features
        attn_weights, pooled_local_feats = compute_attn_pooled_features(model, query_feats, local_feat)
    
    if visualize: # Visualize Normalized Cut
        upscaled_mask = fg_mask[0].cpu().numpy().astype(np.float32)
        upscaled_mask = cv2.resize(upscaled_mask, dsize=(args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST)
        img = (img[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        unary_potentials = torch.from_numpy(upscaled_mask).long()
        unary_potentials = F.one_hot(unary_potentials.long(), num_classes=2).float().numpy()
        out = denseCRF.densecrf(img, unary_potentials, (10, 40, 13, 3, 3, 5.0))
        seg_mask = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
        for label, color in get_label_colors().items():
            seg_mask[out == label] = color
            seg_mask[out == 0] = 0
            if label == out.max():
                break
        seg_mask = cv2.resize(seg_mask, dsize=(args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST)   
        result = cv2.addWeighted(img, 0.5, seg_mask, 0.5, 0)
        plt.figure(figsize=(8, 6))
        plt.imshow(result)
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(f'NCut_{iter_name}.png', bbox_inches='tight', dpi=300)
    
    return global_feat, pooled_local_feats, attn_weights

tfm1 = T.Compose([
    T.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
    T.RandomCrop(args.input_size),
])
tfm2 = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=torch.tensor(args.image_mean),
        std=torch.tensor(args.image_std))
])

rand_id = np.random.randint(len(dataset_val))
# Fix the following when you need to test a specific image!!
# rand_id = 5712
print(f'Sample ID: {rand_id}')
try:
    dataloader_returns = dataset_val.samples[rand_id]
except:
    dataloader_returns = dataset_val.data[rand_id]
    
# Try until you get the particual
    
img_path, gt_label = dataloader_returns
img_orig = Image.open(img_path).convert('RGB')
img = tfm1(img_orig)
img_train_weak = tfm2(img).unsqueeze(0).cuda()
global_feat, local_feat, cls_token = model(img_train_weak, return_patch_tokens=True, return_cls_token=True)

# patch_feats = dino.get_intermediate_layers(img_train_weak, n=2)[0]
# dino_feats = patch_feats[:, 1]
# patch_feats = patch_feats[:, 1:]
# patch_feats = patch_feats.view(patch_feats.size(0), 14, 14, patch_feats.size(-1)).permute(0, 3, 1, 2)
# patch_feats = patch_feats.flatten(2).transpose(1, 2)
# grid_size = int(np.sqrt(patch_feats.size(1)))
# fg_mask, fiedler_vector = get_foreground_mask_ncut(patch_feats, grid_size)

# masked_feat = patch_feats * fg_mask.flatten(1, 2).unsqueeze(-1)  # (B, N, D)
# mask_sum = fg_mask.flatten(1, 2).sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
# query_feats = masked_feat.sum(dim=1) / mask_sum  # (B, D)

grid_size = int(np.sqrt(local_feat.size(1)))
fg_mask, fiedler_vector = get_foreground_mask_ncut(local_feat, grid_size)
masked_feat = local_feat * fg_mask.flatten(1, 2).unsqueeze(-1)  # (B, N, D)
mask_sum = fg_mask.flatten(1, 2).sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
query_feats = masked_feat.sum(dim=1) / mask_sum  # (B, D)

# Compute attention weights
attn_weights = torch.softmax(model.query_proj(query_feats).unsqueeze(1) @ model.key_proj(local_feat).permute(0, 2, 1) / math.sqrt(local_feat.size(-1)), dim=-1)
attn_weights = attn_weights.squeeze(1)

pooled_local_feats = torch.sum(model.value_proj(local_feat) * attn_weights.unsqueeze(-1), dim=1)
pooled_local_feats = F.normalize(pooled_local_feats @ model.clip_model.visual.proj, p=2, dim=-1)

attn_weights = (attn_weights - attn_weights.min(dim=1, keepdim=True)[0]) / (attn_weights.max(dim=1, keepdim=True)[0] - attn_weights.min(dim=1, keepdim=True)[0] + 1e-6)

upscaled_attn_weights = F.interpolate(attn_weights.reshape(7, 7).unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
upscaled_attn_weights = upscaled_attn_weights.squeeze(0).squeeze(0).cpu().numpy()

upscaled_mask = fg_mask[0].cpu().numpy().astype(np.float32)
upscaled_mask = cv2.resize(upscaled_mask, dsize=(args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST)
unary_potentials = torch.from_numpy(upscaled_mask)
unary_potentials = F.one_hot(unary_potentials.long(), num_classes=2).float().numpy()
out = denseCRF.densecrf(img, unary_potentials, (10, 40, 13, 3, 3, 5.0))
seg_mask = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
for label, color in get_label_colors().items():
    seg_mask[out == label] = color
    seg_mask[out == 0] = 0
    if label == out.max():
        break
seg_mask = cv2.resize(seg_mask, dsize=(args.input_size, args.input_size), interpolation=cv2.INTER_NEAREST)   
image = (T.ToTensor()(img).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
result = cv2.addWeighted(image, 0.5, seg_mask, 0.5, 0)

CLIP_attn = F.normalize(local_feat, p=2, dim=-1) @ F.normalize(cls_token, p=2, dim=-1).t()
CLIP_attn = (CLIP_attn - CLIP_attn.min(dim=1, keepdim=True)[0]) / (CLIP_attn.max(dim=1, keepdim=True)[0] - CLIP_attn.min(dim=1, keepdim=True)[0] + 1e-6)
CLIP_attn = F.interpolate(CLIP_attn.reshape(7, 7).unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
CLIP_attn = CLIP_attn.squeeze(0).squeeze(0).cpu().numpy()

overlaid_img = (0.5 * T.ToTensor()(img).permute(1, 2, 0).numpy() + 0.5 * plt.cm.jet(upscaled_attn_weights)[..., :3]) / 2
image_np = (T.ToTensor()(img).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
# overlaid_img = image_np.copy()
patch_h, patch_w = args.input_size // 7, args.input_size // 7
attn_map = attn_weights.reshape(7, 7).cpu().numpy()
attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)

# for i in range(7):
#     for j in range(7):
#         alpha = attn_map[i, j]
#         if alpha < 0.05:
#             continue
#         overlay_color = np.array([255, 0, 0], dtype=np.uint8)  # red
#         y1, y2 = i * patch_h, (i + 1) * patch_h
#         x1, x2 = j * patch_w, (j + 1) * patch_w
#         overlaid_img[y1:y2, x1:x2] = (
#             alpha * overlay_color + (1 - alpha) * overlaid_img[y1:y2, x1:x2]
#         ).astype(np.uint8)

CLIP_attn_overlaid_img = (0.5 * T.ToTensor()(img).permute(1, 2, 0).numpy() + 0.5 * plt.cm.jet(CLIP_attn)[..., :3]) / 2
# CLIP_attn_map = CLIP_attn.reshape(7, 7)
# clip_attn_img = image_np.copy()
# CLIP_attn_map = (CLIP_attn_map - CLIP_attn_map.min()) / (CLIP_attn_map.max() - CLIP_attn_map.min() + 1e-6)

# for i in range(7):
#     for j in range(7):
#         alpha = CLIP_attn_map[i, j]
#         if alpha < 0.05:
#             continue
#         overlay_color = np.array([255, 0, 0], dtype=np.uint8)  # red
#         y1, y2 = i * patch_h, (i + 1) * patch_h
#         x1, x2 = j * patch_w, (j + 1) * patch_w
#         clip_attn_img[y1:y2, x1:x2] = (
#             alpha * overlay_color + (1 - alpha) * clip_attn_img[y1:y2, x1:x2]
#         ).astype(np.uint8)

# CLIP_attn_overlaid_img = clip_attn_img

local_logits = pooled_local_feats @ model.get_classifier().t()
local_preds = torch.argmax(local_logits, dim=1)
global_logits = global_feat @ model.get_classifier().t()
global_preds = torch.argmax(global_logits, dim=1)
avg_logits = (local_logits + global_logits) / 2
avg_preds = torch.argmax(avg_logits, dim=1)

print('Ground Truth:', classnames[gt_label])
print('Local Predictions:', classnames[local_preds.item()])
print('Global Predictions:', classnames[global_preds.item()])
print('Average Predictions:', classnames[avg_preds.item()])

# ==== Comment out the following lines when you decied on one image only. Otherwise it'll save multiple images to the disk ====
# fig = plt.figure(figsize=(8, 6))
# plt.imshow(img)
# plt.tight_layout()
# plt.axis('off')
# plt.savefig(f'image_{rand_id}.png', bbox_inches='tight', dpi=300)
# plt.close(fig)

# fig = plt.figure(figsize=(5, 5))
# plt.imshow(overlaid_img)
# plt.axis('off')
# plt.tight_layout()
# plt.savefig(f'local_attn_{args.dataset}.png')
# plt.close(fig)

# fig = plt.figure(figsize=(5, 5))
# plt.imshow(CLIP_attn_overlaid_img)
# plt.axis('off')
# plt.tight_layout()
# plt.savefig(f'CLIP_attn_{args.dataset}.png')
# plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(overlaid_img)
ax[1].set_title('Local Attention Overlay')
ax[1].axis('off')
ax[2].imshow(CLIP_attn_overlaid_img)
ax[2].set_title('CLIP Global Attention Overlay')
ax[2].axis('off')
plt.tight_layout()