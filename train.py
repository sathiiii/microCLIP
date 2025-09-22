import os
import sys
import time
import json
import torch
import copy
import random
import argparse
import datetime
import warnings
import numpy as np
import torch.nn as nn
from torch import optim
from pathlib import Path
from contextlib import suppress
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn

import utils.utils as utils
from utils.model import CLIPClassifier
from utils.build_dataset import build_dataset
from engine_self_training import train_one_epoch, evaluate, cupl_eval, zs_eval
from utils.utils import NativeScalerWithGradNormCount as NativeScaler


warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser('MUST training and evaluation script', add_help=False)
    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--ablation_name', default='', type=str)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--eval_freq', default=1, type=int) 
    # CLIP parameters
    parser.add_argument("--template", default='templates.json', type=str)
    parser.add_argument("--classname", default='classes.json', type=str)
    parser.add_argument('--clip_model', default=None, help='pretrained clip model name') 
    parser.add_argument('--image_mean', default=(0.48145466, 0.4578275, 0.40821073)) 
    parser.add_argument('--image_std', default=(0.26862954, 0.26130258, 0.27577711)) 
    parser.add_argument('--input_size', default=224, type=int, help='images input size') 
    # training parameters
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument("--train_config", default='ours_vit_b_32_cupl_proto', type=str, help='training configurations')
    parser.add_argument("--text_descriptions_path", default='./all_prompts/train_prompts', type=str, help='path to the text descriptions')
    parser.add_argument("--ce_weight", type=float, default=None, help='cross entropy loss weight')
    parser.add_argument("--fairness_weight", type=float, default=None, help='fairness loss weight')
    parser.add_argument("--n_crops", default=None, type=int, help='number of random crops per image')
    parser.add_argument("--alpha", default=0.5, type=float, help='lower bound for random crop ratio')
    parser.add_argument("--beta", default=0.9, type=float, help='upper bound for random crop ratio')
    parser.add_argument("--gamma", default=None, type=float, help='static-dynamic knowledge fusion ratio for pseudo labelling')
    parser.add_argument("--fusion_ratio", default=0.5, type=float, help='local-global fusion ratio for pseudo labelling')
    # Ablation parameters
    parser.add_argument("--fully_supervised", action='store_true', help='fully supervised training')
    parser.add_argument("--baseline", action='store_true')
    parser.add_argument("--wca_baseline", action='store_true')
    parser.add_argument("--use_fixed_classifier", action='store_true', help='use fixed prototypical classifier')
    # ====== Ablation parameters for attention pooling ======
    parser.add_argument("--use_token_avg_for_query", action='store_true', help='use token average instead of CLS token for attention pooling query')
    parser.add_argument("--use_naive_token_avg", action='store_true', help='use naive token average instead of attention pooling')
    parser.add_argument("--use_ncut_token_avg", action='store_true', help='use NCut selected token average instead of attention pooling')
    parser.add_argument("--use_global_feature_for_query", action='store_true', help='use CLIP global feature for attention pooling query')
    parser.add_argument("--use_random_selection_for_query", action='store_true', help='use random selection of tokens for attention pooling query')
    parser.add_argument("--use_unr_token", action='store_true', help='use UNR token for attention pooling')
    # Optimizer parameters
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate')
    parser.add_argument('--layer_decay', type=float, default=0.65)
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    # Augmentation parameters  
    parser.add_argument('--train_crop_min', default=0.3, type=float)
    parser.add_argument('--color_jitter', type=float, default=0, metavar='PCT')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int, help='number of the classification types')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')
    parser.add_argument('--output_dir', default='', help='path to save checkpoint and log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    # distributed training parameters
    parser.add_argument('--amp', action='store_true')
    return parser.parse_args()

def main(args):
    args.vis = False
    #-------------------------------- Train config --------------------------------
    if os.path.exists(args.train_config):
        train_config_path = args.train_config
    else:
        train_config_path = os.path.join("configs/train_configs/", args.train_config + ".json")
    with open(train_config_path, 'r') as train_config_file:
        train_config = json.load(train_config_file)
        if train_config['method'] == 'wca': dataset_params = train_config
        else:
            dataset_config_path = os.path.join("configs/dataset_configs/", args.dataset + ".json")
            with open(dataset_config_path, 'r') as dataset_config_file:
                dataset_params = json.load(dataset_config_file)
    if args.use_fixed_classifier:
        print('Using a fixed prototypical classifier')
        train_config['use_learnable_classifier'] = False
    if args.lr is not None:
        print(f'Using lr: {args.lr}')
        dataset_params['lr'] = args.lr
    else:
        args.lr = dataset_params['lr']
    if args.clip_model is not None:
        print(f'Using vision backbone: {args.clip_model}')
        train_config['vision_backbone'] = args.clip_model
    args.clip_model = train_config['vision_backbone']
    if args.epochs is not None:
        dataset_params['epochs'] = args.epochs
    else:
        args.epochs = dataset_params['epochs']
    if args.ce_weight is not None:
        dataset_params['ce_weight'] = args.ce_weight
    if args.fairness_weight is not None:
        dataset_params['fairness_weight'] = args.fairness_weight
    if args.n_crops is not None:
        print(f'Using n_crops: {args.n_crops}')
        dataset_params['n_crops'] = args.n_crops
    else:
        args.n_crops = dataset_params['n_crops']
        
    if args.gamma is not None:
        print(f'Using gamma (local-global fusion ratio for PL): {args.gamma}')
        dataset_params['gamma'] = args.gamma
    else:
        args.gamma = dataset_params['gamma']
        
    if not args.output_dir:
        args.output_dir = os.path.join('output', args.dataset)    
        if args.ablation_name != '':
            args.output_dir = os.path.join(args.output_dir, args.ablation_name)
        if train_config['method'] == 'wca':
            args.output_dir = os.path.join(args.output_dir, 
                    "%s_%s%s_epoch%d"%(f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}", 
                        args.clip_model.replace('/', '_'), '_' + args.exp_name if len(args.exp_name) > 0 else '', 
                            dataset_params['epochs']))
        else:
            args.output_dir = os.path.join(args.output_dir, 
                    "%s_%s%s_epoch%d_lr%s"%(f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}", 
                        args.clip_model.replace('/', '_'), '_' + args.exp_name if len(args.exp_name) > 0 else '', 
                            dataset_params['epochs'], str(args.lr)))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.output_dir:    
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(dict(args._get_kwargs())) + "\n")
    # Redirect the stdout of the program to TextLogger object.
    sys.stdout = utils.TextLogger(os.path.join(args.output_dir, "stdout_log.txt"))
    device = torch.device(args.device)
    # ----------------- fix the seed for reproducibility -----------------
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # ------------------- Train Dataset -------------------------------
    args.train_config = train_config
    args.dataset_params = dataset_params
    if args.batch_size is not None:
        print(f'Using batch size: {args.batch_size}')
        dataset_params["batch_size"] = args.batch_size
    batch_size = dataset_params["batch_size"]
    args.batch_size = batch_size
    dataset_train, len_original = build_dataset(is_train=True, args=args)
    print(f'Total number of training samples : { len(dataset_train) }')
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler     = sampler_train,
        batch_size  = batch_size,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False,
    )
    len_data_loader_train = len(data_loader_train)
    args.len_original = len_original
    # -------------------------------- Eval Dataset --------------------------------
    dataset_val, _ = build_dataset(is_train=False, args=args)  
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler     = sampler_val,
        batch_size  = 4*batch_size,
        num_workers = args.num_workers,
        pin_memory  = False,
        drop_last   = False
    )
    # -------------------------------- Build Model --------------------------------
    model = CLIPClassifier(args)

    args.nb_classes = len(model.classnames)
    print("List of learnable parameters:")
    print("-----------------------------------------------------------------------")
    if train_config['method'] == 'ours':
        ## ------------------------  Freeze every thing except the layer norm ------------------------
        params = list()
        for name, param in model.named_parameters():
            param.requires_grad_(False)
            if not 'zs' in name:
                if 'ln' in name or 'bn' in name:
                    param.requires_grad = True
                if 'classifier' in name:
                   param.requires_grad = True
                if not args.wca_baseline:
                    if 'unr_token' in name:
                        param.requires_grad = True
                    if 'query_proj' in name:
                        param.requires_grad = True
                    if 'key_proj' in name:
                        param.requires_grad = True
                    if 'value_proj' in name:
                        param.requires_grad = True
                if param.requires_grad:
                    params.append((name, param))
                    print(f'{name}')
        # -------------------------------- optimizer --------------------------------
        args.min_lr       = args.min_lr * 2
        args.eval_freq    = train_config['eval_freq']
        num_training_steps_per_epoch = len_data_loader_train
        model_without_ddp = model

        print("-----------------------------------------------------------------------")
        no_decay = ['LayerNorm.bias', 'LayerNorm.weight']
        print(f'Using learning rate: {args.lr}')
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], \
                'weight_decay': 0.1},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)], \
                'weight_decay': 0.0}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
        if args.amp:
            loss_scaler = NativeScaler()
            amp_autocast = torch.cuda.amp.autocast
        else:
            loss_scaler = None
            amp_autocast = suppress
    elif train_config['method'] == 'wca':
        for name, param in model_without_ddp.named_parameters():
            param.requires_grad_(False)
        optimizer = None
        loss_scaler = None
        
    n_parameters      = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print('-----------------------------------------------------------------------')
    print(f'n_parameters : {n_parameters}')
    print('-----------------------------------------------------------------------')
    #--------------------------------- load Model --------------------------
    if train_config['source_model'] == 'CLIP':
        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=None)
    
    test_stats = evaluate(args, data_loader_val, model, None, device, eval_func=zs_eval if train_config["use_handcrafted"] else cupl_eval, classnames=model.classnames, show_per_class=True, show_harmonic_mean=True)
    print(f"Zero-shot accuracy on the {len(dataset_val)} test images: {test_stats['acc']:.2f}%")
    if args.eval: exit(0)
    
    # -------------------------------- Train ----------------------------------------
    start_time = time.time()
    gpu_mem_usage = []
    acc_list = []
    PL_acc = []
    
    print(f"\nStarting training for {args.epochs} epochs.")
    print("=====================================")
    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.reset_peak_memory_stats()
        mem_start = torch.cuda.memory_allocated(device)
        if train_config['method'] == 'ours':
            train_stats = train_one_epoch(
                args, 
                model, 
                data_loader_train, 
                optimizer, 
                amp_autocast, 
                device, 
                epoch,
                loss_scaler,
                lr_schedule_values,
                train_config,
                start_steps=epoch * num_training_steps_per_epoch,
                )
            PL_acc.append(train_stats['acc_PL'])
            if data_loader_val is not None:
                test_stats = evaluate(args, data_loader_val, model, device, classnames=model.classnames, show_per_class=True, show_harmonic_mean=True)
                print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc']:.2f}%")
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'test_{k}': v for k, v in test_stats.items()},
                                'epoch': epoch}
                acc_list.append(test_stats['acc'])

            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, epoch_name="last", model_ema=None)
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
            
        peak_mem = torch.cuda.max_memory_allocated(device)
        avg_epoch_mem = (mem_start + peak_mem) / 2
        gpu_mem_usage.append(avg_epoch_mem)
    #------------------------------------------------------------------------------------------
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Adaptation time {total_time_str}')
    avg_gpu_mem = np.mean(gpu_mem_usage) / (1024**2)
    print(f"\nAverage GPU Memory Usage over {args.epochs} epochs: {avg_gpu_mem:.2f} MB")

    if train_config['method'] == 'ours':
        plt.figure(figsize=(10, 5))
        plt.plot(acc_list)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epoch')
        plt.grid()
        plt.savefig(os.path.join(args.output_dir, 'accuracy.png'))
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.plot(PL_acc)
        plt.xlabel('Epoch')
        plt.ylabel('PL Accuracy')
        plt.title('PL Accuracy vs Epoch')
        plt.grid()
        plt.savefig(os.path.join(args.output_dir, 'PL_accuracy.png'))
        plt.close()

if __name__ == '__main__':
    opts = get_args()
    main(opts)
