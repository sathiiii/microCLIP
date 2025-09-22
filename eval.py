#!/usr/bin/env python3
import os
import sys
import json
import time
import torch
import random
import argparse
import warnings
import numpy as np
from pathlib import Path

import utils.utils as utils
from utils.model import CLIPClassifier
from utils.build_dataset import build_dataset
from engine_self_training import evaluate, cupl_eval, zs_eval

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser('microCLIP evaluation script', add_help=False)

    # --- Bookkeeping / IO ---
    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--ablation_name', default='', type=str)
    parser.add_argument('--output_dir', default='', help='Path to write logs/metrics')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--seed', default=0, type=int)

    # --- Dataset / batching ---
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')
    parser.add_argument('--batch_size', default=None, type=int, help='eval batch size (defaults to train config)')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--image_mean', default=(0.48145466, 0.4578275, 0.40821073))
    parser.add_argument('--image_std', default=(0.26862954, 0.26130258, 0.27577711))

    # --- Text / CLIP setup ---
    parser.add_argument("--template", default='templates.json', type=str)
    parser.add_argument("--classname", default='classes.json', type=str)
    parser.add_argument('--clip_model', default=None, help='pretrained CLIP model name (overrides config)')

    # --- microCLIP training config resolver (same behavior as train.py) ---
    parser.add_argument("--train_config", default='ours_vit_b_32_cupl_proto', type=str,
                        help='training configuration key or JSON file path')
    parser.add_argument("--text_descriptions_path", default='./all_prompts/train_prompts', type=str)

    # --- Checkpoint to load ---
    parser.add_argument('--ckpt-path', default='', type=str,
                        help='Path to a checkpoint .pth file (if empty, runs zero-shot / text-prototype eval)')

    # --- Display options ---
    parser.add_argument('--show_per_class', action='store_true', help='print per-class accuracy')
    parser.add_argument('--show_harmonic_mean', action='store_true', help='print harmonic mean across classes')

    # parity with train.py flags that influence model building / ablations
    parser.add_argument("--wca_baseline", action='store_true')
    parser.add_argument("--use_fixed_classifier", action='store_true', help='use fixed prototypical classifier')

    return parser.parse_args()


def resolve_train_and_dataset_cfg(args):
    # Resolve train_config path exactly as in train.py
    if os.path.exists(args.train_config):
        train_config_path = args.train_config
    else:
        train_config_path = os.path.join("configs/train_configs/", args.train_config + ".json")

    with open(train_config_path, 'r') as f:
        train_config = json.load(f)

    if train_config['method'] == 'wca':
        dataset_params = train_config
    else:
        dataset_config_path = os.path.join("configs/dataset_configs/", args.dataset + ".json")
        with open(dataset_config_path, 'r') as df:
            dataset_params = json.load(df)

    # Allow simple overrides (kept minimal for eval)
    if args.clip_model is not None:
        train_config['vision_backbone'] = args.clip_model
    args.clip_model = train_config['vision_backbone']

    # Batch size for eval: default to 4x train batch for speed unless user sets it
    if args.batch_size is None:
        args.batch_size = 4 * dataset_params.get("batch_size", 32)

    return train_config, dataset_params


def build_val_loader(args):
    # Build validation dataset & loader — mirrors train.py (SequentialSampler, larger batch)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return dataset_val, data_loader_val


def main(args):
    # Output directory
    if not args.output_dir:
        # Put eval logs alongside dataset folder by default
        args.output_dir = os.path.join('output', args.dataset, 'eval')
        if args.ablation_name:
            args.output_dir = os.path.join(args.output_dir, args.ablation_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Log CLI args
    with open(os.path.join(args.output_dir, "eval_args.txt"), "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(args.__dict__), indent=2) + "\n")

    # Optional: redirect stdout to a text logger (same style as train.py)
    sys.stdout = utils.TextLogger(os.path.join(args.output_dir, "stdout_log.txt"))

    # Device & seeds
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Resolve configs (keeps parity with train.py)
    train_config, dataset_params = resolve_train_and_dataset_cfg(args)
    args.train_config = train_config
    args.dataset_params = dataset_params

    # Build val loader
    dataset_val, data_loader_val = build_val_loader(args)
    print(f"Eval dataset: {args.dataset}  |  #samples: {len(dataset_val)}  |  batch_size: {args.batch_size}")

    # Build model
    model = CLIPClassifier(args)
    model.eval()
    args.nb_classes = len(model.classnames)

    # Select evaluation function:
    # - If the train config uses handcrafted prompts, run pure CLIP zero-shot (zs_eval)
    # - Otherwise, run the LLM-description prototypical classifier (cupl_eval)
    eval_fn = zs_eval if train_config.get("use_handcrafted", False) else cupl_eval

    # Load checkpoint if provided
    if args.ckpt_path:
        if not os.path.isfile(args.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
        print(f"Loading checkpoint from: {args.ckpt_path}")
        # We reuse the same util as train.py — it respects args.resume if set
        # This will load model state (optimizer/scaler ignored since we pass None)
        args.resume = args.ckpt_path
        utils.auto_load_model(
            args=args,
            model=model,
            model_without_ddp=model,
            optimizer=None,
            loss_scaler=None,
            model_ema=None
        )
    else:
        print("No --ckpt-path provided. Running zero-shot / text-prototype evaluation with current initialization.")

    # Run evaluation
    t0 = time.time()
    with torch.no_grad():
        stats = evaluate(
            args=args,
            data_loader=data_loader_val,
            model=model,
            device=device,
            eval_func=eval_fn,
            classnames=model.classnames,
            show_per_class=args.show_per_class,
            show_harmonic_mean=args.show_harmonic_mean
        )
    dt = time.time() - t0

    # Print & persist summary
    mode = "Fine-tuned" if args.ckpt_path else ("Zero-shot" if eval_fn is zs_eval else "Text-proto (LLM) zero-shot")
    print("-----------------------------------------------------------------------")
    print(f"{mode} accuracy on {len(dataset_val)} test images: {stats['acc']:.2f}%")
    if 'hmean' in stats:
        print(f"Harmonic mean: {stats['hmean']:.2f}%")
    print(f"Eval time: {dt/60:.2f} min")
    print("-----------------------------------------------------------------------")

    # Save JSON metrics
    metrics_out = {
        "mode": mode,
        "dataset": args.dataset,
        "num_samples": len(dataset_val),
        "acc": float(stats.get("acc", 0.0)),
        "hmean": float(stats.get("hmean", 0.0)) if 'hmean' in stats else None,
        "ckpt_path": args.ckpt_path if args.ckpt_path else None,
        "eval_time_sec": dt,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Optional: write per-class metrics if provided by evaluate(...)
    if 'per_class' in stats and isinstance(stats['per_class'], dict):
        with open(os.path.join(args.output_dir, "per_class.json"), "w") as f:
            json.dump(stats['per_class'], f, indent=2)


if __name__ == '__main__':
    opts = get_args()
    main(opts)
