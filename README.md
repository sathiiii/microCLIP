# microCLIP: Unsupervised CLIP Adaptation via Coarse-Fine Token Fusion for Fine-Grained Image Classification

![Status](https://img.shields.io/badge/status-active-success.svg)
![Conference](https://img.shields.io/badge/WACV-2026-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

---

## 🌸 Overview

**microCLIP** is a lightweight self-training framework that adapts CLIP for **fine-grained image classification** without requiring labeled data.  

While CLIP is strong in zero-shot transfer, it primarily relies on coarse global features. microCLIP enhances CLIP with localized, fine-grained cues, enabling sharper attention, more accurate pseudo-labels, and improved classification accuracy across challenging benchmarks.  

**Key ideas:**
- **Saliency-Oriented Attention Pooling (SOAP):** builds a fine-grained `[FG]` token from salient patch embeddings.  
- **TokenFusion:** fuses `[FG]` with the global `[CLS]` token for coarse–fine alignment.  
- **Two-headed LLM-derived classifier:** a frozen prior (`W_LLM`) and a learnable classifier (`W*_LLM`) stabilize pseudo-labeling.  
- **Dynamic Knowledge Aggregation:** convexly combines static CLIP/LLM priors with evolving TokenFusion logits.  

microCLIP improves **+2.90%** average accuracy across 13 fine-grained benchmarks, setting a new state-of-the-art for unsupervised CLIP adaptation.

---

## 🚀 Features

- Unsupervised CLIP adaptation (no labels required).  
- Fine-grained token `[FG]` guided by saliency.  
- Coarse–fine fusion via TokenFusion.  
- Stable pseudo-labeling with dual classifiers.  
- State-of-the-art performance on 13 fine-grained datasets.  

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/username/microCLIP.git
cd microCLIP

# Create environment
conda env create -f environment.yml
conda activate microclip
```

## 🔧 Usage

### Train (UA Fine-tuning)

```bash
python train.py --dataset dataset-name --train_config ours_vit_b_32_cupl_proto
```

### Evaluate

```bash
python evaluate.py --dataset dataset-name --ckpt-path path/to/checkpoint.pth
```