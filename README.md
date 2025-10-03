# microCLIP: Unsupervised CLIP Adaptation via Coarse-Fine Token Fusion for Fine-Grained Image Classification

![Status](https://img.shields.io/badge/status-active-success.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2501.01234-b31b1b.svg)](https://arxiv.org/abs/2510.02270)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## 📢 Latest Updates

- **Oct-03-25:** Preprint available on [arXiv](https://arxiv.org/abs/2510.02270).
- **Oct-03-25:** Initial release of microCLIP code.

## Table of Contents
- [💡 Overview](#-overview)
- [📊 Results](#-results)
- [📦 Installation](#-installation)
- [🗂️ Datasets](#-datasets)
- [🔧 Usage](#-usage)
- [🙏 Acknowledgements](#-acknowledgements)
- [📜 Citation](#-citation)

---

## 💡 Overview

**microCLIP** is a lightweight self-training framework that adapts CLIP for **fine-grained image classification** without requiring labeled data.  

While CLIP is strong in zero-shot transfer, it primarily relies on coarse global features. microCLIP enhances CLIP with localized, fine-grained cues, enabling sharper attention, more accurate pseudo-labels, and improved classification accuracy across challenging benchmarks.  

**Key ideas:**
- **Saliency-Oriented Attention Pooling (SOAP):** builds a fine-grained `[FG]` token from salient patch embeddings.  
- **TokenFusion:** fuses `[FG]` with the global `[CLS]` token for coarse–fine alignment.  
- **Two-headed LLM-derived classifier:** a frozen prior and a learnable classifier stabilize pseudo-labeling.  
- **Dynamic Knowledge Aggregation:** convexly combines static CLIP/LLM priors with evolving TokenFusion logits.  

microCLIP improves **+2.90%** average accuracy across 13 fine-grained benchmarks, setting a new state-of-the-art for unsupervised CLIP adaptation.

### Overall Architecture
<p align="center">
  <img src="./docs/assets/figure2.png" alt="Overall architecture of microCLIP" width="100%" />
</p>

---

## 📊 Results

### Comparison to Zero-shot and UA Baselines
<p align="center">
  <img src="./docs//assets/table1.png" alt="Top-1 accuracy comparison across 13 datasets (ViT-B/32 backbone)" width="100%" />
</p>

### Ablation Studies
**Effect of coarse vs fine-grained cues:**  
<p align="center">
  <img src="./docs//assets/table2.png" alt="Ablation on coarse-feature baselines" width="80%" />
</p>

**Effect of SOAP:**  
<p align="center">
  <img src="./docs//assets/table3.png" alt="Ablation on Attention Pooling (SOAP vs baselines)" width="80%" />
</p>

**Dynamic Knowledge Aggregation:**  
<p align="center">
  <img src="./docs//assets/table4.png" alt="Ablation on pseudo-labeler" width="80%" />
</p>

**Two-headed classifier initialization:**  
<p align="center">
  <img src="./docs//assets/table5.png" alt="Two-headed classifier ablation" width="80%" />
</p>

### Backbone Scaling
<p align="center">
  <img src="./docs//assets/table6.png" alt="Results with ViT-B/16 backbone" width="70%" />
</p>

---

### Visualizations

**Sharper local attention via SOAP-guided [FG]:**  
<p align="center">
  <img src="./docs//assets/figure1.png" alt="Attention maps (Birdsnap/RESISC)" width="60%" />
</p>

**[CLS] vs [FG] attention across datasets:**  
<p align="center">
  <img src="./docs//assets/figure9.png" alt="Attention comparison between CLS and FG tokens" width="80%" />
</p>

**Pseudo-label accuracy progression:**  
<p align="center">
  <img src="./docs//assets/figure3.png" alt="Pseudo-labeling accuracy curves" width="60%" />
</p>

**NCut saliency masks:**  
<p align="center">
  <img src="./docs//assets/figure4.png" alt="NCut-based saliency maps on Birdsnap" width="60%" />
</p>

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/sathiiii/microCLIP.git
cd microCLIP

# Create environment
conda env create -f environment.yml
conda activate microclip
```

## 🗂️ Datasets

- Dataset paths are defined in [`configs/dataset_catalog.json`](./configs/dataset_catalog.json).
You will need to update these paths to point to your local dataset locations.

- Dataset label files are provided in [`configs/classes.json`](./configs/classes.json).

- For dataset preparation, we recommend using the scripts from the [VISSL repository](https://github.com/facebookresearch/vissl/tree/main/extra_scripts/datasets).

- Check [this issue](https://github.com/pytorch/vision/issues/7545) for guidance on downloading Stanford Cars dataset.

## 🔧 Usage

### Train (UA Fine-tuning)

```bash
python train.py --dataset dataset-name --train_config ours_vit_b_32_cupl_proto
```

### Evaluate

```bash
python evaluate.py --dataset dataset-name --ckpt-path path/to/checkpoint.pth
```

## 🙏 Acknowledgements

This work builds upon the [MUST](https://github.com/salesforce/MUST) repository. We thank the authors for their open-source code.

We thank the authors of [MetaCLIP](https://github.com/facebookresearch/MetaCLIP) for releasing their codebase, which we use in our additional experiments.  

We also acknowledge [CuPL](https://github.com/prattai/cupl) for providing GPT-3 generated class descriptions, which we include in our repository under [`all_prompts/`](./all_prompts).

## 📜 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{silva2025microclipunsupervisedclipadaptation,
      title={microCLIP: Unsupervised CLIP Adaptation via Coarse-Fine Token Fusion for Fine-Grained Image Classification}, 
      author={Sathira Silva and Eman Ali and Chetan Arora and Muhammad Haris Khan},
      year={2025},
      eprint={2510.02270},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.02270}, 
}
```