---
title: microCLIP
layout: default
---

# microCLIP: Unsupervised CLIP Adaptation via Coarse–Fine Token Fusion for Fine-Grained Image Classification

[![arXiv](https://img.shields.io/badge/arXiv-PREPRINT-b31b1b.svg)](https://arxiv.org/abs/2510.02270)
[![Project](https://img.shields.io/badge/Project-microCLIP-2ea44f.svg)](https://github.com/sathiiii/microCLIP)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

---

## Authors
**Sathira Silva $^1$** · **Eman Ali $^{1,2}$** · **Chetan Arora $^3$** · **Muhammad Haris Khan $^1$**  
$^1$<sub>Mohamed Bin Zayed University of Artificial Intelligence</sub> · $^2$<sub>Alexandria University</sub> · $^3$<sub>IIT Delhi</sub>

---

## Abstract

Unsupervised adaptation of CLIP-based vision-language models (VLMs) for fine-grained image classification requires sensitivity to microscopic local cues. While CLIP exhibits strong zero-shot transfer, its reliance on coarse global features restricts its performance on fine-grained classification tasks. Prior efforts inject fine-grained knowledge by aligning large language model (LLM) descriptions with the CLIP $\texttt{[CLS]}$ token; however, this approach overlooks spatial precision. We propose $\textbf{microCLIP}$, a self-training framework that jointly refines CLIP's visual and textual representations using fine-grained cues. At its core is Saliency-Oriented Attention Pooling (SOAP) within a lightweight TokenFusion module, which builds a saliency-guided $\texttt{[FG]}$ token from patch embeddings and fuses it with the global $\texttt{[CLS]}$ token for coarse-fine alignment. To stabilize adaptation, we introduce a two-headed LLM-derived classifier: a frozen classifier that, via multi-view alignment, provides a stable text-based prior for pseudo-labeling, and a learnable classifier initialized from LLM descriptions and fine-tuned with TokenFusion. We further develop Dynamic Knowledge Aggregation, which convexly combines fixed LLM/CLIP priors with TokenFusion's evolving logits to iteratively refine pseudo-labels. Together, these components uncover latent fine-grained signals in CLIP, yielding a consistent $2.90\%$ average accuracy gain across 13 fine-grained benchmarks while requiring only light adaptation. Our code is available at [https://github.com/sathiiii/microCLIP](https://github.com/sathiiii/microCLIP).

---

## Overall Architecture
<p align="center">
  <img src="./assets/figure2.png" alt="Overall architecture of microCLIP" width="100%" />
</p>

---

## Results

### Comparison to Zero-shot and UA Baselines
<p align="center">
  <img src="./assets/table1.png" alt="Top-1 accuracy comparison across 13 datasets (ViT-B/32 backbone)" width="100%" />
</p>

### Ablation Studies
**Effect of coarse vs fine-grained cues:**  
<p align="center">
  <img src="./assets/table2.png" alt="Ablation on coarse-feature baselines" width="80%" />
</p>

**Effect of SOAP:**  
<p align="center">
  <img src="./assets/table3.png" alt="Ablation on Attention Pooling (SOAP vs baselines)" width="80%" />
</p>

**Dynamic Knowledge Aggregation:**  
<p align="center">
  <img src="./assets/table4.png" alt="Ablation on pseudo-labeler" width="80%" />
</p>

**Two-headed classifier initialization:**  
<p align="center">
  <img src="./assets/table5.png" alt="Two-headed classifier ablation" width="80%" />
</p>

### Backbone Scaling
<p align="center">
  <img src="./assets/table6.png" alt="Results with ViT-B/16 backbone" width="70%" />
</p>

---

## Visualizations

**Sharper local attention via SOAP-guided [FG]:**  
<p align="center">
  <img src="./assets/figure1.png" alt="Attention maps (Birdsnap/RESISC)" width="60%" />
</p>

**[CLS] vs [FG] attention across datasets:**  
<p align="center">
  <img src="./assets/figure9.png" alt="Attention comparison between CLS and FG tokens" width="80%" />
</p>

**Pseudo-label accuracy progression:**  
<p align="center">
  <img src="./assets/figure3.png" alt="Pseudo-labeling accuracy curves" width="60%" />
</p>

**NCut saliency masks:**  
<p align="center">
  <img src="./assets/figure4.png" alt="NCut-based saliency maps on Birdsnap" width="60%" />
</p>

---

## Acknowledgements
This work builds upon the [MUST](https://github.com/salesforce/MUST) repository.  
We thank the authors of [MetaCLIP](https://github.com/facebookresearch/MetaCLIP) for releasing their codebase, which we use in additional experiments.  
We also acknowledge [CuPL](https://github.com/prattai/cupl) for providing GPT-3 generated class descriptions, included in [`all_prompts/`](./all_prompts).

---

## Citation
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

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body);"></script>
