# Vision Transformer (ViT-B/16) — Feature Extraction vs Fine-tuning

> Transfer learning with ViT-B/16 on CIFAR-10 & STL10
> MSc Artificial Intelligence — Deep Learning & Multimedia Data Analysis

---

## Overview

This project extends Assignment 1 (ResNet18) by evaluating a **Vision Transformer (ViT-B/16)** under the same transfer learning framework on the same datasets. Two strategies are compared:

- **Feature Extraction (FE):** All pretrained layers are frozen. Only a new classification head (768 → 10) is trained — approximately **7,690 trainable parameters**.
- **Fine-tuning (FT):** The last two transformer encoder blocks (blocks 10 and 11), the final LayerNorm, and the new head are unfrozen — approximately **14.2M trainable parameters**.

Both strategies are evaluated on **CIFAR-10** and **STL10**, and results are compared directly against ResNet18 (Assignment 1).

---

## Results

### ViT-B/16

| | FE — CIFAR-10 | FT — CIFAR-10 | FE — STL10 | FT — STL10 |
|---|---|---|---|---|
| **Test Accuracy** | 95.22% | 96.72% | 98.35% | 98.89% |
| **Accuracy Gain (FT vs FE)** | — | +1.50% | — | +0.54% |
| **Trainable Params (FE)** | ~7,690 | — | ~7,690 | — |
| **Trainable Params (FT)** | — | 14,184,970 | — | 14,184,970 |
| **Training Time** | 14.2 min | 16.8 min | 1.5 min | 1.7 min |
| **Training Images** | 45,000 | 45,000 | 4,500 | 4,500 |
| **Image Resolution** | 32×32 | 32×32 | 96×96 | 96×96 |

### Comparison with ResNet18 (Assignment 1)

| Model | Strategy | CIFAR-10 | STL10 |
|---|---|---|---|
| ResNet18 | FE | 79.19% | 93.16% |
| ResNet18 | FT | 94.95% | 95.54% |
| **ViT-B/16** | **FE** | **95.22%** | **98.35%** |
| **ViT-B/16** | **FT** | **96.72%** | **98.89%** |

> ViT Feature Extraction (95.22%) already surpasses ResNet18 Fine-tuning (94.95%) on CIFAR-10.

---

## Key Findings

- **ViT FE outperforms ResNet18 FT** on both datasets — global attention produces more transferable representations than local convolutions.
- **The FE vs FT gap collapses for ViT**: +1.50% on CIFAR-10 vs +15.76% for ResNet18. Feature Extraction alone is sufficient.
- **Resolution sensitivity is reduced**: ViT FE drops only 3% from STL10 to CIFAR-10, vs 14% for ResNet18 FE, due to patch-based processing.
- **ViT trains faster than ResNet18** despite 8x more parameters, due to GPU-parallelized attention and reduced batch size.
- **Overfitting is more aggressive in ViT FT**: train accuracy reaches 100% by epoch 3 on STL10 — best-weights checkpointing is essential.

---

## Model Architecture

ViT-B/16 pretrained on ImageNet (1.2M images, 1000 classes).

- Input image split into 196 patches of 16×16 pixels
- Each patch projected to a 768-d embedding
- 12 Transformer encoder blocks with Multi-Head Self-Attention
- [CLS] token used for classification

The original classification head is replaced with:

```
Dropout(p=0.3) → Linear(768 → 10)
```

In fine-tuning, `encoder.layers[-2]`, `encoder.layers[-1]`, and `encoder.ln` are additionally unfrozen.

---

## Datasets

**CIFAR-10**
- 60,000 color images at 32×32 pixels across 10 classes
- Split: 45,000 train / 5,000 val / 10,000 test

**STL10**
- 13,000 labeled images at 96×96 pixels across 10 classes
- Split: 4,500 train / 500 val / 8,000 test

All images are resized to 224×224 and normalized with ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Training data is augmented with random horizontal flips and random crops.

---

## Training Setup

| Setting | Feature Extraction | Fine-tuning |
|---|---|---|
| Epochs | 15 | 15 |
| Optimizer | Adam | Adam |
| Learning Rate | 1e-3 | 1e-4 |
| LR Scheduler | StepLR (×0.5 every 5 epochs) | StepLR (×0.5 every 5 epochs) |
| Batch Size | 64 | 64 |
| Loss | Cross-Entropy | Cross-Entropy |
| Checkpointing | Best val accuracy | Best val accuracy |

The lower learning rate for fine-tuning avoids catastrophic forgetting of pretrained transformer weights.

---

## Repository Structure

```
├── drnn_cifar10_vit.ipynb    # CIFAR-10 experiment
├── drnn_stl10_vit.ipynb      # STL10 experiment
├── DRNN_Report.pdf           # Full report
└── README.md
```

---

## Requirements

```bash
torch
torchvision
numpy
matplotlib
scikit-learn
opencv-python
```

---

## Visualizations

The notebooks include:
- Training/validation accuracy and loss curves
- Normalized confusion matrices
- Grad-CAM attention maps for both Feature Extraction and Fine-tuning models (hooked into `encoder.layers[-1].mlp`)
