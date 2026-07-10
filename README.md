# GANs for Data Augmentation and Domain Adaptation

Improving pneumonia classification on pediatric chest X-rays with **generative data
augmentation** and **adversarial domain adaptation**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=weightsandbiases&logoColor=black)

---

## Overview

The "Chest X-Ray Pneumonia" dataset (Kermany et al.) is imbalanced: roughly **1 healthy
case for every 3 pneumonia cases**. A ResNet-18 trained on it tends to sacrifice recall on
the minority NORMAL class, which lowers the Macro F1-score.

This project studies whether **synthetic images of the minority class** can help. Two
conditional GANs generate NORMAL X-rays, the synthetic data is added to the training set,
and the effect on the classifier is measured. Because synthetic images introduce a
**domain gap** (real vs. generated), a DANN aligns the two feature distributions, both
within the same hospital (synthetic → real) and across hospitals (Kermany → VinDr).

## Method at a glance

| Component | Choice |
|---|---|
| Classifier | ResNet-18, ImageNet-pretrained, `layer3`/`layer4` + head fine-tuned |
| Metric | Macro F1-score (robust to imbalance) |
| Generators | conditional **WGAN-GP** and **SNGAN** (Spectral Norm + hinge), both with a PatchGAN critic and BAGAN-style balanced sampling |
| Generative quality | **FID** and **KID** |
| Feature-space analysis | **PCA**, **t-SNE**, **UMAP** on the 512-dim features |
| Domain adaptation | **DANN** with Gradient Reversal (synthetic → real and cross-hospital) |

## Repository layout

```
.
├── config.py                     # paths + all hyperparameters + random seed
│
├── models/                       # network architectures
│   ├── resnet.py                 #   ResNet-18 classifier
│   ├── wgan.py                   #   conditional WGAN-GP (128x128)
│   ├── sngan_128.py              #   conditional SNGAN (128x128)
│   ├── sngan.py                  #   conditional SNGAN (256x256)
│   ├── dann_synth.py             #   DANNSynth (feature extractor + label + domain heads)
│   └── dann.py                   #   standalone unsupervised DANN model
│
├── dataset/                      # data preparation and loaders
│   ├── loader.py                 #   dataset setup + ResNet/GAN/balanced dataloaders
│   ├── create_modified_dataset.py#   builds the re-split dataset from the raw Kermany data
│   ├── create_dataset_VINDR.ipynb#   prepares the VinDr target set (cross-hospital)
│   ├── synth_real_loader.py      #   source/target loaders for synthetic->real DANN
│   ├── dann_loader.py            #   generic source/target loaders for DANN
│   └── build_augmented.py        #   builds an augmented train set from a GAN checkpoint
│
├── training/                     # reusable training loops
│   ├── train.py                  #   train_resnet, train_wgangp, train_sngan
│   └── train_dann_synth.py       #   DANN synthetic->real loop + evaluation helper
│
├── evaluation/                   # metrics and analysis (CLI, load checkpoints)
│   ├── eval.py                   #   shared helpers (evaluate_on_test, generate, plots)
│   ├── eval_resnet.py            #   Macro F1 of a ResNet checkpoint
│   ├── eval_dann.py              #   Macro F1 of a DANN (synthetic->real) checkpoint
│   ├── eval_dann_cross.py        #   DANN across multiple test sets (+ AUPRC)
│   ├── eval_roc_pr.py            #   ROC / Precision-Recall curves
│   ├── eval_gradcam.py           #   Grad-CAM attention + border-energy metric
│   └── metrics_kid_fid.py        #   FID / KID over GAN training epochs
│
├── experiments/                  # runnable pipelines and experiments
│   ├── main_wgan.py              #   full WGAN-GP pipeline (baseline -> GAN -> augmented)
│   ├── main_sngan.py             #   full SNGAN pipeline (baseline -> GAN -> augmented)
│   ├── Resnet_classifier.py      #   train + evaluate a ResNet on a given train folder
│   ├── balance_strategies.py     #   class-weighting and oversampling baselines
│   ├── main_classical_aug.py     #   ablation: classical geometric augmentation
│   ├── ablation_synth_study.py   #   ablation: % of synthetic images added
│   ├── feature_analysis.py       #   PCA / t-SNE / UMAP of the feature space
│   ├── main_dann_synth.py        #   DANN synthetic -> real (control vs DANN)
│   └── main_dann_cross_hospital.py#  DANN Kermany -> VinDr (source-only vs DANN)
│
└── utils/                        # helpers
    ├── seed.py                   #   set_seed for reproducibility
    ├── visualization.py          #   save a grid of generated images during training
    └── logging.py                #   Weights & Biases logging helpers
```

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies: PyTorch, torchvision, scikit-learn, matplotlib, seaborn, numpy,
`torch-fidelity` (FID/KID), `umap-learn`, and `wandb` for logging.

> **Run every script from the repository root.** Scripts in subfolders add the root to
> `sys.path`, and `config.py` uses relative paths (`./data`, `./results`) that only resolve
> from the root. From a notebook, prefix commands with `!`.

## Running the code

### Full pipelines

These run the whole thing end to end (baseline ResNet → train GAN → train ResNet on the
augmented set → compare) and take no arguments.

```bash
python experiments/main_sngan.py     # SNGAN
python experiments/main_wgan.py      # WGAN-GP
```

### Modular workflow

The same steps can be run separately and composed. Given a trained GAN checkpoint:

```bash
# 1. build an augmented training set (fill 75% of the class gap with synthetic NORMAL)
python dataset/build_augmented.py --gan-ckpt G_epoch_210.pth --gap 75 \
    --out-dir ./results/augmented_dataset --synth-dir ./results/augmented_synth

# 2. train + evaluate a ResNet on it
python experiments/Resnet_classifier.py \
    --train-dir ./results/augmented_dataset/train --tag Augmented

# 3. (baseline for comparison) train + evaluate a ResNet on the real data
python experiments/Resnet_classifier.py --tag Baseline
```

### Experiments

| Script | What it does | Arguments |
|---|---|---|
| `experiments/main_sngan.py` / `main_wgan.py` | full pipeline for each GAN | none |
| `experiments/Resnet_classifier.py` | train + evaluate a ResNet on a folder | `--train-dir`, `--tag` |
| `experiments/balance_strategies.py` | class-weighting and oversampling baselines | none |
| `experiments/main_classical_aug.py` | classical-augmentation ablation (25/50/75/100%) | none |
| `experiments/ablation_synth_study.py` | synthetic-percentage ablation (25/50/75/100%) | config block at top of file |
| `experiments/feature_analysis.py` | PCA / t-SNE / UMAP of real vs synthetic features | `--extractor {resnet,dann}`, `--resnet-ckpt`, `--dann-ckpt`, `--gan-ckpt`, `--num-samples` |
| `experiments/main_dann_synth.py` | synthetic → real DANN (control vs DANN) | config block at top of file |
| `experiments/main_dann_cross_hospital.py` | cross-hospital DANN (Kermany → VinDr) | config block at top of file |

### Evaluation and analysis

All evaluation scripts take checkpoint paths from the command line.

| Script | What it does | Arguments |
|---|---|---|
| `evaluation/eval_resnet.py` | Macro F1 of a ResNet checkpoint | `--ckpt` (required), `--wandb`, `--out` |
| `evaluation/eval_dann.py` | Macro F1 of a DANN (synthetic → real) checkpoint | `--ckpt` (required), `--wandb`, `--out` |
| `evaluation/eval_dann_cross.py` | DANN across source/target test sets (+ AUPRC) | `--ckpt`, `--label` |
| `evaluation/eval_roc_pr.py` | ROC / PR curves, baseline vs augmented | `--baseline-ckpt`, `--finale-ckpt`, `--label` |
| `evaluation/eval_gradcam.py` | Grad-CAM attention + border-energy metric | `--ckpt`, `--baseline-ckpt` |
| `evaluation/metrics_kid_fid.py` | FID / KID over GAN epochs | `--ckpt-dir`, `--gan {sngan128,sngan256,wgan}` |
| `dataset/build_augmented.py` | build an augmented train set from a GAN checkpoint | `--gan-ckpt` (required), `--gap`, `--gan`, `--out-dir`, `--synth-dir`, `--no-reuse` |

Examples:

```bash
python evaluation/eval_resnet.py --ckpt best_model_Augmented.pth
python evaluation/eval_dann.py --ckpt best_DANN_Synth.pth --wandb
python evaluation/eval_roc_pr.py --baseline-ckpt best_Baseline.pth --finale-ckpt best_Finale.pth
python evaluation/metrics_kid_fid.py --ckpt-dir ./results/sngan_checkpoints/
```

## Dataset

- **Source (Kermany).** Pediatric chest X-rays, NORMAL vs PNEUMONIA. The original
  validation split contains only 16 images, so `dataset/create_modified_dataset.py`
  re-partitions train/val into a larger, more balanced validation set while leaving the
  test set untouched.
- **Target (VinDr).** Used only for the cross-hospital experiment, prepared by
  `dataset/create_dataset_VINDR.ipynb`. Its labels are used only to evaluate the aligned
  model on a different hospital's distribution.

## Notes

- Some pipelines and analyses (`ablation_synth_study.py`, `main_dann_synth.py`,
  `main_dann_cross_hospital.py`, `eval_gradcam.py`) select their checkpoints and options
  from a small config block at the top of the file; edit it before running.
- Reproducibility is controlled by the global seed in `config.py` (`set_seed`).
- All training and most evaluation runs log metrics, curves and images to Weights & Biases.
