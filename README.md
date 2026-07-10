# GANs for Data Augmentation and Domain Adaptation

Pneumonia classification on pediatric chest X-rays (Kermany dataset) using generative
data augmentation and adversarial domain adaptation.

The pipeline trains a ResNet-18 classifier on an imbalanced dataset, generates synthetic
images of the minority class (NORMAL) with two conditional GANs (WGAN-GP and SNGAN), and
studies how the synthetic data affects the classifier. It also applies a DANN
(Domain-Adversarial Neural Network) to align the real and synthetic feature distributions,
both within the same hospital (synthetic -> real) and across hospitals (Kermany -> VinDr).

## How to run

All scripts are meant to be launched **from the repository root**, e.g.:

```bash
python experiments/main_sngan.py
python evaluation/eval_resnet.py --ckpt path/to/best_model.pth
```

Scripts inside subfolders add the repository root to `sys.path`, so imports and the
relative paths in `config.py` (`./data`, `./results`) resolve correctly only when run
from the root. Training and evaluation log to Weights & Biases.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Repository structure

### `config.py`
Central configuration: input/output paths and all hyperparameters (ResNet, WGAN-GP,
SNGAN, DANN) and the global random seed.

### `models/`
Network architectures.

- `resnet.py` — ResNet-18 classifier, ImageNet-pretrained, with only `layer3`, `layer4`
  and the final linear layer trainable.
- `wgan.py` — conditional WGAN-GP (128x128): generator, PatchGAN critic (and a plain
  critic variant), and the gradient-penalty function.
- `sngan_128.py` — conditional SNGAN (128x128) with Spectral Normalization and hinge loss:
  generator, PatchGAN critic (and a plain critic variant).
- `sngan.py` — the same SNGAN architecture at 256x256.
- `dann_synth.py` — `DANNSynth`: ResNet-18 feature extractor + label predictor + domain
  discriminator + Gradient Reversal Layer (used for both DANN experiments).
- `dann.py` — a standalone DANN model for unsupervised domain adaptation.

### `dataset/`
Dataset preparation and data loaders.

- `loader.py` — dataset setup and dataloaders: RGB loaders for the ResNet, a grayscale
  loader for the GANs, and a balanced (oversampling) loader.
- `create_modified_dataset.py` — one-off script that builds the re-partitioned dataset
  (larger, balanced validation split) from the original Kermany data.
- `create_dataset_VINDR.ipynb` — notebook that prepares the VinDr target dataset used in
  the cross-hospital experiment.
- `synth_real_loader.py` — source/target loaders for the synthetic -> real DANN
  (real images as source, synthetic images as target).
- `dann_loader.py` — generic source/target dataloaders for DANN.
- `build_augmented.py` — builds an augmented training set from a GAN checkpoint and a
  gap percentage: generates synthetic NORMAL images and merges them with the real train
  set. Runnable via CLI (`--gan-ckpt`, `--gap`, `--gan`, ...).

### `training/`
Reusable training loops.

- `train.py` — `train_resnet`, `train_wgangp`, `train_sngan`.
- `train_dann_synth.py` — DANN synthetic -> real training loop and its evaluation helper.

### `evaluation/`
Metrics and analysis. All scripts take checkpoint paths from the command line.

- `eval.py` — shared helpers: `evaluate_on_test` (classification report, confusion matrix,
  ROC/PR logging), `generate_synthetic_images`, `plot_comparison`.
- `eval_resnet.py` — evaluate a ResNet checkpoint (Macro F1) on the real test set.
- `eval_dann.py` — evaluate a DANN (synthetic -> real) checkpoint on the real test set.
- `eval_dann_cross.py` — evaluate a DANN checkpoint across several test sets (source and
  target), reporting F1, Macro F1, Accuracy and AUPRC.
- `eval_roc_pr.py` — ROC and Precision-Recall curves comparing the baseline and the
  augmented ResNet.
- `eval_gradcam.py` — Grad-CAM attention analysis on real vs synthetic images, with a
  border-energy metric to detect shortcuts.
- `metrics_kid_fid.py` — FID and KID of a GAN's samples across its training epochs.

### `experiments/`
Runnable pipelines and experiments.

- `main_wgan.py` — full WGAN-GP pipeline: baseline ResNet, train the WGAN-GP, train a
  ResNet on the augmented set, compare.
- `main_sngan.py` — full SNGAN (complete) pipeline with the same three phases.
- `Resnet_classifier.py` — train and evaluate a ResNet on a given training folder
  (real or augmented) via `--train-dir` / `--tag`.
- `balance_strategies.py` — imbalance baselines: class weighting and oversampling.
- `main_classical_aug.py` — ablation of classical geometric augmentation (small rotation,
  translation, scaling) at 25/50/75/100% of the gap.
- `ablation_synth_study.py` — ablation of the amount of synthetic images added
  (25/50/75/100% of the gap) from a GAN checkpoint.
- `feature_analysis.py` — PCA, t-SNE and UMAP projections of the 512-dim feature space
  (real vs synthetic).
- `main_dann_synth.py` — synthetic -> real DANN pipeline: a control run (no alignment) vs
  a DANN run, with the resulting comparison.
- `main_dann_cross_hospital.py` — cross-hospital DANN (Kermany source, VinDr target):
  source-only baseline vs DANN.

### `utils/`
Helpers.

- `seed.py` — `set_seed` for reproducibility.
- `visualization.py` — `save_gan_samples`, a grid of generated images saved during GAN
  training.
- `logging.py` — Weights & Biases logging helpers (DANN comparison plots).
