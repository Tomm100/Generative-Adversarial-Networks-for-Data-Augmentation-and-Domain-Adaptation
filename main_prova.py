"""
main_prova.py — Esperimento finale: Data Augmentation classica + JPEG + Grad-CAM.

Riusa i pesi GAN già salvati (senza riaddestrare il GAN).
Per ogni checkpoint GAN disponibile (ep. 10, 20, ..., 100):
  1. Rigenera le sintetiche come JPEG
  2. Allena ResNet con data augmentation classica + ImageNet norm
  3. Valuta e salva il miglior risultato
Alla fine, produce Grad-CAM per analizzare dove guarda la ResNet.

Uso:
    python main_prova.py
"""

import torch
import torch.nn.functional as F
import os
import shutil
import numpy as np
import wandb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

from config import (
    DATASET_DIR, RESULTS_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_EPOCHS, RESNET_LR,
    GAN_NZ, GAN_N_CLASS, GAN_NC, GAN_D,
    SEED,
)
from dataset.loader import setup_dataset
from models.wgan import Generator
from models.resnet import ResNetClassifier
from eval import evaluate_on_test, generate_synthetic_images, plot_comparison
from train import train_resnet
from utils.seed import set_seed


# ─── CONFIG ESPERIMENTO ──────────────────────────────────────
# Path dei pesi GAN dal run precedente (su Google Drive montato in Colab)
PREV_GAN_CHECKPOINTS_DIR = "/content/drive/MyDrive/ProgettoMLVM/results_with_norm/gan_checkpoints"

# Directory dedicate per questo esperimento
EXP_DIR = os.path.join(RESULTS_DIR, "experiment_augmentation")
EXP_SYNTHETIC_DIR = os.path.join(EXP_DIR, "synthetic_images")
EXP_AUGMENTED_DIR = os.path.join(EXP_DIR, "augmented_dataset")
EXP_METRICS_DIR = os.path.join(EXP_DIR, "metrics")
EXP_CHECKPOINTS_DIR = os.path.join(EXP_DIR, "checkpoints")
EXP_GRADCAM_DIR = os.path.join(EXP_DIR, "gradcam")

# Quante epoche ResNet per la validazione veloce su ogni checkpoint GAN
VAL_RESNET_EPOCHS = 5


# ─── DATALOADERS CON DATA AUGMENTATION CLASSICA ────────────────
def get_augmented_dataloaders(train_dir, val_dir, test_dir, img_size=128, batch_size=16):
    """
    DataLoader con data augmentation classica sul TRAINING set.
    Val/Test usano solo Resize + Normalize (nessuna augmentation).

    L'augmentation classica (flip, rotation, crop) "sporca" le texture
    impedendo alla ResNet di fare shortcut learning sugli artefatti GAN.
    """
    # ImageNet normalization stats
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes


# ─── GRAD-CAM ──────────────────────────────────────────────────
class GradCAM:
    """Grad-CAM per ResNet18. Hookka l'ultimo layer convoluzionale (layer4)."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Per ResNet18, l'ultimo blocco conv è backbone.layer4
        if target_layer is None:
            target_layer = model.backbone.layer4
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx=None):
        """Genera la heatmap Grad-CAM per un singolo input."""
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Pesi: media globale dei gradienti su H,W
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        # Combinazione pesata delle attivazioni
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Solo attivazioni positive
        # Upsample alla dimensione dell'immagine
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        # Normalizza [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.squeeze().cpu().numpy(), class_idx, output


def generate_gradcam_grid(model, test_loader, classes, device, out_dir, tag, num_samples=8):
    """
    Genera una griglia Grad-CAM su campioni dal test set.
    Seleziona campioni da entrambe le classi.
    """
    os.makedirs(out_dir, exist_ok=True)
    gradcam = GradCAM(model)

    # Raccogli campioni bilanciati per classe
    samples_per_class = {0: [], 1: []}
    for x, y in test_loader:
        for i in range(x.size(0)):
            cls = y[i].item()
            if len(samples_per_class[cls]) < num_samples:
                samples_per_class[cls].append(x[i])
        if all(len(v) >= num_samples for v in samples_per_class.values()):
            break

    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 2.5, 10))

    # ImageNet denorm per visualizzazione
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for cls_idx in [0, 1]:
        for i, img_tensor in enumerate(samples_per_class[cls_idx][:num_samples]):
            inp = img_tensor.unsqueeze(0).to(device)
            cam, pred_cls, logits = gradcam.generate(inp)
            probs = F.softmax(logits, dim=1).detach().cpu()

            # Denormalizza per visualizzazione
            img_vis = img_tensor * std + mean
            img_vis = img_vis.clamp(0, 1).permute(1, 2, 0).numpy()

            row_img = cls_idx * 2       # Riga immagine originale
            row_cam = cls_idx * 2 + 1   # Riga Grad-CAM overlay

            # Immagine originale
            axes[row_img, i].imshow(img_vis)
            axes[row_img, i].set_title(
                f"True: {classes[cls_idx]}\nPred: {classes[pred_cls]}",
                fontsize=7, color='green' if pred_cls == cls_idx else 'red')
            axes[row_img, i].axis('off')

            # Grad-CAM overlay
            axes[row_cam, i].imshow(img_vis)
            axes[row_cam, i].imshow(cam, alpha=0.5, cmap='jet')
            conf = probs[0, pred_cls].item()
            axes[row_cam, i].set_title(f"Conf: {conf:.2f}", fontsize=7)
            axes[row_cam, i].axis('off')

    plt.suptitle(f'Grad-CAM — {tag}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(out_dir, f'gradcam_{tag}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Grad-CAM salvato: {save_path}")
    return save_path


# ─── FUNZIONI DI SUPPORTO ─────────────────────────────────────
def get_all_gan_epochs(models_dir):
    """Trova tutte le epoche con checkpoint GAN disponibili."""
    if not os.path.exists(models_dir):
        return []
    checkpoints = [f for f in os.listdir(models_dir)
                   if f.startswith('G_epoch_') and f.endswith('.pth')]
    epochs = sorted([int(f.replace('G_epoch_', '').replace('.pth', ''))
                     for f in checkpoints])
    return epochs


def create_augmented_dataset(G, train_dir, aug_train_dir, syn_dir,
                              num_gen_normal, num_gen_pneumonia,
                              device, nz, n_class):
    """Genera sintetiche e crea il dataset augmented."""
    if os.path.exists(syn_dir):
        shutil.rmtree(syn_dir)
    if os.path.exists(os.path.dirname(aug_train_dir)):
        shutil.rmtree(os.path.dirname(aug_train_dir))

    generate_synthetic_images(
        G, num_gen_normal, num_gen_pneumonia,
        nz=nz, n_class=n_class, device=device, syn_dir=syn_dir)

    shutil.copytree(train_dir, aug_train_dir)
    for cat in ['NORMAL', 'PNEUMONIA']:
        syn_cat = os.path.join(syn_dir, cat)
        if os.path.exists(syn_cat):
            for f in os.listdir(syn_cat):
                shutil.copy(os.path.join(syn_cat, f),
                            os.path.join(aug_train_dir, cat, f))


# ─── MAIN ─────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  ESPERIMENTO: Data Aug classica + JPEG + ImageNet norm")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # --- W&B ---
    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name="experiment_augmentation_gradcam",
        config={
            "experiment": "classical_augmentation_jpeg_gradcam",
            "seed": SEED,
            "resnet_img_size": RESNET_IMG_SIZE,
            "resnet_batch_size": RESNET_BATCH_SIZE,
            "resnet_epochs": RESNET_EPOCHS,
            "resnet_lr": RESNET_LR,
            "val_resnet_epochs": VAL_RESNET_EPOCHS,
            "gan_d": GAN_D,
            "normalization": "imagenet",
            "synthetic_format": "jpeg",
            "data_augmentation": "flip+rotation+resized_crop",
        }
    )

    # --- 1. SETUP DATASET ---
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        return
    train_dir, val_dir, test_dir = res

    n_train_n = len([f for f in os.listdir(os.path.join(train_dir, 'NORMAL'))
                     if not f.startswith('.')])
    n_train_p = len([f for f in os.listdir(os.path.join(train_dir, 'PNEUMONIA'))
                     if not f.startswith('.')])
    print(f"  Train: {n_train_n} NORMAL + {n_train_p} PNEUMONIA")

    num_gen_normal = n_train_p - n_train_n
    num_gen_pneumonia = 0

    # --- 2. EVAL DATALOADERS (senza augmentation, per test/val) ---
    from dataset.loader import get_dataloaders
    _, val_loader, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    # --- 3. TRAIN DATALOADER BASELINE (con augmentation classica) ---
    train_loader_aug, _, _, _ = get_augmented_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(EXP_METRICS_DIR, exist_ok=True)
    os.makedirs(EXP_GRADCAM_DIR, exist_ok=True)

    # Sovrascrivi CHECKPOINTS_DIR per non sporcare
    import config
    original_checkpoints = config.CHECKPOINTS_DIR
    config.CHECKPOINTS_DIR = EXP_CHECKPOINTS_DIR

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: BASELINE (dati reali + data aug classica + ImageNet norm)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Phase 1: Baseline ResNet (ImageNet norm + Data Aug)")
    print(f"{'='*60}")

    model_p1, hist_p1, ckpt_p1 = train_resnet(
        train_loader_aug, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Exp_Phase1")

    report_p1, cm_p1 = evaluate_on_test(
        model_p1, ckpt_p1, test_loader, classes, device,
        tag="Exp_Phase1", out_dir=EXP_METRICS_DIR)

    # Grad-CAM sulla baseline
    gradcam_p1 = generate_gradcam_grid(
        model_p1, test_loader, classes, device,
        out_dir=EXP_GRADCAM_DIR, tag="Baseline")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: SWEEP SU TUTTI I CHECKPOINT GAN
    # Per ogni epoca: genera sintetiche → allena ResNet → valida
    # ═══════════════════════════════════════════════════════════
    gan_epochs = get_all_gan_epochs(PREV_GAN_CHECKPOINTS_DIR)
    if not gan_epochs:
        print(f"\n❌ Nessun checkpoint GAN trovato in {PREV_GAN_CHECKPOINTS_DIR}")
        wandb.finish()
        return

    print(f"\n{'='*60}")
    print(f"  Phase 2: Sweep su {len(gan_epochs)} checkpoint GAN")
    print(f"  Epoche disponibili: {gan_epochs}")
    print(f"  ResNet validation: {VAL_RESNET_EPOCHS} epoche per checkpoint")
    print(f"{'='*60}")

    val_results = []
    best_val_f1 = -1.0
    best_val_epoch = 0

    for gan_ep in gan_epochs:
        print(f"\n  {'─'*50}")
        print(f"  [GAN Ep.{gan_ep}] Caricamento e generazione...")
        print(f"  {'─'*50}")

        gan_ckpt = os.path.join(PREV_GAN_CHECKPOINTS_DIR, f'G_epoch_{gan_ep}.pth')
        G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GAN_D).to(device)
        G.load_state_dict(torch.load(gan_ckpt, map_location=device))
        G.eval()

        # Genera sintetiche e crea dataset augmented
        aug_train_dir = os.path.join(EXP_AUGMENTED_DIR, 'train')
        create_augmented_dataset(
            G, train_dir, aug_train_dir, EXP_SYNTHETIC_DIR,
            num_gen_normal, num_gen_pneumonia,
            device, GAN_NZ, GAN_N_CLASS)

        # DataLoader augmented CON data augmentation classica
        aug_loader, _, _, _ = get_augmented_dataloaders(
            aug_train_dir, val_dir, test_dir,
            img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

        # Allena ResNet (validazione rapida)
        _, hist, _ = train_resnet(
            aug_loader, val_loader, device,
            epochs=VAL_RESNET_EPOCHS, lr=RESNET_LR,
            tag=f"AugSweep_ep{gan_ep}")

        best_f1 = max(hist['val_macro_f1'])
        best_acc = hist['val_acc'][hist['val_macro_f1'].index(best_f1)]

        val_results.append({
            'epoch': gan_ep, 'macro_f1': best_f1, 'accuracy': best_acc
        })

        wandb.log({
            "Sweep/macro_f1": best_f1,
            "Sweep/accuracy": best_acc,
            "Sweep/gan_epoch": gan_ep
        })

        print(f"  [GAN Ep.{gan_ep}] Macro F1: {best_f1:.4f} | Acc: {best_acc:.2f}%")

        if best_f1 > best_val_f1:
            best_val_f1 = best_f1
            best_val_epoch = gan_ep
            print(f"  ★ Nuovo miglior F1!")

    # --- Sweep Summary ---
    print(f"\n{'='*60}")
    print(f"  Sweep Summary (Data Aug + JPEG + ImageNet norm)")
    print(f"{'='*60}")
    for r in val_results:
        marker = " ← BEST" if r['epoch'] == best_val_epoch else ""
        print(f"  Ep.{r['epoch']:3d} | Macro F1: {r['macro_f1']:.4f} | "
              f"Acc: {r['accuracy']:.2f}%{marker}")
    print(f"\n  Miglior epoca GAN: {best_val_epoch} (F1: {best_val_f1:.4f})")

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: TRAINING COMPLETO CON LA MIGLIORE EPOCA GAN
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Phase 3: Training completo (GAN epoca {best_val_epoch})")
    print(f"{'='*60}")

    # Rigenera con la migliore
    gan_ckpt_best = os.path.join(PREV_GAN_CHECKPOINTS_DIR,
                                  f'G_epoch_{best_val_epoch}.pth')
    G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GAN_D).to(device)
    G.load_state_dict(torch.load(gan_ckpt_best, map_location=device))
    G.eval()

    aug_train_dir = os.path.join(EXP_AUGMENTED_DIR, 'train')
    create_augmented_dataset(
        G, train_dir, aug_train_dir, EXP_SYNTHETIC_DIR,
        num_gen_normal, num_gen_pneumonia,
        device, GAN_NZ, GAN_N_CLASS)

    aug_loader_final, _, _, _ = get_augmented_dataloaders(
        aug_train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    model_p3, hist_p3, ckpt_p3 = train_resnet(
        aug_loader_final, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Exp_Phase3")

    report_p3, cm_p3 = evaluate_on_test(
        model_p3, ckpt_p3, test_loader, classes, device,
        tag="Exp_Phase3", out_dir=EXP_METRICS_DIR)

    # Grad-CAM sul modello augmented
    gradcam_p3 = generate_gradcam_grid(
        model_p3, test_loader, classes, device,
        out_dir=EXP_GRADCAM_DIR, tag="Augmented")

    # ═══════════════════════════════════════════════════════════
    # CONFRONTO FINALE + Log Grad-CAM
    # ═══════════════════════════════════════════════════════════
    plot_comparison(hist_p1, hist_p3, cm_p1, cm_p3, classes,
                    report_p1, report_p3, out_dir=EXP_METRICS_DIR)

    # Log Grad-CAM su W&B
    if os.path.exists(gradcam_p1):
        wandb.log({"GradCAM/Baseline": wandb.Image(gradcam_p1)})
    if os.path.exists(gradcam_p3):
        wandb.log({"GradCAM/Augmented": wandb.Image(gradcam_p3)})

    # Ripristina config
    config.CHECKPOINTS_DIR = original_checkpoints

    print(f"\n📊 Risultati salvati in: {EXP_DIR}")
    print(f"🔬 Grad-CAM salvate in: {EXP_GRADCAM_DIR}")
    print("✅ Esperimento completato!")

    wandb.finish()


if __name__ == '__main__':
    main()
