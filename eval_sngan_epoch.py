import torch
import os
import shutil
import random
import argparse
import wandb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config import (
    DATASET_DIR, RESULTS_DIR, METRICS_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_LR,
    GAN_NZ, GAN_N_CLASS, GAN_NC, SEED
)
from dataset.loader import setup_dataset, get_dataloaders
from models.sngan import SNGenerator
from train import train_resnet
from eval import evaluate_on_test, generate_synthetic_images
from utils.seed import set_seed

SNGAN_D = 128
SNGAN_SYNTH_DIR = os.path.join(RESULTS_DIR, "sngan_synthetic_images_eval")
SNGAN_AUG_DIR = os.path.join(RESULTS_DIR, "sngan_augmented_dataset_eval")

class SimpleDataset(Dataset):
    def __init__(self, dir_path, transform=None, max_samples=500):
        self.files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
        random.shuffle(self.files)
        self.files = self.files[:max_samples] # Limita per velocizzare t-SNE
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def extract_features(loader, model, device):
    model.eval()
    features = []
    with torch.no_grad():
        for x in loader:
            feats = model(x.to(device)).cpu().numpy()
            features.append(feats)
    return np.concatenate(features, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Valuta una specifica epoca SNGAN e ne analizza il manifold")
    parser.add_argument('--epoch', type=int, default=220, help="Epoca del checkpoint Generator da caricare")
    parser.add_argument('--ckpt_dir', type=str, default="/content/drive/MyDrive/ProgettoMLVM/GAN_CHECKPOINTS_BACKUP", help="Path assoluto ai checkpoint su Drive")
    parser.add_argument('--drive_plot_dir', type=str, default="/content/drive/MyDrive/ProgettoMLVM/Eval_Results", help="Path assoluto su Drive dove salvare i plot")
    args = parser.parse_args()

    target_epoch = args.epoch

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Avvio Valutazione SNGAN (Epoca {target_epoch}) su Device: {device}")

    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name=f"SNGAN_Eval_Manifold_Ep{target_epoch}",
        config={"epoch_evaluated": target_epoch, "resnet_custom_epochs": 30}
    )

    # 1. SETUP DATASET
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res: return
    train_dir, val_dir, test_dir = res

    n_train_n = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    n_train_p = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    num_gen_normal = n_train_p - n_train_n
    num_gen_pneumonia = 0

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: BASELINE RESNET (30 Epoche)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  PHASE 1: Baseline ResNet (30 EPOCHE, solo dati reali)\n{'='*60}")
    resnet_model, _, ckpt_p1 = train_resnet(
        train_loader, val_loader, device,
        epochs=30, lr=RESNET_LR, tag="Phase1_Eval")
    report_p1, _ = evaluate_on_test(
        resnet_model, ckpt_p1, test_loader, classes, device,
        tag="Phase1_Eval", out_dir=METRICS_DIR)

    # ══════════════════════════════════════════════════════════════
    # PREPARAZIONE GENERATORE SNGAN
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  PHASE 2: Caricamento SNGAN Epoca {target_epoch}\n{'='*60}")
    
    G = SNGenerator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=SNGAN_D).to(device)
    # Cerca il file su Drive con il prefisso 'sngan'
    ckpt_path = os.path.join(args.ckpt_dir, f'G_sngan_epoch_{target_epoch}.pth')
    
    if not os.path.exists(ckpt_path):
        print(f"ERRORE: Impossibile trovare il checkpoint: {ckpt_path}")
        print("Assicurati di aver allenato la SNGAN fino a quell'epoca.")
        return

    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"  Checkpoint G_epoch_{target_epoch}.pth caricato con successo!")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: RESNET SU DATASET AUGMENTED (30 Epoche)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  PHASE 3: Generazione e Training Augmented (30 EPOCHE)\n{'='*60}")

    if os.path.exists(SNGAN_SYNTH_DIR): shutil.rmtree(SNGAN_SYNTH_DIR)
    if os.path.exists(SNGAN_AUG_DIR): shutil.rmtree(SNGAN_AUG_DIR)

    generate_synthetic_images(
        G, num_gen_normal, num_gen_pneumonia,
        nz=GAN_NZ, n_class=GAN_N_CLASS, device=device,
        syn_dir=SNGAN_SYNTH_DIR)

    aug_train_dir = os.path.join(SNGAN_AUG_DIR, 'train')
    shutil.copytree(train_dir, aug_train_dir)
    for cat in ['NORMAL', 'PNEUMONIA']:
        syn_cat = os.path.join(SNGAN_SYNTH_DIR, cat)
        if os.path.exists(syn_cat):
            for f in os.listdir(syn_cat):
                shutil.copy(os.path.join(syn_cat, f), os.path.join(aug_train_dir, cat, f))

    aug_loader, _, _, _ = get_dataloaders(
        aug_train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    resnet_aug, _, ckpt_p3 = train_resnet(
        aug_loader, val_loader, device,
        epochs=30, lr=RESNET_LR, tag=f"Phase3_Eval_ep{target_epoch}")
    report_p3, _ = evaluate_on_test(
        resnet_aug, ckpt_p3, test_loader, classes, device,
        tag=f"Phase3_Eval_ep{target_epoch}", out_dir=METRICS_DIR)

    # ══════════════════════════════════════════════════════════════
    # CONFRONTO FINALE
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  CONFRONTO FINALE SU 30 EPOCHE: Baseline vs SNGAN (Epoca {target_epoch})")
    print(f"{'='*60}")

    for cls in classes:
        for m in ['precision', 'recall', 'f1-score']:
            v1 = report_p1[cls][m]
            v3 = report_p3[cls][m]
            d  = v3 - v1
            print(f"  {cls} {m:12s}: {v1:.4f} → {v3:.4f}  ({'↑' if d > 0 else '↓'} {abs(d):.4f})")

    acc1 = report_p1['accuracy']
    acc3 = report_p3['accuracy']
    print(f"\n  Overall Acc: {acc1:.4f} → {acc3:.4f}  ({'↑' if acc3 > acc1 else '↓'} {abs(acc3 - acc1):.4f})")
    
    # ══════════════════════════════════════════════════════════════
    # MANIFOLD ANALYSIS (PCA & t-SNE)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  ANALISI MANIFOLD (PCA & t-SNE) su classe NORMAL\n{'='*60}")
    
    resnet_feat = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet_feat.fc = nn.Identity()
    resnet_feat = resnet_feat.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((RESNET_IMG_SIZE, RESNET_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    real_normal_dir = os.path.join(train_dir, 'NORMAL')
    synth_normal_dir = os.path.join(SNGAN_SYNTH_DIR, 'NORMAL')

    # Usiamo max 500 campioni per tipo per non far esplodere i tempi di calcolo della t-SNE
    ds_real = SimpleDataset(real_normal_dir, transform=transform, max_samples=500)
    ds_synth = SimpleDataset(synth_normal_dir, transform=transform, max_samples=500)

    loader_real = DataLoader(ds_real, batch_size=64, shuffle=False)
    loader_synth = DataLoader(ds_synth, batch_size=64, shuffle=False)

    print("  Estrazione feature Real NORMAL...")
    feat_real = extract_features(loader_real, resnet_feat, device)
    print("  Estrazione feature Synth NORMAL...")
    feat_synth = extract_features(loader_synth, resnet_feat, device)

    all_feats = np.concatenate([feat_real, feat_synth], axis=0)
    
    print("  Calcolo PCA...")
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(all_feats)

    print("  Calcolo t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)
    tsne_res = tsne.fit_transform(all_feats)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # PCA Plot
    axes[0].scatter(pca_res[:len(feat_real), 0], pca_res[:len(feat_real), 1], c='gray', label='Real NORMAL', alpha=0.6, s=15)
    axes[0].scatter(pca_res[len(feat_real):, 0], pca_res[len(feat_real):, 1], c='crimson', label='Synth NORMAL', alpha=0.6, s=15)
    axes[0].set_title('PCA Manifold (NORMAL)')
    axes[0].legend()

    # t-SNE Plot
    axes[1].scatter(tsne_res[:len(feat_real), 0], tsne_res[:len(feat_real), 1], c='gray', label='Real NORMAL', alpha=0.6, s=15)
    axes[1].scatter(tsne_res[len(feat_real):, 0], tsne_res[len(feat_real):, 1], c='crimson', label='Synth NORMAL', alpha=0.6, s=15)
    axes[1].set_title('t-SNE Manifold (NORMAL)')
    axes[1].legend()

    plot_path_local = os.path.join(RESULTS_DIR, f'sngan_manifold_ep{target_epoch}.png')
    
    # Crea cartella su Drive per i plot se non esiste
    os.makedirs(args.drive_plot_dir, exist_ok=True)
    plot_path_drive = os.path.join(args.drive_plot_dir, f'sngan_manifold_ep{target_epoch}.png')
    
    plt.savefig(plot_path_local, dpi=150)
    plt.savefig(plot_path_drive, dpi=150)
    plt.close(fig)

    wandb.log({"Manifold_Analysis": wandb.Image(plot_path_local)})
    print(f"  Analisi Manifold salvata in locale: {plot_path_local}")
    print(f"  Analisi Manifold salvata su Drive: {plot_path_drive}")
    
    wandb.finish()
    print("\n  Valutazione e Analisi Manifold completate.")

if __name__ == '__main__':
    main()
