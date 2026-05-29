import os
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import numpy as np
import wandb

from models.resnet import ResNetClassifier
from models.wgan import Generator
from dataset.loader import setup_dataset, get_dataloaders
from utils.seed import set_seed
from config import (
    DATASET_DIR, RESULTS_DIR, GAN_NZ, GAN_N_CLASS, GAN_NC, SEED,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE
)

# ==============================================================================
# CONFIGURAZIONI UMAP E MODELLI
# ==============================================================================
# AGGIORNA CON I TUOI PATH REALI
RESNET_WEIGHTS_PATH = "/content/drive/Mydrive/ProgettoMLVM/results_SNGAN_pg_bg_128/ResnetCheckpoint/best_model_Ablation_75pct.pth"
GAN_WEIGHTS_PATH = "/content/drive/MyDrive/ProgettoMLVM/results_SNGAN_pg_bg_128/sngan_checkpoints/G_epoch_220.pth"

GEN_D = 128
NUM_SAMPLES_PER_CLASS = 500  # 500 Real Normal, 500 Real Pneumonia, 500 Fake Normal

def load_resnet_extractor(device):
    """Carica la ResNet e crea un estrattore sostituendo l'ultimo strato con Identity."""
    print(f"\n[1/5] Caricamento ResNet da: {RESNET_WEIGHTS_PATH}")
    model = ResNetClassifier(num_classes=2)
    
    if not os.path.exists(RESNET_WEIGHTS_PATH):
        print(f"⚠️ ATTENZIONE: Modello ResNet non trovato. Verranno usati i pesi pre-addestrati (baseline).")
    else:
        model.load_state_dict(torch.load(RESNET_WEIGHTS_PATH, map_location=device))
        print("✅ Pesi ResNet caricati correttamente.")
    
    # Bypassiamo il classificatore finale: l'output sarà direttamente il vettore [B, 512]
    model.backbone.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    return model

def get_balanced_real_features(extractor, loader, device, num_per_class):
    """Estrae esattamente `num_per_class` feature per NORMAL (0) e PNEUMONIA (1) dal loader."""
    features_normal, features_pneumonia = [], []
    count_normal, count_pneumonia = 0, 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = extractor(x).cpu().numpy()  # shape: [B, 512]
            labels = y.numpy()
            
            for i in range(len(labels)):
                if labels[i] == 0 and count_normal < num_per_class:  # NORMAL
                    features_normal.append(feats[i])
                    count_normal += 1
                elif labels[i] == 1 and count_pneumonia < num_per_class:  # PNEUMONIA
                    features_pneumonia.append(feats[i])
                    count_pneumonia += 1
                    
            if count_normal >= num_per_class and count_pneumonia >= num_per_class:
                break
                
    feat_n = np.array(features_normal)
    feat_p = np.array(features_pneumonia)
    
    # Creiamo i label: 0 per Real Normal, 1 per Real Pneumonia
    labels_n = np.zeros(len(feat_n))
    labels_p = np.ones(len(feat_p))
    
    features = np.vstack([feat_n, feat_p])
    labels = np.concatenate([labels_n, labels_p])
    
    print(f"✅ Estratte {len(feat_n)} Real NORMAL e {len(feat_p)} Real PNEUMONIA.")
    return features, labels

def get_synthetic_features(extractor, generator, device, num_samples):
    """Genera immagini sintetiche (Normal) on-the-fly, le processa e ne estrae le feature."""
    features = []
    generated = 0
    
    # Transform per mappare l'output della GAN all'input atteso da ResNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = transforms.Resize((RESNET_IMG_SIZE, RESNET_IMG_SIZE))
    
    generator.eval()
    onehot = torch.eye(GAN_N_CLASS).view(GAN_N_CLASS, GAN_N_CLASS, 1, 1).to(device)
    cls_idx = 0  # Assumiamo NORMAL = classe 0
    
    with torch.no_grad():
        while generated < num_samples:
            batch_sz = min(RESNET_BATCH_SIZE, num_samples - generated)
            z = torch.randn(batch_sz, GAN_NZ, 1, 1).to(device)
            y_gen = onehot[torch.full((batch_sz,), cls_idx, dtype=torch.long).to(device)]
            
            fakes = generator(z, y_gen)  # shape: [B, 1, 128, 128], range [-1, 1]
            
            # 1. Da [-1, 1] a [0, 1]
            fakes = (fakes + 1.0) / 2.0 
            
            # 2. Da 1 canale (Grayscale) a 3 canali (RGB) per la ResNet
            fakes_rgb = fakes.repeat(1, 3, 1, 1)
            
            # 3. Resize e Normalize (ImageNet)
            fakes_resnet = resize(fakes_rgb)
            fakes_resnet = normalize(fakes_resnet)
            
            # 4. Estrazione feature
            feat = extractor(fakes_resnet).cpu().numpy()  # shape: [B, 512]
            features.append(feat)
            
            generated += batch_sz
            
    features = np.concatenate(features, axis=0)
    labels = np.full(num_samples, 2)  # Label = 2 per Fake NORMAL
    
    print(f"✅ Generate ed estratte feature da {generated} Fake NORMAL.")
    return features, labels

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\n  UMAP FEATURE SPACE ANALYSIS (TRAINING MANIFOLD)\n{'='*60}")
    print(f"Device: {device}")
    
    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name="UMAP_Analysis_75pct",
        config={
            "num_samples_per_class": NUM_SAMPLES_PER_CLASS,
            "resnet_weights": RESNET_WEIGHTS_PATH,
            "gan_weights": GAN_WEIGHTS_PATH,
        }
    )
    
    # --- 1. SETUP DATASET (TRAIN SET) ---
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        return
    train_dir, val_dir, test_dir = res
    
    # Prendiamo il train_loader (fondamentale per vedere i dati su cui ha studiato)
    train_loader, _, _, classes = get_dataloaders(
        train_dir, val_dir, test_dir, 
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
    )
    
    # --- 2. CARICAMENTO MODELLI ---
    extractor = load_resnet_extractor(device)
    
    print(f"\n[2/5] Caricamento Generator da: {GAN_WEIGHTS_PATH}")
    generator = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GEN_D).to(device)
    if not os.path.exists(GAN_WEIGHTS_PATH):
         print(f"❌ ERRORE: Modello GAN non trovato. Verifica il percorso.")
         return
    generator.load_state_dict(torch.load(GAN_WEIGHTS_PATH, map_location=device))
    print("✅ Pesi SNGAN caricati correttamente.")
    
    # --- 3. ESTRAZIONE FEATURE ---
    print("\n[3/5] Estrazione feature dalle immagini REALI del TRAINING Set...")
    real_features, real_labels = get_balanced_real_features(
        extractor, train_loader, device, num_per_class=NUM_SAMPLES_PER_CLASS
    )
    
    print(f"\n[4/5] Generazione on-the-fly e feature extraction per le immagini SINTETICHE...")
    syn_features, syn_labels = get_synthetic_features(
        extractor, generator, device, num_samples=NUM_SAMPLES_PER_CLASS
    )
    
    # Uniamo tutte le feature: 1500 campioni totali a 512 dimensioni
    X = np.vstack([real_features, syn_features])
    y = np.concatenate([real_labels, syn_labels])
    
    target_names = ['Real Normal', 'Real Pneumonia', 'Fake Normal']
    
    # --- 4. PROIEZIONE UMAP ---
    print("\n[5/5] Esecuzione UMAP (riduzione da 512D a 2D)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=SEED)
    X_umap = reducer.fit_transform(X)
    
    # --- 5. VISUALIZZAZIONE ---
    print("Creazione scatter plot...")
    plt.figure(figsize=(12, 9))
    
    # Palette e z-order (mettiamo le Fake Normal con zorder più alto per farle risaltare sopra le altre)
    colors = ['#1f77b4', '#d62728', '#2ca02c']  # Blu, Rosso, Verde acceso
    markers = ['o', 's', '^']
    alphas = [0.5, 0.4, 0.9]
    zorders = [1, 1, 3] 
    
    for i, name in enumerate(target_names):
        idx = (y == i)
        plt.scatter(
            X_umap[idx, 0], X_umap[idx, 1], 
            label=name, color=colors[i], marker=markers[i], 
            alpha=alphas[i], edgecolors='white' if i==2 else 'none', 
            s=60 if i==2 else 40, zorder=zorders[i]
        )
        
    plt.title("UMAP Projection of ResNet-18 Features (Train Manifold)", fontsize=16, fontweight='bold')
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(title="Data Type", fontsize=11, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    out_dir = os.path.join(RESULTS_DIR, "umap_analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "umap_resnet_features.png")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n🎉 GRAFICO COMPLETATO! Salvato in: {out_path}")
    
    # --- 6. LOG WANDB ---
    wandb.log({
        "UMAP/Train_Manifold_Projection": wandb.Image(out_path, caption="UMAP Projection")
    })
    wandb.finish()

if __name__ == "__main__":
    main()