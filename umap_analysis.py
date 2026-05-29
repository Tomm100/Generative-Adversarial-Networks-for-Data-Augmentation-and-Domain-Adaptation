import os
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import numpy as np
import wandb
from sklearn.metrics import silhouette_score

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
# IMPORTANTE: Inserire i path corretti generati durante i training
RESNET_WEIGHTS_PATH = os.path.join(RESULTS_DIR, "checkpoints", "best_model_Ablation_75pct.pth")
GAN_WEIGHTS_PATH = "/content/drive/MyDrive/ProgettoMLVM/results_SNGAN_pg_bg_128/sngan_checkpoints/G_epoch_XXX.pth" # AGGIORNA CON IL TUO PATH
GEN_D = 128
NUM_SYNTHETIC = 300 # Numero di immagini sintetiche da generare per l'analisi

def load_resnet_extractor(device):
    """Carica la ResNet e crea un estrattore rimuovendo l'ultimo layer FC."""
    print(f"Caricamento ResNet da: {RESNET_WEIGHTS_PATH}")
    model = ResNetClassifier(num_classes=2)
    if not os.path.exists(RESNET_WEIGHTS_PATH):
        print(f"ATTENZIONE: Modello {RESNET_WEIGHTS_PATH} non trovato. Verrà usata la rete pre-addestrata (non fine-tuned).")
    else:
        model.load_state_dict(torch.load(RESNET_WEIGHTS_PATH, map_location=device))
    
    # Rimuoviamo l'ultimo fully-connected per estrarre le feature di dimensione 512
    # Il backbone di ResNetClassifier è resnet18
    extractor = nn.Sequential(*list(model.backbone.children())[:-1])
    extractor = extractor.to(device)
    extractor.eval()
    return extractor

def get_real_features(extractor, loader, device, max_samples=None):
    """Estrae le feature 512D dalle immagini reali del loader."""
    features = []
    labels = []
    extracted = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # x shape: [B, 3, H, W]
            feat = extractor(x) # shape: [B, 512, 1, 1]
            feat = torch.flatten(feat, 1) # shape: [B, 512]
            
            features.append(feat.cpu().numpy())
            labels.append(y.cpu().numpy())
            
            extracted += x.size(0)
            if max_samples and extracted >= max_samples:
                break
                
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    if max_samples:
        features = features[:max_samples]
        labels = labels[:max_samples]
        
    return features, labels

def get_synthetic_features(extractor, generator, device, num_samples):
    """Genera immagini sintetiche (Normal) on-the-fly, le processa e ne estrae le feature."""
    features = []
    labels = []
    generated = 0
    
    # Transform per mappare l'output della GAN all'input atteso da ResNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    resize = transforms.Resize((RESNET_IMG_SIZE, RESNET_IMG_SIZE))
    
    generator.eval()
    onehot = torch.eye(GAN_N_CLASS).view(GAN_N_CLASS, GAN_N_CLASS, 1, 1).to(device)
    # Assumiamo NORMAL = classe 0
    cls_idx = 0 
    
    with torch.no_grad():
        while generated < num_samples:
            batch_sz = min(RESNET_BATCH_SIZE, num_samples - generated)
            z = torch.randn(batch_sz, GAN_NZ, 1, 1).to(device)
            y_gen = onehot[torch.full((batch_sz,), cls_idx, dtype=torch.long).to(device)]
            
            fakes = generator(z, y_gen) # shape: [B, 1, 128, 128], range [-1, 1]
            
            # Da [-1, 1] a [0, 1]
            fakes = (fakes + 1) / 2.0 
            
            # ResNet aspetta 3 canali. La GAN genera in scala di grigi (1 canale)
            fakes_rgb = fakes.repeat(1, 3, 1, 1)
            
            # Resize e Normalize
            fakes_resnet = resize(fakes_rgb)
            fakes_resnet = normalize(fakes_resnet)
            
            # Estrazione feature
            feat = extractor(fakes_resnet)
            feat = torch.flatten(feat, 1)
            
            features.append(feat.cpu().numpy())
            # Label = 2 (Synthetic Normal) per distinguerla dal Test Set
            labels.append(np.full(batch_sz, 2))
            
            generated += batch_sz
            
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return features, labels

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione UMAP su: {device}")
    
    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name="UMAP_Analysis_ResNet",
        config={
            "num_synthetic": NUM_SYNTHETIC,
            "resnet_weights": RESNET_WEIGHTS_PATH,
            "gan_weights": GAN_WEIGHTS_PATH,
        }
    )
    
    # 1. Carica Test Set Reale
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        print("Dataset non trovato!")
        return
    train_dir, val_dir, test_dir = res
    
    _, _, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir, 
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
    )
    
    # 2. Carica i Modelli
    extractor = load_resnet_extractor(device)
    
    print(f"Caricamento Generator da: {GAN_WEIGHTS_PATH}")
    generator = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GEN_D).to(device)
    if not os.path.exists(GAN_WEIGHTS_PATH):
         print(f"ERRORE: Modello GAN {GAN_WEIGHTS_PATH} non trovato. Verifica il percorso.")
         return
    generator.load_state_dict(torch.load(GAN_WEIGHTS_PATH, map_location=device))
    
    # 3. Estrazione Feature
    print("Estrazione feature dalle immagini reali del Test Set...")
    real_features, real_labels = get_real_features(extractor, test_loader, device)
    
    print(f"Generazione {NUM_SYNTHETIC} immagini sintetiche ed estrazione feature...")
    syn_features, syn_labels = get_synthetic_features(extractor, generator, device, num_samples=NUM_SYNTHETIC)
    
    # Uniamo tutte le feature
    X = np.vstack([real_features, syn_features])
    y = np.concatenate([real_labels, syn_labels])
    
    # Nomi delle categorie:
    # 0 -> Real Normal
    # 1 -> Real Pneumonia
    # 2 -> Synthetic Normal
    target_names = ['Real Normal', 'Real Pneumonia', 'Synthetic Normal']
    
    # 4. Proiezione UMAP
    print("Esecuzione UMAP (da 512D a 2D)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=SEED)
    X_umap = reducer.fit_transform(X)
    
    # 5. Plotting
    print("Creazione grafico UMAP...")
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    for i, name in enumerate(target_names):
        idx = (y == i)
        plt.scatter(
            X_umap[idx, 0], X_umap[idx, 1], 
            label=name, color=colors[i], marker=markers[i], 
            alpha=0.7, edgecolors='w', s=50
        )
        
    plt.title("UMAP Projection of ResNet-18 Features", fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Data Type")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_dir = os.path.join(RESULTS_DIR, "umap_analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "umap_resnet_features.png")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"✅ Grafico salvato in: {out_path}")
    
    # 6. Metriche
    print("\nCalcolo delle metriche di clustering (Silhouette Score)...")
    # Silhouette globale (su tutte e 3 le classi) nello spazio originale (512D)
    score_512d = silhouette_score(X, y, metric='cosine')
    # Silhouette nello spazio UMAP 2D
    score_umap = silhouette_score(X_umap, y, metric='euclidean')
    
    print(f"  Silhouette Score (512D): {score_512d:.4f}")
    print(f"  Silhouette Score (UMAP 2D): {score_umap:.4f}")
    
    # 7. Salvataggio su WandB
    wandb.log({
        "UMAP/Silhouette_Score_512D": score_512d,
        "UMAP/Silhouette_Score_2D": score_umap,
        "UMAP/Projection": wandb.Image(out_path, caption="UMAP Projection")
    })
    wandb.finish()

if __name__ == "__main__":
    main()
