import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

# Import dal progetto esistente
from config import DATASET_DIR, GAN_NZ, GAN_N_CLASS, GAN_NC, GAN_D, SEED, RESNET_IMG_SIZE
from models.wgan import Generator
from dataset.loader import setup_dataset
from utils.seed import set_seed

# ==============================================================================
# ⚙️ IMPOSTAZIONI MANUALI
# ==============================================================================
CUSTOM_GAN_WEIGHTS_PATH = "/content/drive/MyDrive/ProgettoMLVM/results_BAGAN/gan_checkpoints/G_epoch_200.pth"
TARGET_CLASS = "NORMAL"  # La classe su cui valutare la diversity (solitamente la minoritaria)
NUM_SAMPLES = 100        # Quante immagini reali vs fake confrontare
RESULTS_DIR = "./results/feature_diversity"
# ==============================================================================

def get_feature_extractor(device):
    """
    Carica una ResNet18 pre-addestrata su ImageNet e rimuove l'ultimo layer (fc).
    Restituisce un modello che mappa un'immagine [3, 128, 128] -> feature vector [512].
    """
    print("\n[Setup] Caricamento ResNet18 Feature Extractor...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Sostituisci l'ultimo layer con un Identity layer per ottenere le feature da 512
    resnet.fc = nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()  # Assolutamente fondamentale: no dropout, stat batchnorm bloccate
    return resnet


def prepare_fake_images(G, device, out_dir, num_samples, target_class_idx):
    """
    Genera `num_samples` immagini della classe target e le salva fisicamente su disco
    per poi caricarle con la stessa identica normalizzazione ImageNet delle immagini reali.
    """
    print(f"\n[Generazione] Creazione di {num_samples} immagini sintetiche per la classe {TARGET_CLASS}...")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    class_dir = os.path.join(out_dir, TARGET_CLASS)
    os.makedirs(class_dir, exist_ok=True)
    
    G.eval()
    batch_size = 32
    generated = 0
    
    with torch.no_grad():
        while generated < num_samples:
            bs = min(batch_size, num_samples - generated)
            z = torch.randn(bs, GAN_NZ, 1, 1, device=device)
            labels = torch.full((bs,), target_class_idx, dtype=torch.long, device=device)
            labels_onehot = F.one_hot(labels, GAN_N_CLASS).view(bs, GAN_N_CLASS, 1, 1).float()
            
            # Genera fakes in range [-1, 1]
            fakes = G(z, labels_onehot)
            # Denormalizza a [0, 1]
            fakes = (fakes + 1.0) / 2.0
            
            # Salva ogni immagine
            for i in range(bs):
                img_tensor = fakes[i].cpu().squeeze()  # [1, H, W] -> [H, W]
                img_np = (img_tensor.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_np, mode='L')
                img.save(os.path.join(class_dir, f"fake_{generated + i}.png"))
            
            generated += bs
            
    print(f"✅ Generate {generated} immagini in {class_dir}")


def extract_features(model, dataloader, device, desc="Estraggo feature"):
    """
    Passa le immagini attraverso la ResNet e restituisce un array numpy [N, 512].
    """
    features = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc=desc):
            x = x.to(device)
            # ResNet estrae un tensore [Batch, 512]
            feats = model(x)
            features.append(feats.cpu().numpy())
            
    return np.concatenate(features, axis=0)


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # --- 1. SETUP DATASET REALE ---
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        return
    train_dir, _, _ = res
    
    # Troviamo l'indice della classe (solitamente NORMAL=0, PNEUMONIA=1)
    dataset_temp = datasets.ImageFolder(root=train_dir)
    classes = dataset_temp.classes
    if TARGET_CLASS not in classes:
        print(f"❌ Errore: Classe {TARGET_CLASS} non trovata.")
        return
    target_class_idx = classes.index(TARGET_CLASS)
    
    # --- 2. TRASFORMAZIONI CONDIVISE (IMAGENET) ---
    # Questa è la chiave: sia real che fake devono passare per questa normalizzazione
    imagenet_transform = transforms.Compose([
        transforms.Resize((RESNET_IMG_SIZE, RESNET_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # --- 3. CAMPIONAMENTO IMMAGINI REALI ---
    print(f"\n[Campionamento] Selezione di {NUM_SAMPLES} immagini REALI della classe {TARGET_CLASS}...")
    real_dataset = datasets.ImageFolder(root=train_dir, transform=imagenet_transform)
    
    # Trova tutti gli indici delle immagini della classe target
    target_indices = [i for i, (_, label) in enumerate(real_dataset.samples) if label == target_class_idx]
    
    if len(target_indices) < NUM_SAMPLES:
        print(f"⚠️ Attenzione: Trovate solo {len(target_indices)} immagini reali (richieste {NUM_SAMPLES}).")
        num_real_to_sample = len(target_indices)
    else:
        num_real_to_sample = NUM_SAMPLES
        
    # Prendi randomicamente N indici
    sampled_real_indices = np.random.choice(target_indices, num_real_to_sample, replace=False)
    real_subset = Subset(real_dataset, sampled_real_indices)
    real_loader = DataLoader(real_subset, batch_size=32, shuffle=False)
    
    # --- 4. GENERAZIONE IMMAGINI FAKE ---
    G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GAN_D).to(device)
    if not os.path.exists(CUSTOM_GAN_WEIGHTS_PATH):
        print(f"❌ Errore: Pesi GAN non trovati in {CUSTOM_GAN_WEIGHTS_PATH}")
        return
    G.load_state_dict(torch.load(CUSTOM_GAN_WEIGHTS_PATH, map_location=device))
    
    fake_dir = os.path.join(RESULTS_DIR, "fake_samples")
    prepare_fake_images(G, device, fake_dir, NUM_SAMPLES, target_class_idx)
    
    fake_dataset = datasets.ImageFolder(root=fake_dir, transform=imagenet_transform)
    fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=False)
    
    # --- 5. ESTRAZIONE DELLE FEATURES ---
    extractor = get_feature_extractor(device)
    
    real_features = extract_features(extractor, real_loader, device, desc="Estraendo Real")
    fake_features = extract_features(extractor, fake_loader, device, desc="Estraendo Fake")
    
    print(f"\n✅ Feature estratte: Real {real_features.shape}, Fake {fake_features.shape}")
    
    # --- 6. CALCOLO METRICHE (Pairwise Cosine Distance & Traccia Covarianza) ---
    print("\n[Analisi] Calcolo metriche di Feature Diversity...")
    
    # Cosine distances (più è vicino a 0, più sono identiche; più si avvicina a 1, più sono diverse)
    # Calcoliamo le distanze all'interno del set REALE
    dist_real_real = cosine_distances(real_features)
    # Rimuoviamo la diagonale (distanza di un'immagine da se stessa, che è 0)
    dist_real_real = dist_real_real[np.triu_indices_from(dist_real_real, k=1)]
    
    # Calcoliamo le distanze all'interno del set FAKE
    dist_fake_fake = cosine_distances(fake_features)
    dist_fake_fake = dist_fake_fake[np.triu_indices_from(dist_fake_fake, k=1)]
    
    # Calcoliamo le distanze incrociate REALE vs FAKE
    dist_real_fake = cosine_distances(real_features, fake_features).flatten()
    
    mean_rr, std_rr = np.mean(dist_real_real), np.std(dist_real_real)
    mean_ff, std_ff = np.mean(dist_fake_fake), np.std(dist_fake_fake)
    mean_rf, std_rf = np.mean(dist_real_fake), np.std(dist_real_fake)
    
    # Calcolo della "grandezza" del manifold (Feature Variance via Traccia della Covarianza)
    cov_real = np.cov(real_features, rowvar=False)
    cov_fake = np.cov(fake_features, rowvar=False)
    trace_real = np.trace(cov_real)
    trace_fake = np.trace(cov_fake)
    
    print(f"  ➜ Real-Real Cosine Dist: {mean_rr:.4f} ± {std_rr:.4f}")
    print(f"  ➜ Fake-Fake Cosine Dist: {mean_ff:.4f} ± {std_ff:.4f}")
    print(f"  ➜ Real-Fake Cosine Dist: {mean_rf:.4f} ± {std_rf:.4f}")
    print(f"  ➜ Real Manifold Volume (Trace): {trace_real:.4f}")
    print(f"  ➜ Fake Manifold Volume (Trace): {trace_fake:.4f}")
    
    # --- 7. VISUALIZZAZIONI ---
    print("\n[Plot] Generazione visualizzazioni PCA e t-SNE...")
    
    # Uniamo le feature per PCA/t-SNE
    all_features = np.vstack([real_features, fake_features])
    labels = np.array(['Real'] * len(real_features) + ['Fake'] * len(fake_features))
    
    plt.figure(figsize=(18, 5))
    
    # PLOT 1: Istogramma Distanze
    plt.subplot(1, 3, 1)
    sns.kdeplot(dist_real_real, fill=True, label="Real-Real", color="blue")
    sns.kdeplot(dist_fake_fake, fill=True, label="Fake-Fake", color="orange")
    sns.kdeplot(dist_real_fake, fill=True, label="Real-Fake", color="green", alpha=0.3)
    plt.title("Pairwise Cosine Distances KDE")
    plt.xlabel("Cosine Distance")
    plt.ylabel("Density")
    plt.legend()
    
    # PLOT 2: PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_features)
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, palette={"Real":"blue", "Fake":"orange"}, alpha=0.7)
    plt.title(f"PCA 2D Projection (Explains {pca.explained_variance_ratio_.sum()*100:.1f}%)")
    
    # PLOT 3: t-SNE
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    tsne_result = tsne.fit_transform(all_features)
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette={"Real":"blue", "Fake":"orange"}, alpha=0.7)
    plt.title("t-SNE 2D Projection")
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "manifold_diversity_plots.png")
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Grafici salvati in {plot_path}")
    
    # --- 8. CONCLUSIONE AUTOMATICA ---
    print("\n" + "="*60)
    print(" 🤖 DIAGNOSI AUTOMATICA")
    print("="*60)
    
    # Rapporto tra covarianze
    trace_ratio = trace_fake / trace_real
    # Rapporto tra distanze medie
    dist_ratio = mean_ff / mean_rr
    
    if trace_ratio < 0.5 or dist_ratio < 0.6:
        print("🚨 SEVERE MANIFOLD CONTRACTION (DIVERSITY COLLAPSE) RILEVATO!")
        print("La GAN sta generando immagini troppo simili tra loro rispetto alla varietà del mondo reale.")
        print("Questo spiega perché la ResNet non migliora: i dati sintetici non offrono 'nuovi' pattern utili,")
        print("ma ripetono sempre le stesse features, non arricchendo il decision boundary.")
    elif trace_ratio > 1.2:
        print("⚠️ SOVRAGENERAZIONE / ARTEFATTI ECCESSIVI RILEVATI!")
        print("La GAN ha una varianza esagerata, molto superiore ai dati reali. Potrebbe star generando")
        print("molti artefatti irrealistici (le distanze Fake-Fake sono immense).")
    else:
        print("✅ FAKE DIVERSITY COMPARABLE TO REAL!")
        print("La GAN copre il manifold reale in modo eccellente. Non c'è diversity collapse.")
        print("Se la ResNet non migliora la F1-score, il motivo NON è la mancanza di diversità.")
        print("Ipotizzabili cause alternative:")
        print("  - Il dataset originale conteneva già informazioni sufficienti (Classificatore già a plateau).")
        print("  - Le immagini fake sono 'belle' ma povere di caratteristiche clinicamente discriminanti (mancano le vere patologie).")
        
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
