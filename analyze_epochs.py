import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.linalg
from tqdm import tqdm

# Import dal progetto (ora che il file è nella root, funzionano direttamente)
from config import DATASET_DIR, GAN_NZ, GAN_N_CLASS, GAN_NC, GAN_D, SEED, RESNET_IMG_SIZE
from models.wgan import Generator
from dataset.loader import setup_dataset
from utils.seed import set_seed
from utils.exp.main_prova3 import DomainDataset

# ==============================================================================
# ⚙️ IMPOSTAZIONI DELL'ESPERIMENTO
# ==============================================================================
# Cartella dove sono salvati i checkpoint della GAN
GAN_CHECKPOINTS_DIR = "./results/gan_checkpoints"

# Epoche da analizzare e confrontare
EPOCHS_TO_ANALYZE = [50, 100, 150, 200, 250, 300]

# Classe da analizzare (il problema di recall è su NORMAL)
TARGET_CLASS = "NORMAL"

# Numero di campioni reali e fake per epoca (1000 è buono, ma 500 è più veloce per testare)
NUM_SAMPLES = 500

# Dove salvare i risultati
RESULTS_DIR = "./results/multi_epoch_analysis"
# ==============================================================================

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calcola la distanza di Fréchet (FID proxy) tra due distribuzioni normali multivariate."""
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

def extract_features(loader, model, device):
    """Estrae le feature ResNet18 per un DataLoader."""
    model.eval()
    features = []
    with torch.no_grad():
        for x, _ in tqdm(loader, leave=False, desc="Estrazione Feature"):
            feats = model(x.to(device)).cpu().numpy()
            features.append(feats)
    return np.concatenate(features, axis=0)

def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Setup Dataset Reale
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res: 
        print("Errore: Dataset non trovato.")
        return
    train_dir, _, _ = res
    
    real_class_dir = os.path.join(train_dir, TARGET_CLASS)
    if not os.path.exists(real_class_dir):
        print(f"Errore: Cartella {real_class_dir} non trovata.")
        return
        
    print(f"\n[1] Setup Modello ResNet per Feature Extraction...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity() # Rimuoviamo il fully connected per ottenere feature raw a 512d
    resnet = resnet.to(device).eval()
    
    transform = transforms.Compose([
        transforms.Resize((RESNET_IMG_SIZE, RESNET_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n[2] Elaborazione Dati Reali...")
    # Usiamo un dataset dummy per caricare le immagini
    ds_real = DomainDataset(real_class_dir, real_class_dir, transform=transform)
    # Prendiamo solo i primi NUM_SAMPLES per bilanciare
    indices = np.random.choice(len(os.listdir(real_class_dir)), min(NUM_SAMPLES, len(os.listdir(real_class_dir))), replace=False)
    loader_real = DataLoader(Subset(ds_real, indices), batch_size=64, shuffle=False)
    
    real_features = extract_features(loader_real, resnet, device)
    real_mu = np.mean(real_features, axis=0)
    real_sigma = np.cov(real_features, rowvar=False)
    
    # 3. Setup Generatore
    G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GAN_D).to(device)
    target_class_idx = 0 if TARGET_CLASS == "NORMAL" else 1 # Assumendo NORMAL=0, PNEUMONIA=1 nel loader
    
    # Dizionari per salvare i risultati
    all_features = [real_features]
    all_labels = ['Real'] * len(real_features)
    epoch_fd_scores = {}
    
    print("\n[3] Generazione e Analisi per le diverse Epoche GAN...")
    for epoch in EPOCHS_TO_ANALYZE:
        ckpt_path = os.path.join(GAN_CHECKPOINTS_DIR, f"G_epoch_{epoch}.pth")
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️ Checkpoint {ckpt_path} non trovato. Salto Epoca {epoch}.")
            continue
            
        print(f"\n--- Analisi Epoca {epoch} ---")
        G.load_state_dict(torch.load(ckpt_path, map_location=device))
        G.eval()
        
        # Generazione batch di immagini
        fake_tensors = []
        with torch.no_grad():
            generated = 0
            while generated < NUM_SAMPLES:
                bs = min(64, NUM_SAMPLES - generated)
                z = torch.randn(bs, GAN_NZ, 1, 1, device=device)
                labels = torch.full((bs,), target_class_idx, dtype=torch.long, device=device)
                labels_onehot = torch.nn.functional.one_hot(labels, GAN_N_CLASS).view(bs, GAN_N_CLASS, 1, 1).float()
                
                fakes = G(z, labels_onehot)
                fakes = (fakes + 1.0) / 2.0 # [-1, 1] -> [0, 1]
                fake_tensors.append(fakes.cpu())
                generated += bs
                
        # Estrazione feature direttamente dai tensori (bypassando il disco)
        all_fakes = torch.cat(fake_tensors, dim=0) # [NUM_SAMPLES, 1, 128, 128]
        # ResNet vuole 3 canali (RGB), i fakes sono 1 canale (Grayscale). Duplichiamo.
        all_fakes_rgb = all_fakes.repeat(1, 3, 1, 1) 
        
        # Riapplichiamo la normalizzazione ImageNet
        norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        all_fakes_norm = (all_fakes_rgb - norm_mean) / norm_std
        
        fake_features = []
        with torch.no_grad():
            for i in range(0, NUM_SAMPLES, 64):
                batch = all_fakes_norm[i:i+64].to(device)
                fake_features.append(resnet(batch).cpu().numpy())
        fake_features = np.concatenate(fake_features, axis=0)
        
        # Calcolo FD
        fake_mu = np.mean(fake_features, axis=0)
        fake_sigma = np.cov(fake_features, rowvar=False)
        fd = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
        epoch_fd_scores[epoch] = fd
        
        print(f"  ➜ Fréchet Distance (Proxy FID) Epoca {epoch}: {fd:.4f}")
        
        # Aggiungo alle feature globali per PCA/t-SNE
        all_features.append(fake_features)
        all_labels.extend([f"Epoch_{epoch}"] * len(fake_features))

    if not epoch_fd_scores:
        print("\nNessun checkpoint trovato. Assicurati che i path in GAN_CHECKPOINTS_DIR siano corretti.")
        return

    # 4. Spazio e Plot (PCA & t-SNE)
    print("\n[4] Generazione Dashboard PCA e t-SNE multi-epoca...")
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    # Colori: Reali in blu/nero, Epoche sintetiche in una palette che va dal chiaro allo scuro
    unique_labels = ['Real'] + [f"Epoch_{e}" for e in epoch_fd_scores.keys()]
    palette = sns.color_palette("rocket_r", len(epoch_fd_scores))
    color_dict = {'Real': '#000000'} # Nero per i dati reali
    for i, e in enumerate(epoch_fd_scores.keys()):
        color_dict[f"Epoch_{e}"] = palette[i]

    plt.figure(figsize=(18, 8))
    
    # PCA
    print("  Calcolo PCA...")
    pca = PCA(n_components=2).fit_transform(X)
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=pca[:,0], y=pca[:,1], hue=y, hue_order=unique_labels, palette=color_dict, alpha=0.6, s=15)
    plt.title(f"PCA Manifold Trajectory ({TARGET_CLASS})")
    
    # t-SNE
    print("  Calcolo t-SNE (potrebbe richiedere un minuto)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED).fit_transform(X)
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=tsne[:,0], y=tsne[:,1], hue=y, hue_order=unique_labels, palette=color_dict, alpha=0.6, s=15)
    plt.title(f"t-SNE Manifold Trajectory ({TARGET_CLASS})")
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"trajectory_analysis_{TARGET_CLASS}.png")
    plt.savefig(plot_path, dpi=200)
    print(f"✅ Dashboard salvata in {plot_path}")
    
    # Riepilogo FD
    print("\n" + "="*50)
    print(" 📊 RIEPILOGO FRÉCHET DISTANCE (Più basso è meglio)")
    print("="*50)
    for epoch, fd in sorted(epoch_fd_scores.items()):
        print(f"  Epoca {epoch:3d} : {fd:.2f}")
    
    best_epoch = min(epoch_fd_scores, key=epoch_fd_scores.get)
    print(f"\n🏆 L'Epoca più 'realistica' (vicina al manifold vero) è: {best_epoch}")

if __name__ == '__main__':
    main()
