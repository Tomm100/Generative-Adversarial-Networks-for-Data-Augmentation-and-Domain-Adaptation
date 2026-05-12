import os
import shutil
import numpy as np
import scipy.linalg
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
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
TARGET_CLASS = "NORMAL"  
NUM_SAMPLES = 1000        
RESULTS_DIR = "./results/domain_shift_analysis"
# ==============================================================================

# ------------------------------------------------------------------------------
# DATASET BINARIO: REAL (0) vs FAKE (1)
# ------------------------------------------------------------------------------
class DomainDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        # Real = label 0
        real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for f in real_files:
            self.samples.append((f, 0))
            
        # Fake = label 1
        fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        for f in fake_files:
            self.samples.append((f, 1))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def prepare_data_folders(G, device, train_dir, num_samples, target_class_idx):
    print(f"\n[1/7] Preparazione Dataset Domain Shift ({num_samples} Real vs {num_samples} Fake)")
    
    out_real_dir = os.path.join(RESULTS_DIR, "data", "real")
    out_fake_dir = os.path.join(RESULTS_DIR, "data", "fake")
    if os.path.exists(out_real_dir): shutil.rmtree(out_real_dir)
    if os.path.exists(out_fake_dir): shutil.rmtree(out_fake_dir)
    os.makedirs(out_real_dir, exist_ok=True)
    os.makedirs(out_fake_dir, exist_ok=True)
    
    # Prendi reali
    src_real_dir = os.path.join(train_dir, TARGET_CLASS)
    real_files = [f for f in os.listdir(src_real_dir) if not f.startswith('.')]
    selected_real = np.random.choice(real_files, min(num_samples, len(real_files)), replace=False)
    for i, f in enumerate(selected_real):
        shutil.copy(os.path.join(src_real_dir, f), os.path.join(out_real_dir, f"real_{i}.png"))
        
    # Genera fake
    G.eval()
    generated = 0
    batch_size = 64
    with torch.no_grad():
        while generated < len(selected_real): # match esatto
            bs = min(batch_size, len(selected_real) - generated)
            z = torch.randn(bs, GAN_NZ, 1, 1, device=device)
            labels = torch.full((bs,), target_class_idx, dtype=torch.long, device=device)
            labels_onehot = F.one_hot(labels, GAN_N_CLASS).view(bs, GAN_N_CLASS, 1, 1).float()
            
            fakes = G(z, labels_onehot)
            fakes = (fakes + 1.0) / 2.0
            
            for i in range(bs):
                img_tensor = fakes[i].cpu().squeeze()
                img_np = (img_tensor.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_np, mode='L')
                img.save(os.path.join(out_fake_dir, f"fake_{generated + i}.png"))
            generated += bs
            
    print(f"✅ Trovati {len(selected_real)} Real, Generati {generated} Fake.")
    return out_real_dir, out_fake_dir

# ------------------------------------------------------------------------------
# RESNET DOMAIN CLASSIFIER
# ------------------------------------------------------------------------------
def train_domain_classifier(train_loader, test_loader, device, mode="frozen", epochs=5):
    """
    mode='frozen': addestra solo l'ultimo strato.
    mode='finetune': addestra tutto il network end-to-end.
    """
    print(f"\n[2/7] Domain Classification - Mode: {mode.upper()}")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    if mode == "frozen":
        for param in model.parameters():
            param.requires_grad = False
            
    model.fc = nn.Linear(model.fc.in_features, 1) # classificazione binaria (sigmoid + BCELoss)
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters() if mode == "frozen" else model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoca {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            
    # Valutazione
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device).float().unsqueeze(1)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    print(f"  ➜ {mode.upper()} Results: Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    return acc, f1, auc

# ------------------------------------------------------------------------------
# MATHEMATICAL DISTRIBUTION METRICS
# ------------------------------------------------------------------------------
def get_features(real_dir, fake_dir, device):
    print("\n[3/7] Estrazione Feature per Statistiche di Distribuzione...")
    transform = transforms.Compose([
        transforms.Resize((RESNET_IMG_SIZE, RESNET_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    ds_real = DomainDataset(real_dir, real_dir, transform=transform) # trucco per caricare solo i reali
    ds_fake = DomainDataset(fake_dir, fake_dir, transform=transform)
    
    loader_real = DataLoader(Subset(ds_real, range(len(os.listdir(real_dir)))), batch_size=64, shuffle=False)
    loader_fake = DataLoader(Subset(ds_fake, range(len(os.listdir(fake_dir)))), batch_size=64, shuffle=False)
    
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device).eval()
    
    real_feats, fake_feats = [], []
    with torch.no_grad():
        for x, _ in loader_real: real_feats.append(resnet(x.to(device)).cpu().numpy())
        for x, _ in loader_fake: fake_feats.append(resnet(x.to(device)).cpu().numpy())
            
    return np.concatenate(real_feats), np.concatenate(fake_feats)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calcola la distanza di Fréchet tra due gaussiane multivariate."""
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

def compute_mmd(X, Y, gamma=1.0):
    """Calcola Maximum Mean Discrepancy (RBF kernel approssimato per velocità su batch)."""
    X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y)
    
    xx, yy, zz = torch.mm(X_tensor, X_tensor.t()), torch.mm(Y_tensor, Y_tensor.t()), torch.mm(X_tensor, Y_tensor.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * zz
    
    XX, YY, XY = (torch.zeros(xx.shape).to(X_tensor), torch.zeros(xx.shape).to(X_tensor), torch.zeros(xx.shape).to(X_tensor))
    bandwidth_range = [0.2, 0.5, 0.9, 1.3]
    for a in bandwidth_range:
        XX += torch.exp(-0.5 * dxx / a)
        YY += torch.exp(-0.5 * dyy / a)
        XY += torch.exp(-0.5 * dxy / a)
        
    return torch.mean(XX + YY - 2.0 * XY).item()

# ------------------------------------------------------------------------------
# LOW LEVEL SPECTRAL STATISTICS (FFT & HISTOGRAM)
# ------------------------------------------------------------------------------
def calculate_1d_psd(image_np):
    """Calcola lo spettro di potenza radiale 1D da una 2D FFT."""
    f = np.fft.fft2(image_np)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)**2
    
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r = r.astype(int)
    
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile

def analyze_low_level_stats(real_dir, fake_dir):
    print("\n[5/7] Analisi Statistiche di Basso Livello (Pixel & Frequenze)...")
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir)[:100]]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)[:100]]
    
    real_psds, fake_psds = [], []
    real_pixels, fake_pixels = [], []
    
    for f in real_files:
        img = np.array(Image.open(f).convert('L'))
        real_psds.append(calculate_1d_psd(img))
        real_pixels.append(img.flatten())
        
    for f in fake_files:
        img = np.array(Image.open(f).convert('L'))
        fake_psds.append(calculate_1d_psd(img))
        fake_pixels.append(img.flatten())
        
    min_len = min([len(p) for p in real_psds])
    avg_real_psd = np.mean([p[:min_len] for p in real_psds], axis=0)
    avg_fake_psd = np.mean([p[:min_len] for p in fake_psds], axis=0)
    
    return np.concatenate(real_pixels), np.concatenate(fake_pixels), avg_real_psd, avg_fake_psd

# ------------------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------------------
def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res: return
    train_dir, _, _ = res
    
    dataset_temp = datasets.ImageFolder(root=train_dir)
    target_class_idx = dataset_temp.classes.index(TARGET_CLASS)
    
    G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GAN_D).to(device)
    G.load_state_dict(torch.load(CUSTOM_GAN_WEIGHTS_PATH, map_location=device))
    
    # 1. Dataset
    real_dir, fake_dir = prepare_data_folders(G, device, train_dir, NUM_SAMPLES, target_class_idx)
    
    transform = transforms.Compose([
        transforms.Resize((RESNET_IMG_SIZE, RESNET_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = DomainDataset(real_dir, fake_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 2. Classificazione
    acc_frozen, _, auc_frozen = train_domain_classifier(train_loader, test_loader, device, mode="frozen")
    acc_fine, _, auc_fine = train_domain_classifier(train_loader, test_loader, device, mode="finetune")
    
    # 3. Metriche Distribuzione
    real_feats, fake_feats = get_features(real_dir, fake_dir, device)
    
    mu_r, cov_r = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu_f, cov_f = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
    
    fid = calculate_frechet_distance(mu_r, cov_r, mu_f, cov_f)
    coral = np.sum(np.square(cov_r - cov_f)) / (4 * 512 * 512)
    mmd = compute_mmd(real_feats[:500], fake_feats[:500]) # Su subset per memoria
    
    print("\n[4/7] Metriche Formali di Allineamento:")
    print(f"  ➜ Fréchet Distance (Proxy FID): {fid:.4f}")
    print(f"  ➜ CORAL Distance (Covariance Alignment): {coral:.4f}")
    print(f"  ➜ MMD (Maximum Mean Discrepancy): {mmd:.4f}")
    
    # 4. Spazio e Plot
    print("\n[6/7] Generazione Visualizzazioni...")
    rp, fp, r_psd, f_psd = analyze_low_level_stats(real_dir, fake_dir)
    
    plt.figure(figsize=(20, 10))
    
    # PCA
    plt.subplot(2, 3, 1)
    all_f = np.vstack([real_feats, fake_feats])
    labels = np.array(['Real']*len(real_feats) + ['Fake']*len(fake_feats))
    pca = PCA(n_components=2).fit_transform(all_f)
    sns.scatterplot(x=pca[:,0], y=pca[:,1], hue=labels, alpha=0.5)
    plt.title("PCA (Feature Space)")
    
    # t-SNE
    plt.subplot(2, 3, 2)
    tsne = TSNE(n_components=2, perplexity=30).fit_transform(all_f)
    sns.scatterplot(x=tsne[:,0], y=tsne[:,1], hue=labels, alpha=0.5)
    plt.title("t-SNE (Feature Space)")
    
    # Hist
    plt.subplot(2, 3, 3)
    sns.kdeplot(np.random.choice(rp, 10000), label="Real", fill=True)
    sns.kdeplot(np.random.choice(fp, 10000), label="Fake", fill=True)
    plt.title("Pixel Intensity Histogram")
    plt.legend()
    
    # PSD
    plt.subplot(2, 3, 4)
    plt.plot(np.log(r_psd + 1e-8), label="Real")
    plt.plot(np.log(f_psd + 1e-8), label="Fake")
    plt.title("1D Power Spectrum (FFT)")
    plt.xlabel("Spatial Frequency")
    plt.ylabel("Log Power")
    plt.legend()
    
    # ROC Plot Proxy
    plt.subplot(2, 3, 5)
    plt.bar(["Frozen AUC", "Fine-Tuned AUC"], [auc_frozen, auc_fine], color=['blue', 'orange'])
    plt.ylim(0.4, 1.0)
    plt.axhline(0.5, color='red', linestyle='--')
    plt.title("Domain Separation Predictability")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "domain_shift_dashboard.png"), dpi=150)
    print("✅ Dashboard salvata!")
    
    # 7. Diagnosi Automatica
    print("\n" + "="*60)
    print(" 🤖 DIAGNOSI AUTOMATICA DEL DOMAIN SHIFT")
    print("="*60)
    
    if auc_frozen > 0.95 or auc_fine > 0.98:
        print("🚨 SEVERE SYNTHETIC DOMAIN SHIFT!")
        print("Il classificatore separa perfettamente (o quasi) i reali dai fake.")
        print("La GAN genera immagini con una 'firma' facilmente riconoscibile dalla rete (shortcut).")
        if auc_frozen > 0.90:
            print("→ Il problema è già presente nelle feature pre-addestrate (texture/frequenze).")
            print("✅ CONCLUSIONE: Un approccio di DOMAIN ADAPTATION (es. DANN) è STRATAMENTE CONSIGLIATO")
            print("   per forzare il classificatore a estrarre feature invarianti rispetto all'origine (Real/Fake).")
    elif auc_fine > 0.80:
        print("⚠️ MODERATE DOMAIN SHIFT.")
        print("I domini sono separabili ma non in modo triviale.")
        print("✅ CONCLUSIONE: Una DANN potrebbe fornire un boost marginale, ma è sufficiente")
        print("   una pesante data augmentation classica per confondere la rete.")
    else:
        print("✅ DOMAIN SHIFT ASSENTE O MINIMO.")
        print("La rete fa molta fatica a distinguere veri da finti.")
        print("La GAN genera un manifold perfettamente allineato.")
        print("Se la Data Augmentation non funziona, la DANN non aiuterà. Il problema è altrove.")
        
    print("="*60)

if __name__ == '__main__':
    main()
