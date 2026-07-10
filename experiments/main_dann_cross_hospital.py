"""
Esperimento Cross-Hospital Domain Adaptation (DANN, unsupervised).

  Source domain : Kermany chest X-ray (etichettato)        -> SOURCE_ROOT
  Target domain : VinDr-PCXR   (usato SENZA etichette       -> TARGET_ROOT
                  per l'allineamento; le etichette del test
                  servono solo per la valutazione finale)

Confronto controllato: due run identici (init ImageNet, stessi dati, stesso seed)
che differiscono SOLO nella forza della GRL:
   lambda_max = 0.0  -> nessun allineamento = baseline SOURCE-ONLY  (misura il domain shift)
   lambda_max = 1.0  -> DANN attiva                                 (misura il recupero)

Ogni cartella (source/target) deve avere la struttura:
   <root>/{train,val,test}/{NORMAL,PNEUMONIA}/*.png

Uso:  python experiments/main_dann_cross_hospital.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, average_precision_score
from tqdm import tqdm
from PIL import Image

# Root del repo nel path: consente l'esecuzione da experiments/ (python experiments/main_dann_cross_hospital.py)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASET_DIR, RESNET_IMG_SIZE, METRICS_DIR, SEED, SYNTHETIC_DIR
from models.dann_synth import DANNSynth
from utils.seed import set_seed

# CONFIG -- MODIFICA QUI
SOURCE_ROOT = DATASET_DIR                       # Kermany (train/val/test dentro)
TARGET_ROOT = "./VinDr_Target_DA_Ready"         # VinDr scompattato (train/val/test)

# --- SOURCE augmentato con i sintetici della BestGAN ---
#   AUGMENT_SOURCE=True -> il TRAIN del source = Kermany reale + sintetici NORMAL
#                          (val/test restano reali). False -> source solo reale.
AUGMENT_SOURCE    = True
BEST_GAN_CKPT     = "./results/gan_checkpoints/G_epoch_210.pth"   # <- generatore BestGAN (SNGAN128 Complete, ep.210)
SYNTH_GAP_PERCENT = 75                           # % del gap Normal/Pneumonia da colmare
GEN_D             = 128                           # base channels del generatore

EPOCHS     = 30
LR_FEAT    = 1e-4
LR_CLASS   = 1e-3
BETA1      = 0.5
BATCH      = 32
CKPT_DIR   = "./results/dann_cross_checkpoints"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]           # ImageFolder: N=0, P=1


def make_loaders(root, img_size, batch, balanced_train):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tr = datasets.ImageFolder(os.path.join(root, "train"), tf)
    va = datasets.ImageFolder(os.path.join(root, "val"),   tf)
    te = datasets.ImageFolder(os.path.join(root, "test"),  tf)

    if balanced_train:
        labels = [l for _, l in tr.samples]
        w = 1.0 / np.bincount(labels)
        sampler = WeightedRandomSampler([w[l] for l in labels], len(labels), replacement=True)
        tr_loader = DataLoader(tr, batch, sampler=sampler, drop_last=True, num_workers=2, pin_memory=True)
    else:
        tr_loader = DataLoader(tr, batch, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    va_loader = DataLoader(va, batch, shuffle=False, num_workers=2, pin_memory=True)
    te_loader = DataLoader(te, batch, shuffle=False, num_workers=2, pin_memory=True)
    print(f"  {root}: train {len(tr)} | val {len(va)} | test {len(te)} | classi {tr.classes}")
    return tr_loader, va_loader, te_loader


class _SynthNormalDS(torch.utils.data.Dataset):
    """Immagini sintetiche NORMAL (etichetta 0)."""
    def __init__(self, syn_dir, transform):
        exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
        self.files = [os.path.join(syn_dir, f) for f in os.listdir(syn_dir) if f.endswith(exts)]
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        return self.transform(Image.open(self.files[i]).convert("RGB")), 0


def _num_synth_from_gap():
    tr = os.path.join(SOURCE_ROOT, "train")
    n_norm = len(os.listdir(os.path.join(tr, "NORMAL")))
    n_pneu = len(os.listdir(os.path.join(tr, "PNEUMONIA")))
    return int((n_pneu - n_norm) * SYNTH_GAP_PERCENT / 100.0), n_norm, n_pneu


def _ensure_synthetic(device):
    """Genera i sintetici NORMAL dalla BestGAN se non gia' presenti in SYNTHETIC_DIR/NORMAL."""
    normal_dir = os.path.join(SYNTHETIC_DIR, "NORMAL")
    num_synth, n_norm, n_pneu = _num_synth_from_gap()
    have = len(os.listdir(normal_dir)) if os.path.isdir(normal_dir) else 0
    print(f"  Gap Normal/Pneumonia nel train: {n_pneu - n_norm}  ->  {SYNTH_GAP_PERCENT}% = {num_synth} sintetici NORMAL")
    if have >= num_synth:
        print(f"  Sintetici gia' presenti ({have}). Skip generazione.")
        return
    from config import GAN_NZ, GAN_N_CLASS, GAN_NC
    from evaluation.eval import generate_synthetic_images
    from models.sngan_128 import SNGenerator as Generator
    print(f"  Genero {num_synth} sintetici NORMAL da {BEST_GAN_CKPT} ...")
    G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GEN_D).to(device)
    G.load_state_dict(torch.load(BEST_GAN_CKPT, map_location=device))
    generate_synthetic_images(G, num_gen_normal=num_synth, num_gen_pneumonia=0,
                              nz=GAN_NZ, n_class=GAN_N_CLASS, device=device, syn_dir=SYNTHETIC_DIR)


def get_source_loaders(device):
    """Source = Kermany. Se AUGMENT_SOURCE, il TRAIN include i sintetici NORMAL
    (val/test restano SEMPRE reali)."""
    tf = transforms.Compose([
        transforms.Resize((RESNET_IMG_SIZE, RESNET_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    real_train = datasets.ImageFolder(os.path.join(SOURCE_ROOT, "train"), tf)
    va = datasets.ImageFolder(os.path.join(SOURCE_ROOT, "val"),  tf)
    te = datasets.ImageFolder(os.path.join(SOURCE_ROOT, "test"), tf)

    if AUGMENT_SOURCE:
        _ensure_synthetic(device)
        synth = _SynthNormalDS(os.path.join(SYNTHETIC_DIR, "NORMAL"), tf)
        train_ds = torch.utils.data.ConcatDataset([real_train, synth])
        print(f"  Source AUGMENTATO: {len(real_train)} reali + {len(synth)} sintetici NORMAL = {len(train_ds)}")
    else:
        train_ds = real_train
        print(f"  Source REALE: {len(real_train)}")

    dl = dict(num_workers=2, pin_memory=True)
    tr_loader = DataLoader(train_ds, BATCH, shuffle=True, drop_last=True, **dl)
    va_loader = DataLoader(va, BATCH, shuffle=False, **dl)
    te_loader = DataLoader(te, BATCH, shuffle=False, **dl)
    return tr_loader, va_loader, te_loader


def lambda_schedule(p):
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)   # Ganin et al. 2016


@torch.no_grad()
def evaluate(model, loader, device):
    """Ritorna (report_dict, AUPRC_pneumonia)."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        logits, _, _ = model(x.to(device), lambda_=0.0)
        probs.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())   # P(PNEUMONIA)
        labels.append(y.numpy())
    probs = np.concatenate(probs); labels = np.concatenate(labels)
    preds = (probs >= 0.5).astype(int)
    rep = classification_report(labels, preds, target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    auprc = average_precision_score((labels == 1).astype(int), probs)
    return rep, auprc


def train_dann_uda(source_loader, target_loader, val_loader, device,
                   lambda_max, tag, epochs=EPOCHS):
    """DANN unsupervised: class loss SOLO sul source, domain loss su source+target.
    lambda_max=0 -> baseline source-only (nessun allineamento)."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    set_seed(SEED)
    model = DANNSynth(num_classes=2, pretrained=True).to(device)   # cold-start ImageNet

    feat = [p for p in model.feature_extractor.parameters() if p.requires_grad]
    opt = optim.Adam([
        {"params": feat,                                 "lr": LR_FEAT},
        {"params": model.label_predictor.parameters(),   "lr": LR_CLASS},
        {"params": model.domain_discriminator.parameters(), "lr": LR_CLASS},
    ], betas=(BETA1, 0.999))
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    n_batches = min(len(source_loader), len(target_loader))
    total = epochs * n_batches
    hist = {"val_f1": [], "dom_acc": [], "lambda": []}
    best_f1, best_state = -1.0, None
    ckpt = os.path.join(CKPT_DIR, f"best_{tag}.pth")

    print(f"\n{'#'*60}\n  RUN {tag}  (lambda_max={lambda_max}) | {n_batches} batch/epoca\n{'#'*60}")
    for epoch in range(epochs):
        model.train()
        s_it, t_it = iter(source_loader), iter(target_loader)
        dom_ok, dom_tot = 0, 0
        for i in tqdm(range(n_batches), desc=f"  Epoch {epoch+1}/{epochs}", leave=False):
            lam = lambda_max * lambda_schedule((epoch * n_batches + i) / total)
            xs, ys = next(s_it); xt, _ = next(t_it)
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

            cls_s, dom_s, _ = model(xs, lam)
            _,     dom_t, _ = model(xt, lam)

            loss_cls = ce(cls_s, ys)                                  # SOLO source
            loss_dom = (bce(dom_s, torch.zeros_like(dom_s)) +
                        bce(dom_t, torch.ones_like(dom_t)))
            loss = loss_cls + loss_dom

            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                dom_ok += (torch.sigmoid(dom_s) < 0.5).sum().item()
                dom_ok += (torch.sigmoid(dom_t) >= 0.5).sum().item()
                dom_tot += xs.size(0) + xt.size(0)

        rep, _ = evaluate(model, val_loader, device)      # model selection su target VAL
        val_f1 = rep["macro avg"]["f1-score"]
        dom_acc = dom_ok / dom_tot
        hist["val_f1"].append(val_f1); hist["dom_acc"].append(dom_acc)
        hist["lambda"].append(lambda_max * lambda_schedule((epoch + 1) / epochs))
        print(f"  Epoch {epoch+1}: target-val MacroF1={val_f1:.4f} | dom_acc={dom_acc:.3f} | lam={hist['lambda'][-1]:.3f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state); torch.save(best_state, ckpt)
    print(f"  Best target-val MacroF1: {best_f1:.4f}  ({ckpt})")
    return model, hist


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\n  Cross-Hospital DANN  (device: {device})\n{'='*60}")

    print("Source (Kermany):")
    src_train, _, src_test = get_source_loaders(device)
    print("Target (VinDr):")
    tgt_train, tgt_val, tgt_test = make_loaders(TARGET_ROOT, RESNET_IMG_SIZE, BATCH, balanced_train=False)

    os.makedirs(METRICS_DIR, exist_ok=True)
    results, hists = {}, {}
    for lam, tag in [(0.0, "SourceOnly_lam0"), (1.0, "DANN_lam1")]:
        model, hist = train_dann_uda(src_train, tgt_train, tgt_val, device, lam, tag)
        rep_t, auprc_t = evaluate(model, tgt_test, device)
        results[tag] = (rep_t, auprc_t)
        hists[tag] = hist
        if lam == 0.0:   # baseline: mostra anche la performance IN-DOMAIN (source test)
            rep_s, auprc_s = evaluate(model, src_test, device)
            results["SourceOnly_SRCtest"] = (rep_s, auprc_s)

    # ---- figura dinamiche lambda (run DANN) ----
    h = hists["DANN_lam1"]; ep = range(1, len(h["val_f1"]) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ep, h["val_f1"], "b-o", ms=3, label="Target-val MacroF1")
    ax1.plot(ep, h["lambda"], "g--", label="lambda"); ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Target-val MacroF1 / lambda")
    ax2 = ax1.twinx(); ax2.plot(ep, h["dom_acc"], "r-s", ms=3, label="Domain-disc acc")
    ax2.set_ylabel("Domain-disc accuracy"); ax2.set_ylim(0.4, 1.02)
    ax1.legend(loc="lower left"); ax2.legend(loc="lower right")
    plt.title("Cross-Hospital DANN: target-val F1 e domain accuracy vs epoca")
    plt.tight_layout()
    fig_path = os.path.join(METRICS_DIR, "dann_cross_lambda_dynamics.png")
    plt.savefig(fig_path, dpi=150); plt.close(fig)

    # ---- tabella riassuntiva ----
    def row(tag, rep, auprc):
        return (f"  {tag:<26}{rep[CLASS_NAMES[0]]['f1-score']:>8.3f}"
                f"{rep[CLASS_NAMES[1]]['f1-score']:>10.3f}"
                f"{rep['macro avg']['f1-score']:>9.3f}{auprc:>10.3f}")
    print(f"\n{'='*70}\n  RISULTATI CROSS-HOSPITAL\n{'='*70}")
    print(f"  {'Config':<26}{'N F1':>8}{'P F1':>10}{'MacroF1':>9}{'AUPRC-P':>10}")
    print(row("SourceOnly (SOURCE test)", *results["SourceOnly_SRCtest"]))   # in-domain
    print(row("SourceOnly (TARGET test)", *results["SourceOnly_lam0"]))      # domain shift
    print(row("DANN       (TARGET test)", *results["DANN_lam1"]))            # recupero
    d = (results["DANN_lam1"][0]["macro avg"]["f1-score"]
         - results["SourceOnly_lam0"][0]["macro avg"]["f1-score"])
    print(f"\n  Delta Macro F1 (DANN - SourceOnly) sul TARGET: {d:+.3f}")
    print(f"  Figura: {fig_path}")


if __name__ == "__main__":
    main()
