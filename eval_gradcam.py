"""
Analisi Grad-CAM del classificatore ResNet-18 addestrato sul dataset
AUGMENTED-75% (real + 75% del gap colmato con sintetici).

Obiettivo: verificare DOVE guarda il classificatore per prendere la decisione,
e confrontare l'attenzione su:
  - immagini REALI Normali
  - immagini REALI Pneumonia
  - immagini SINTETICHE Normali (generate dalla GAN)

Serve a diagnosticare eventuali "shortcut" (attenzione su bordi, marker,
angoli o artefatti della GAN invece che sul parenchima polmonare).

Nessuna dipendenza esterna oltre a torch / torchvision / matplotlib / PIL.
Uso:  python eval_gradcam.py
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms

from config import (
    DATASET_DIR, SYNTHETIC_DIR, CHECKPOINTS_DIR, METRICS_DIR,
    RESNET_IMG_SIZE, RESNET_NUM_CLASSES, SEED,
)
from dataset.loader import setup_dataset
from models.resnet import ResNetClassifier

# ==============================================================================
# CONFIG -- MODIFICA QUI
# ==============================================================================
# Checkpoint del ResNet augmented-75% (quello che raggiunge N-F1 = 0.75).
# train.py salva come  best_model_<tag>.pth  in CHECKPOINTS_DIR: metti il tuo tag.
RESNET_CKPT = os.path.join(CHECKPOINTS_DIR, "best_model_Finale.pth")

# Checkpoint del ResNet BASELINE (real-only, dataset sbilanciato) per il confronto.
BASELINE_CKPT = os.path.join(CHECKPOINTS_DIR, "best_model_Baseline.pth")

N_PER_GROUP = 6                 # quante immagini per categoria (analisi per-gruppo)
N_COMPARE_PER_GROUP = 1         # righe per gruppo nella griglia baseline vs augmented (1 -> 3 righe)
OUT_DIR     = os.path.join(METRICS_DIR, "gradcam")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]   # ImageFolder ordina alfabeticamente -> N=0, P=1
GROUP_TRUE  = {"real_normal": 0, "real_pneumonia": 1, "synth_normal": 0}   # classe vera per gruppo
BORDER_FRAC = 0.15              # spessore cornice (frazione del lato) per la metrica "border energy"
# ==============================================================================


# ------------------------------------------------------------------ Grad-CAM --
class GradCAM:
    """Grad-CAM su un layer convoluzionale target (default: layer4 di ResNet-18)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._h1 = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, out):
        self.activations = out
        # hook sul tensore per ottenere il gradiente rispetto alle attivazioni
        out.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)                       # (1, num_classes)
        probs = F.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward()

        acts = self.activations.detach()             # (1, C, h, w)
        grads = self.gradients.detach()              # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)   # GAP dei gradienti -> (1, C, 1, 1)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))   # (1, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probs.detach().cpu().numpy()[0]

    def remove(self):
        self._h1.remove()


# ------------------------------------------------------------------ utility ---
def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_display_image(path, img_size):
    """Immagine RGB ridimensionata (senza normalizzazione) per l'overlay."""
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    return np.asarray(img).astype(np.float32) / 255.0


def overlay_cam(disp_rgb, cam, alpha=0.5):
    heat = cm.jet(cam)[..., :3]                      # heatmap RGB in [0,1]
    return np.clip((1 - alpha) * disp_rgb + alpha * heat, 0, 1)


def border_energy(cam, frac):
    """Frazione dell'attivazione Grad-CAM che cade nella cornice esterna.
    Valori alti = attenzione sui bordi (possibile shortcut su marker/artefatti)."""
    h, w = cam.shape
    b = max(1, int(round(min(h, w) * frac)))
    total = cam.sum() + 1e-8
    inner = cam[b:h - b, b:w - b].sum()
    return float((total - inner) / total)


def list_images(folder, n, seed):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".JPEG", ".PNG")
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    files.sort()
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:n]


def load_model(ckpt_path, device):
    model = ResNetClassifier(num_classes=RESNET_NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model


def build_comparison_grid(items, gc_baseline, gc_aug, transform, device, img_size, out_dir):
    """Griglia di confronto: righe = immagini, 2 colonne = [Baseline, Augmented-75%].
    items: lista di (filepath, group_name, true_idx). Grad-CAM calcolata sulla classe VERA
    per entrambi i modelli, cosi le due colonne spiegano lo stesso target."""
    if not items:
        print("  [!] Nessuna immagine per la griglia di confronto, salto.")
        return
    nrows = len(items)
    fig, axes = plt.subplots(nrows, 2, figsize=(6.4, 3.1 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, 2)

    for i, (fpath, group, true_idx) in enumerate(items):
        disp = load_display_image(fpath, img_size)
        x = transform(Image.open(fpath).convert("RGB")).unsqueeze(0).to(device)
        cam_b, _, pb = gc_baseline(x, class_idx=true_idx)
        cam_a, _, pa = gc_aug(x, class_idx=true_idx)

        for col, (cam, probs) in enumerate([(cam_b, pb), (cam_a, pa)]):
            ax = axes[i, col]
            ax.imshow(overlay_cam(disp, cam))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"pred={CLASS_NAMES[int(probs.argmax())]}  "
                         f"p={probs[true_idx]:.2f}", fontsize=8)
        axes[i, 0].set_ylabel(f"{group}\n[{CLASS_NAMES[true_idx]}]", fontsize=8)

    fig.text(0.28, 0.99, "BASELINE (real-only)", ha="center", va="top",
             fontsize=11, fontweight="bold")
    fig.text(0.76, 0.99, "AUGMENTED-75% (real+synth)", ha="center", va="top",
             fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "gradcam_compare_baseline_vs_aug75.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] griglia di confronto baseline vs augmented -> {path}")


def run_group(gradcam, files, group_name, transform, device, img_size, out_dir):
    """Genera la figura Grad-CAM per un gruppo e ritorna le metriche."""
    if not files:
        print(f"  [!] Nessuna immagine per il gruppo '{group_name}', salto.")
        return None

    n = len(files)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4))
    if n == 1:
        axes = axes.reshape(2, 1)

    borders = []
    for j, fpath in enumerate(files):
        disp = load_display_image(fpath, img_size)
        x = transform(Image.open(fpath).convert("RGB")).unsqueeze(0).to(device)
        cam, pred_idx, probs = gradcam(x)            # classe predetta
        borders.append(border_energy(cam, BORDER_FRAC))

        axes[0, j].imshow(disp)
        axes[0, j].set_title(os.path.basename(fpath)[:14], fontsize=7)
        axes[0, j].axis("off")

        axes[1, j].imshow(overlay_cam(disp, cam))
        axes[1, j].set_title(f"pred={CLASS_NAMES[pred_idx]}\np={probs[pred_idx]:.2f}", fontsize=8)
        axes[1, j].axis("off")

    axes[0, 0].set_ylabel("originale", fontsize=9)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)
    mean_border = float(np.mean(borders))
    fig.suptitle(f"Grad-CAM -- {group_name}   |   border-energy medio = {mean_border:.2f}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"gradcam_{group_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {group_name}: {n} img -> {path}  (border-energy medio {mean_border:.3f})")
    return {"group": group_name, "mean_border_energy": mean_border, "n": n}


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}\n  Grad-CAM -- ResNet augmented-75%  (device: {device})\n{'='*60}")

    # --- modello ---
    if not os.path.isfile(RESNET_CKPT):
        print(f"[ERRORE] Checkpoint non trovato: {RESNET_CKPT}")
        print("Imposta RESNET_CKPT sul tuo best_model_<tag>.pth del run 75%.")
        return
    model = load_model(RESNET_CKPT, device)
    print(f"  Modello augmented-75% caricato: {RESNET_CKPT}")

    # target layer = ultimo blocco convoluzionale
    gradcam = GradCAM(model, model.backbone.layer4)

    # --- cartelle immagini ---
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        print("[ERRORE] setup_dataset non riuscito."); return
    _, _, test_dir = res

    groups = {
        "real_normal":    os.path.join(test_dir, "NORMAL"),
        "real_pneumonia": os.path.join(test_dir, "PNEUMONIA"),
        "synth_normal":   os.path.join(SYNTHETIC_DIR, "NORMAL"),
    }

    transform = build_transform(RESNET_IMG_SIZE)
    summary = []
    for name, folder in groups.items():
        files = list_images(folder, N_PER_GROUP, SEED)
        r = run_group(gradcam, files, name, transform, device, RESNET_IMG_SIZE, OUT_DIR)
        if r:
            summary.append(r)

    # --- griglia di confronto Baseline vs Augmented-75% (stesse immagini) ---
    print(f"\n  Griglia di confronto Baseline vs Augmented-75%...")
    if os.path.isfile(BASELINE_CKPT):
        model_base = load_model(BASELINE_CKPT, device)
        gc_base = GradCAM(model_base, model_base.backbone.layer4)

        compare_items = []
        for name, folder in groups.items():
            for f in list_images(folder, N_COMPARE_PER_GROUP, SEED + 1):
                compare_items.append((f, name, GROUP_TRUE[name]))

        build_comparison_grid(compare_items, gc_base, gradcam,
                              transform, device, RESNET_IMG_SIZE, OUT_DIR)
        gc_base.remove()
    else:
        print(f"  [!] Checkpoint baseline non trovato: {BASELINE_CKPT} -- griglia saltata.")
        print(f"      Imposta BASELINE_CKPT sul best_model_<tag>.pth del run real-only.")

    gradcam.remove()

    # --- riepilogo metrica shortcut ---
    print(f"\n{'='*60}\n  RIEPILOGO border-energy (frazione attenzione sui bordi)\n{'='*60}")
    for r in summary:
        print(f"  {r['group']:<16}: {r['mean_border_energy']:.3f}   ({r['n']} img)")
    print("\n  Nota: border-energy alto = attenzione sui bordi/angoli (possibile shortcut).")
    print("  Confronta reale vs sintetico: se il sintetico ha border-energy molto")
    print("  diverso, il classificatore potrebbe sfruttare artefatti della GAN.")
    print(f"\n  Figure salvate in: {OUT_DIR}")


if __name__ == "__main__":
    main()
