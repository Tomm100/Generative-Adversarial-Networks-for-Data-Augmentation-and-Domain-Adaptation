"""
main_gradcam.py

Dai il path della cartella augmented con immagini reali e sintetiche.
Il codice:
  - pesca 5 immagini che iniziano con "syn_" (sintetiche)
  - pesca 5 immagini che NON iniziano con "syn_" (reali)
  - traccia la Grad-CAM su entrambi i gruppi
  - salva il plot in ./gradcam_comparison/

Cambia solo i 3 parametri qui sotto, poi lancia:
    python main_gradcam.py
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os, glob, sys

from models.resnet import ResNetClassifier

# ──────────────────────────────────────────────────────────────
#  CAMBIA QUI
# ──────────────────────────────────────────────────────────────

# Cartella che contiene sia immagini reali sia sintetiche (NORMAL o PNEUMONIA)
IMG_DIR = "./results/experiment_augmentation/augmented_dataset/train/NORMAL"

# Modello ResNet da usare
CKPT = "./results/experiment_augmentation/checkpoints/best_model_Exp_Phase3.pth"

# Normalizzazione usata quando il modello è stato addestrato: "imagenet" o "0.5"
NORM = "imagenet"

# (Opzionale) Cartella del dataset ORIGINALE per il confronto baseline
# Lascia stringa vuota "" per saltare il gruppo baseline
BASELINE_DIR = "./data/modified_dataset/test/NORMAL"

# ──────────────────────────────────────────────────────────────
#  NON TOCCARE SOTTO
# ──────────────────────────────────────────────────────────────

NUM_EACH   = 5
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
OUT_DIR    = "./gradcam_comparison"


class GradCAM:
    def __init__(self, model):
        self.model = model.eval()
        self.act = self.grad = None
        model.backbone.layer4.register_forward_hook(
            lambda m, i, o: setattr(self, 'act', o.detach()))
        model.backbone.layer4.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'grad', go[0].detach()))

    def run(self, x):
        out  = self.model(x)
        pred = out.argmax(1).item()
        prob = F.softmax(out, 1).detach().cpu()[0, pred].item()
        self.model.zero_grad()
        out[0, pred].backward()
        w   = self.grad.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((w * self.act).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max().clamp(min=1e-8))
        return cam.cpu().numpy(), pred, prob


def pick_images(folder, prefix_syn, n):
    """Ritorna (reali[:n], sintetiche[:n])."""
    all_files = sorted(
        glob.glob(os.path.join(folder, '*.jpeg')) +
        glob.glob(os.path.join(folder, '*.jpg')) +
        glob.glob(os.path.join(folder, '*.png'))
    )
    syn   = [f for f in all_files if os.path.basename(f).startswith(prefix_syn)][:n]
    real  = [f for f in all_files if not os.path.basename(f).startswith(prefix_syn)][:n]
    return real, syn


def process(paths, model, gc, tf, mean_t, std_t, device):
    results = []
    for p in paths:
        t   = tf(Image.open(p).convert('RGB'))
        cam, pred, conf = gc.run(t.unsqueeze(0).to(device))
        vis = (t * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
        results.append((vis, cam, pred, conf))
    return results


def plot_group(axes_row0, axes_row1, results, title, true_class_guess):
    for i, (vis, cam, pred, conf) in enumerate(results):
        color = 'green' if pred == true_class_guess else 'red'
        axes_row0[i].imshow(vis)
        axes_row0[i].axis('off')
        axes_row1[i].imshow(vis)
        axes_row1[i].imshow(cam, alpha=0.5, cmap='jet')
        axes_row1[i].set_title(f"{CLASS_NAMES[pred]} {conf:.0%}", fontsize=8, color=color)
        axes_row1[i].axis('off')
    axes_row0[0].set_ylabel(title, fontsize=10, fontweight='bold',
                             rotation=0, labelpad=70, va='center')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(IMG_DIR):
        print(f"ERRORE: cartella non trovata → {IMG_DIR}"); sys.exit(1)
    if not os.path.exists(CKPT):
        print(f"ERRORE: checkpoint non trovato → {CKPT}"); sys.exit(1)

    # Normalizzazione
    if NORM == "imagenet":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif NORM == "0.5":
        mean, std = [0.5]*3, [0.5]*3
    else:
        print(f"ERRORE: NORM deve essere 'imagenet' o '0.5'"); sys.exit(1)

    tf     = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(),
                                  transforms.Normalize(mean, std)])
    mean_t = torch.tensor(mean).view(3,1,1)
    std_t  = torch.tensor(std).view(3,1,1)

    # Carica modello
    model = ResNetClassifier(num_classes=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    gc = GradCAM(model)
    print(f"Modello caricato: {CKPT}")

    # Inferisci la classe dalla cartella (NORMAL o PNEUMONIA)
    folder_name = os.path.basename(IMG_DIR).upper()
    true_cls = 0 if "NORMAL" in folder_name else 1

    # ── Separa reali e sintetiche dalla cartella augmented ──
    real_paths, syn_paths = pick_images(IMG_DIR, prefix_syn="syn_", n=NUM_EACH)
    print(f"Reali trovate:      {len(real_paths)}")
    print(f"Sintetiche trovate: {len(syn_paths)}")

    # ── Baseline originale (opzionale) ──
    baseline_paths = []
    if BASELINE_DIR and os.path.exists(BASELINE_DIR):
        baseline_paths = sorted(
            glob.glob(os.path.join(BASELINE_DIR, '*.jpeg')) +
            glob.glob(os.path.join(BASELINE_DIR, '*.jpg')) +
            glob.glob(os.path.join(BASELINE_DIR, '*.png'))
        )[:NUM_EACH]
        print(f"Baseline trovate:   {len(baseline_paths)}")
    elif BASELINE_DIR:
        print(f"ATTENZIONE: BASELINE_DIR non trovato → {BASELINE_DIR}")

    # ── Calcola Grad-CAM ──
    base_res = process(baseline_paths, model, gc, tf, mean_t, std_t, device)
    real_res  = process(real_paths,     model, gc, tf, mean_t, std_t, device)
    syn_res   = process(syn_paths,      model, gc, tf, mean_t, std_t, device)

    # ── Costruisci la lista gruppi da plottare ──
    groups = []
    if base_res:
        groups.append(("BASELINE\n(test originale)", base_res))
    if real_res:
        groups.append(("REALI\n(augmented)", real_res))
    if syn_res:
        groups.append(("SINTETICHE", syn_res))

    if not groups:
        print("ERRORE: nessuna immagine trovata"); sys.exit(1)

    # ── Plot: 2 righe per gruppo × n colonne ──
    n_groups = len(groups)
    n_cols   = NUM_EACH
    n_rows   = n_groups * 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_groups * 5))
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    if n_rows == 2:
        axes = axes.reshape(2, n_cols)

    for g_idx, (label, results) in enumerate(groups):
        row0 = g_idx * 2
        row1 = g_idx * 2 + 1
        plot_group(axes[row0], axes[row1], results, label, true_cls)

    folder_label = os.path.basename(IMG_DIR)
    fig.suptitle(f"Grad-CAM — {folder_label}  |  Verde=corretto, Rosso=errore",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"gradcam_{folder_label}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot salvato: {out}")


if __name__ == "__main__":
    main()

