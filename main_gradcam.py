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

    # Separa reali e sintetiche
    real_paths, syn_paths = pick_images(IMG_DIR, prefix_syn="syn_", n=NUM_EACH)
    print(f"Reali trovate:      {len(real_paths)}")
    print(f"Sintetiche trovate: {len(syn_paths)}")

    if not real_paths and not syn_paths:
        print("ERRORE: nessuna immagine trovata"); sys.exit(1)

    # Inferisci la classe dalla cartella (NORMAL o PNEUMONIA)
    folder_name = os.path.basename(IMG_DIR).upper()
    true_cls = 0 if "NORMAL" in folder_name else 1

    # Calcola Grad-CAM
    real_res = process(real_paths, model, gc, tf, mean_t, std_t, device)
    syn_res  = process(syn_paths,  model, gc, tf, mean_t, std_t, device)

    # Plot: 4 righe × max(5,5) colonne
    # riga 0: reali originali | riga 1: reali gradcam
    # riga 2: sintetiche originali | riga 3: sintetiche gradcam
    n = max(len(real_res), len(syn_res), 1)
    fig, axes = plt.subplots(4, n, figsize=(n * 3, 10))
    if n == 1:
        axes = axes.reshape(4, 1)

    if real_res:
        plot_group(axes[0], axes[1], real_res, "REALI", true_cls)
    else:
        for ax in list(axes[0]) + list(axes[1]):
            ax.set_visible(False)

    if syn_res:
        plot_group(axes[2], axes[3], syn_res, "SINTETICHE", true_cls)
    else:
        for ax in list(axes[2]) + list(axes[3]):
            ax.set_visible(False)

    # Etichette righe
    for ax, lbl in zip([axes[0,0], axes[1,0], axes[2,0], axes[3,0]],
                       ["img", "cam", "img", "cam"]):
        ax.set_ylabel(ax.get_ylabel() or lbl, fontsize=8)

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
