"""
Valutazione su TEST di checkpoint DANN gia' addestrati.

Gli passi una lista di checkpoint (pesi salvati da main_dann_cross_hospital.py)
e una lista di test set; per ogni combinazione stampa e salva:
  F1 Normal | F1 Pneumonia | Macro F1 | Accuracy | AUPRC-Pneumonia

Nota: i checkpoint sono state_dict di DANNSynth. La valutazione usa lambda=0
(solo il ramo di classificazione).

Uso:  python eval_dann_test.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, average_precision_score

from config import DATASET_DIR, RESNET_IMG_SIZE, METRICS_DIR
from models.dann_synth import DANNSynth

# ==============================================================================
# CONFIG -- MODIFICA QUI
# ==============================================================================
# UN solo checkpoint alla volta:
CKPT_PATH  = "./results/dann_cross_checkpoints/best_DANN_lam1.pth"   # <- il file di pesi da valutare
CKPT_LABEL = "DANN (lam1)"                                           # <- etichetta leggibile

# Test set su cui valutare: (nome, cartella con dentro NORMAL/ e PNEUMONIA/)
TEST_SETS = [
    ("TARGET (VinDr)",   "./data/vindr_target/test"),
    ("SOURCE (Kermany)", os.path.join(DATASET_DIR, "test")),
]

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]   # ImageFolder: N=0, P=1
BATCH = 32
OUT_TXT = os.path.join(METRICS_DIR, "dann_test_eval.txt")
# ==============================================================================


def make_test_loader(test_dir, img_size, batch):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(test_dir, tf)
    loader = DataLoader(ds, batch, shuffle=False, num_workers=2, pin_memory=True)
    return loader, ds


@torch.no_grad()
def evaluate(model, loader, device):
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
    try:
        auprc = average_precision_score((labels == 1).astype(int), probs)
    except Exception:
        auprc = float("nan")
    return rep, auprc


def load_model(ckpt_path, device):
    model = DANNSynth(num_classes=2, pretrained=False)   # architettura; i pesi arrivano dal ckpt
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(METRICS_DIR, exist_ok=True)
    print(f"\n{'='*82}\n  Valutazione DANN su TEST  (device: {device})\n{'='*82}")

    header = f"  {'Test set':<18}{'Config':<20}{'N F1':>8}{'P F1':>8}{'Macro':>8}{'Acc':>8}{'AUPRC-P':>10}"
    lines = [header, "  " + "-" * (len(header) - 2)]

    for set_name, test_dir in TEST_SETS:
        if not os.path.isdir(test_dir):
            lines.append(f"  [!] Cartella non trovata: {test_dir} -- salto '{set_name}'")
            continue
        loader, ds = make_test_loader(test_dir, RESNET_IMG_SIZE, BATCH)
        counts = np.bincount([l for _, l in ds.samples], minlength=2)
        lines.append(f"  # {set_name}: {counts[0]} NORMAL + {counts[1]} PNEUMONIA")

        if not os.path.isfile(CKPT_PATH):
            lines.append(f"  {set_name:<18}{CKPT_LABEL:<20}  [checkpoint mancante: {CKPT_PATH}]")
        else:
            model = load_model(CKPT_PATH, device)
            rep, auprc = evaluate(model, loader, device)
            lines.append(
                f"  {set_name:<18}{CKPT_LABEL:<20}"
                f"{rep[CLASS_NAMES[0]]['f1-score']:>8.3f}"
                f"{rep[CLASS_NAMES[1]]['f1-score']:>8.3f}"
                f"{rep['macro avg']['f1-score']:>8.3f}"
                f"{rep['accuracy']:>8.3f}"
                f"{auprc:>10.3f}"
            )
        lines.append("")

    table = "\n".join(lines)
    print("\n" + table)
    with open(OUT_TXT, "a") as f:                       # append: accumula i run fatti uno alla volta
        f.write(f"\n### CKPT: {CKPT_PATH}\n{table}\n")
    print(f"  Tabella accodata in: {OUT_TXT}")


if __name__ == "__main__":
    main()
