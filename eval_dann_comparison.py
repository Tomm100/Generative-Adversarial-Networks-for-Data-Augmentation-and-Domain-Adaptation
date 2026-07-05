"""Confronto Macro F1: Pre-DANN vs Post-DANN sul test set REALE (Tabella 7).

Solo inferenza, nessun training, nessun wandb.
  - Pre-DANN : ResNetClassifier allenata su SNGAN 128 Complete al 75% (augmentation-only)
  - Post-DANN: DANNSynth (adattamento avversariale reale<->sintetico)

Entrambi valutati sullo stesso test set reale, con la stessa metrica (Macro F1).
"""

import torch
from sklearn.metrics import classification_report

from config import DATASET_DIR, RESNET_IMG_SIZE, RESNET_BATCH_SIZE, SEED
from dataset.loader import setup_dataset, get_dataloaders
from models.resnet import ResNetClassifier
from models.dann_synth import DANNSynth
from utils.seed import set_seed


# ==============================================================================
# MODIFICA QUI — percorsi dei due checkpoint
# ==============================================================================
# ResNet allenata su SNGAN 128 Complete al 75% (cella 75% di Ablation_study.py)
PRE_DANN_CKPT  = "/content/drive/MyDrive/ProgettoMLVM/.../best_model_Ablation_75pct.pth"
# Modello DANNSynth prodotto da main_dann_synth.py
POST_DANN_CKPT = "/content/drive/MyDrive/ProgettoMLVM/.../best_DANN_Synth.pth"
# ==============================================================================


@torch.no_grad()
def _infer_resnet(ckpt, test_loader, device):
    model = ResNetClassifier(num_classes=2).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    preds, labels = [], []
    for x, y in test_loader:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
        labels += y.tolist()
    return labels, preds


@torch.no_grad()
def _infer_dann(ckpt, test_loader, device):
    # pretrained=False: l'architettura e' identica, i pesi arrivano dal checkpoint
    model = DANNSynth(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    preds, labels = [], []
    for x, y in test_loader:
        class_out, _, _ = model(x.to(device), lambda_=0.0)   # solo label predictor
        preds += class_out.argmax(1).cpu().tolist()
        labels += y.tolist()
    return labels, preds


def _f1s(labels, preds, names):
    r = classification_report(labels, preds, target_names=names, output_dict=True)
    return r[names[0]]["f1-score"], r[names[1]]["f1-score"], r["macro avg"]["f1-score"]


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        return
    train_dir, val_dir, test_dir = res
    _, _, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    n_pre,  p_pre,  m_pre  = _f1s(*_infer_resnet(PRE_DANN_CKPT,  test_loader, device), classes)
    n_post, p_post, m_post = _f1s(*_infer_dann(POST_DANN_CKPT, test_loader, device), classes)

    c0, c1 = classes[0], classes[1]
    print(f"\n{'Config':<32}{c0+' F1':>12}{c1+' F1':>14}{'Macro F1':>12}")
    print("-" * 70)
    print(f"{'Pre-DANN (SNGAN Compl. 75%)':<32}{n_pre:>12.4f}{p_pre:>14.4f}{m_pre:>12.4f}")
    print(f"{'Post-DANN':<32}{n_post:>12.4f}{p_post:>14.4f}{m_post:>12.4f}")
    print("-" * 70)
    print(f"{'Delta':<32}{n_post-n_pre:>+12.4f}{p_post-p_pre:>+14.4f}{m_post-m_pre:>+12.4f}")


if __name__ == "__main__":
    main()
