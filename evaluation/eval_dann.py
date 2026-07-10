"""Valutazione di una DANN (DANNSynth) su un checkpoint dato via CLI.

Carica i pesi da --ckpt, esegue l'inferenza sul test set reale usando SOLO il
label predictor (lambda=0), e stampa il classification report + Macro F1
(e Accuracy). Solo inferenza, nessun training.

Uso (dalla ROOT del repo):
    python evaluation/eval_dann.py --ckpt /percorso/best_DANN_Synth.pth
    # con confusion matrix + log su W&B:
    python evaluation/eval_dann.py --ckpt X.pth --wandb
"""

import os
import sys
import argparse
import torch
from sklearn.metrics import classification_report

# Root del repo nel path: consente l'esecuzione da evaluation/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASET_DIR, RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_NUM_CLASSES, SEED
from dataset.loader import setup_dataset, get_dataloaders
from models.dann_synth import DANNSynth
from utils.seed import set_seed


@torch.no_grad()
def _infer(model, test_loader, device):
    """Inferenza DANN: usa solo il label predictor (lambda=0)."""
    model.eval()
    preds, labels = [], []
    for x, y in test_loader:
        logits, _, _ = model(x.to(device), lambda_=0.0)
        preds += logits.argmax(1).cpu().tolist()
        labels += y.tolist()
    return labels, preds


def main():
    parser = argparse.ArgumentParser(
        description="Macro F1 di un checkpoint DANN sul test set reale.")
    parser.add_argument("--ckpt", required=True, help="Percorso ai pesi DANN (.pth).")
    parser.add_argument("--wandb", action="store_true",
                        help="Salva confusion matrix + report e logga su W&B (riusa training/train_dann_synth.py).")
    parser.add_argument("--out", default=None,
                        help="Cartella output per il salvataggio ricco (default: METRICS_DIR).")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.ckpt):
        print(f"ERRORE: checkpoint non trovato: {args.ckpt}")
        return

    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        print("ERRORE: dataset non trovato.")
        return
    train_dir, val_dir, test_dir = res
    _, _, test_loader, class_names = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    # ── Percorso ricco (W&B + confusion matrix): riusa evaluate_dann_synth ──
    if args.wandb:
        import wandb
        from config import METRICS_DIR
        from training.train_dann_synth import evaluate_dann_synth
        out_dir = args.out or METRICS_DIR
        wandb.init(project="gan-chest-xray-augmentation",
                   entity="MachineLearningForVisionAndMultimedia",
                   name="eval_dann")
        model = DANNSynth(num_classes=RESNET_NUM_CLASSES, pretrained=False).to(device)
        report, _ = evaluate_dann_synth(model, args.ckpt, test_loader, class_names,
                                        device, tag="Eval_DANN", out_dir=out_dir)
        wandb.finish()
        print(f"\n  Macro F1: {report['macro avg']['f1-score']:.4f}")
        return

    # ── Percorso leggero (default): solo inferenza + print ──
    print(f"\nCaricamento DANN da: {args.ckpt}")
    model = DANNSynth(num_classes=RESNET_NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    labels, preds = _infer(model, test_loader, device)

    report_str  = classification_report(labels, preds, target_names=class_names)
    report_dict = classification_report(labels, preds, target_names=class_names, output_dict=True)
    print(f"\n{'='*50}\n  RISULTATI SUL TEST SET (DANN)\n{'='*50}")
    print(report_str)
    print(f"  Macro F1: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"  Accuracy: {report_dict['accuracy']:.4f}")


if __name__ == "__main__":
    main()
