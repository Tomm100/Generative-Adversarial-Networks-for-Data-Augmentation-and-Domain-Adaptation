import argparse
import os
import torch
from sklearn.metrics import f1_score, classification_report

from config import DATASET_DIR, RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_NUM_CLASSES
from dataset.loader import setup_dataset, get_dataloaders
from models.resnet import ResNetClassifier

def main(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizzato: {device}")

    # Impostazione dataset
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res:
        print("ERRORE: dataset non trovato.")
        return
    train_dir, val_dir, test_dir = res

    # Recupero i dataloader per il test set
    _, _, test_loader, class_names = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
    )

    if not os.path.exists(ckpt_path):
        print(f"ERRORE: file dei pesi non trovato ({ckpt_path})")
        return

    print(f"Caricamento modello da: {ckpt_path}")
    model = ResNetClassifier(num_classes=RESNET_NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Valutazione sul test set in corso...")
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, pred = torch.max(logits, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*50)
    print("RISULTATI F1 SCORE")
    print("="*50)
    print(f"Macro F1 Score:    {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    
    print("\nReport Completo per Classe:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcola F1 Score di un classificatore addestrato")
    parser.add_argument("--ckpt", type=str, required=True, help="Percorso al file dei pesi del classificatore (es. model.pth)")
    args = parser.parse_args()
    main(args.ckpt)
