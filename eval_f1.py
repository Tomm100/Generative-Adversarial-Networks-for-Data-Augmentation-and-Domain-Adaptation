import argparse
import os
import torch
import wandb

from config import DATASET_DIR, RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_NUM_CLASSES
from dataset.loader import setup_dataset, get_dataloaders
from models.resnet import ResNetClassifier
from eval import evaluate_on_test

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

    print(f"Inizializzazione modello...")
    model = ResNetClassifier(num_classes=RESNET_NUM_CLASSES)
    
    # evaluate_on_test usa wandb.log internamente, quindi dobbiamo inizializzarlo
    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name="evaluation_wrapper"
    )

    print(f"Esecuzione valutazione tramite eval.py...")
    # evaluate_on_test carica internamente i pesi da ckpt_path e produce cm, report e log
    report_dict, cm = evaluate_on_test(
        model=model, 
        ckpt_path=ckpt_path, 
        test_loader=test_loader, 
        class_names=class_names, 
        device=device,
        tag="Evaluation", 
        out_dir='./results_provefinali'
    )
    
    wandb.finish()
    print("Valutazione completata. I plot della confusion matrix e report testuali sono stati salvati nella cartella ./results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrapper per valutare un classificatore tramite eval.py")
    parser.add_argument("--ckpt", type=str, required=True, help="Percorso al file dei pesi del classificatore (es. model.pth)")
    args = parser.parse_args()
    main(args.ckpt)
