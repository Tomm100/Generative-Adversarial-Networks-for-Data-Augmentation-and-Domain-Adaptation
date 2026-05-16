import torch
import os
import shutil
import argparse
from config import (
    DATASET_DIR, RESULTS_DIR, METRICS_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_EPOCHS, RESNET_LR,
    GAN_NZ, GAN_N_CLASS, GAN_NC, SEED
)
from dataset.loader import setup_dataset, get_dataloaders
from models.sngan import SNGenerator
from train import train_resnet
from eval import evaluate_on_test, generate_synthetic_images
from utils.seed import set_seed

# Variabili locali per la SNGAN (d=128 come impostato in main_sngan.py)
SNGAN_D = 128
SNGAN_CKPT_DIR = os.path.join(RESULTS_DIR, "sngan_checkpoints")
SNGAN_SYNTH_DIR = os.path.join(RESULTS_DIR, "sngan_synthetic_images_eval")
SNGAN_AUG_DIR = os.path.join(RESULTS_DIR, "sngan_augmented_dataset_eval")

def main():
    parser = argparse.ArgumentParser(description="Valuta una specifica epoca SNGAN (es. 220)")
    parser.add_argument('--epoch', type=int, default=220, help="Epoca del checkpoint Generator da caricare")
    args = parser.parse_args()

    target_epoch = args.epoch

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Avvio Valutazione Specifica SNGAN (Epoca {target_epoch}) su Device: {device}")

    # 1. SETUP DATASET
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res: return
    train_dir, val_dir, test_dir = res

    n_train_n = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    n_train_p = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    num_gen_normal = n_train_p - n_train_n
    num_gen_pneumonia = 0

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: BASELINE RESNET
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  PHASE 1: Baseline ResNet (solo dati reali)\n{'='*60}")
    resnet_model, _, ckpt_p1 = train_resnet(
        train_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Phase1_Eval")
    report_p1, _ = evaluate_on_test(
        resnet_model, ckpt_p1, test_loader, classes, device,
        tag="Phase1_Eval", out_dir=METRICS_DIR)

    # ══════════════════════════════════════════════════════════════
    # PREPARAZIONE GENERATORE SNGAN
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  PHASE 2: Caricamento SNGAN Epoca {target_epoch}\n{'='*60}")
    
    G = SNGenerator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=SNGAN_D).to(device)
    ckpt_path = os.path.join(SNGAN_CKPT_DIR, f'G_epoch_{target_epoch}.pth')
    
    if not os.path.exists(ckpt_path):
        print(f"ERRORE: Impossibile trovare il checkpoint: {ckpt_path}")
        print("Assicurati di aver allenato la SNGAN fino a quell'epoca e di non aver spostato i file.")
        return

    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"  Checkpoint G_epoch_{target_epoch}.pth caricato con successo!")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: RESNET SU DATASET AUGMENTED
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}\n  PHASE 3: Generazione e Training Augmented\n{'='*60}")

    # Pulizia vecchie cartelle
    if os.path.exists(SNGAN_SYNTH_DIR): shutil.rmtree(SNGAN_SYNTH_DIR)
    if os.path.exists(SNGAN_AUG_DIR): shutil.rmtree(SNGAN_AUG_DIR)

    # Genera immagini
    generate_synthetic_images(
        G, num_gen_normal, num_gen_pneumonia,
        nz=GAN_NZ, n_class=GAN_N_CLASS, device=device,
        syn_dir=SNGAN_SYNTH_DIR)

    # Costruisci dataset augmented
    aug_train_dir = os.path.join(SNGAN_AUG_DIR, 'train')
    shutil.copytree(train_dir, aug_train_dir)
    for cat in ['NORMAL', 'PNEUMONIA']:
        syn_cat = os.path.join(SNGAN_SYNTH_DIR, cat)
        if os.path.exists(syn_cat):
            for f in os.listdir(syn_cat):
                shutil.copy(os.path.join(syn_cat, f), os.path.join(aug_train_dir, cat, f))

    aug_loader, _, _, _ = get_dataloaders(
        aug_train_dir, val_dir, test_dir,
        img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    # Allena ResNet su Augmented
    resnet_aug, _, ckpt_p3 = train_resnet(
        aug_loader, val_loader, device,
        epochs=RESNET_EPOCHS, lr=RESNET_LR, tag=f"Phase3_Eval_ep{target_epoch}")
    report_p3, _ = evaluate_on_test(
        resnet_aug, ckpt_p3, test_loader, classes, device,
        tag=f"Phase3_Eval_ep{target_epoch}", out_dir=METRICS_DIR)

    # ══════════════════════════════════════════════════════════════
    # CONFRONTO FINALE
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  CONFRONTO: Baseline vs SNGAN (Epoca {target_epoch} — FID Lowest)")
    print(f"{'='*60}")

    for cls in classes:
        for m in ['precision', 'recall', 'f1-score']:
            v1 = report_p1[cls][m]
            v3 = report_p3[cls][m]
            d  = v3 - v1
            print(f"  {cls} {m:12s}: {v1:.4f} → {v3:.4f}  ({'↑' if d > 0 else '↓'} {abs(d):.4f})")

    acc1 = report_p1['accuracy']
    acc3 = report_p3['accuracy']
    print(f"\n  Overall Acc: {acc1:.4f} → {acc3:.4f}  ({'↑' if acc3 > acc1 else '↓'} {abs(acc3 - acc1):.4f})")
    
    print("\n  Valutazione completata. Confronto salvato anche su WandB (se attivo).")

if __name__ == '__main__':
    main()
