import torch
import os
import shutil
import random
import argparse
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    DATASET_DIR, RESULTS_DIR, METRICS_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_EPOCHS, RESNET_LR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_EPOCHS, RESNET_LR,
    GAN_NZ, GAN_N_CLASS, GAN_NC, SEED
)
from dataset.loader import setup_dataset, get_dataloaders
from models.sngan import SNGenerator
from train import train_resnet
from eval import evaluate_on_test, generate_synthetic_images
from utils.seed import set_seed

SNGAN_D = 128
ABLATION_DIR = os.path.join(RESULTS_DIR, "ablation_study")
FULL_SYNTH_DIR = os.path.join(ABLATION_DIR, "full_synthetic_pool")

def main():
    parser = argparse.ArgumentParser(description="Ablation Study sulle percentuali di Data Augmentation")
    parser.add_argument('--epoch', type=int, default=220, help="Epoca del Generator SNGAN da usare")
    parser.add_argument('--ckpt_dir', type=str, default="/content/drive/MyDrive/ProgettoMLVM/GAN_CHECKPOINTS_BACKUP", help="Path assoluto ai checkpoint su Drive")
    parser.add_argument('--drive_plot_dir', type=str, default="/content/drive/MyDrive/ProgettoMLVM/Ablation_Results", help="Path assoluto su Drive dove salvare i plot")
    args = parser.parse_args()

    target_epoch = args.epoch
    percentages = [0, 25, 50, 75, 100]

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\nAVVIO ABLATION STUDY - SNGAN (Epoca {target_epoch})\n{'='*60}")

    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name=f"Ablation_SNGAN_Ep{target_epoch}",
        config={"epoch_evaluated": target_epoch, "percentages": percentages}
    )

    # 1. SETUP DATASET
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res: return
    train_dir, val_dir, test_dir = res

    n_train_n = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    n_train_p = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    max_deficit = n_train_p - n_train_n
    print(f"Dataset Reale: {n_train_n} NORMAL | {n_train_p} PNEUMONIA (Deficit: {max_deficit})")

    # 2. GENERAZIONE DEL POOL SINTETICO (Solo una volta per velocità)
    print(f"\n[Fase A] Generazione del pool sintetico completo ({max_deficit} immagini)...")
    if os.path.exists(FULL_SYNTH_DIR): shutil.rmtree(FULL_SYNTH_DIR)
    os.makedirs(FULL_SYNTH_DIR, exist_ok=True)

    G = SNGenerator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=SNGAN_D).to(device)
    # Su Drive il file si chiama G_sngan_epoch_220.pth
    ckpt_path = os.path.join(args.ckpt_dir, f'G_sngan_epoch_{target_epoch}.pth')
    
    if not os.path.exists(ckpt_path):
        print(f"\nERRORE: Non trovo il file dei pesi su Drive: {ckpt_path}")
        return
        
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    generate_synthetic_images(
        G, num_gen_normal=max_deficit, num_gen_pneumonia=0,
        nz=GAN_NZ, n_class=GAN_N_CLASS, device=device,
        syn_dir=FULL_SYNTH_DIR)
    
    synth_pool_files = os.listdir(os.path.join(FULL_SYNTH_DIR, 'NORMAL'))
    # Mischiamo la lista per campionare randomicamente in seguito
    random.shuffle(synth_pool_files)

    # Dizionari per salvare i risultati
    results = {
        'pct': percentages,
        'acc': [],
        'f1_normal': [],
        'recall_normal': [],
        'f1_macro': []
    }

    # 3. CICLO DI ABLATION
    for p in percentages:
        print(f"\n{'*'*60}\n  TESTING: {p}% di Data Augmentation\n{'*'*60}")
        
        num_to_add = int(max_deficit * (p / 100.0))
        aug_dir = os.path.join(ABLATION_DIR, f"aug_dataset_{p}pct")
        
        # Pulisci e ricrea cartella
        if os.path.exists(aug_dir): shutil.rmtree(aug_dir)
        shutil.copytree(train_dir, aug_dir)
        
        # Aggiungi le immagini sintetiche
        if num_to_add > 0:
            target_normal_dir = os.path.join(aug_dir, 'NORMAL')
            files_to_copy = synth_pool_files[:num_to_add]
            for f in files_to_copy:
                shutil.copy(os.path.join(FULL_SYNTH_DIR, 'NORMAL', f), os.path.join(target_normal_dir, f))
        
        tot_normal = len(os.listdir(os.path.join(aug_dir, 'NORMAL')))
        tot_pneumo = len(os.listdir(os.path.join(aug_dir, 'PNEUMONIA')))
        print(f"  Composizione Training Set: {tot_normal} NORMAL ({num_to_add} sintetiche) | {tot_pneumo} PNEUMONIA")

        # DataLoader
        aug_loader, val_loader, test_loader, classes = get_dataloaders(
            aug_dir, val_dir, test_dir,
            img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

        # Training ResNet
        tag = f"Ablation_{p}pct"
        resnet_aug, _, ckpt_path = train_resnet(
            aug_loader, val_loader, device,
            epochs=RESNET_EPOCHS, lr=RESNET_LR, tag=tag)
        
        # Valutazione
        report, _ = evaluate_on_test(
            resnet_aug, ckpt_path, test_loader, classes, device,
            tag=tag, out_dir=METRICS_DIR)
        
        # Salva metriche
        acc = report['accuracy']
        f1_norm = report['NORMAL']['f1-score']
        rec_norm = report['NORMAL']['recall']
        f1_mac = report['macro avg']['f1-score']
        
        results['acc'].append(acc)
        results['f1_normal'].append(f1_norm)
        results['recall_normal'].append(rec_norm)
        results['f1_macro'].append(f1_mac)
        
        wandb.log({
            "Ablation/Percent_Synthetic": p,
            "Ablation/Accuracy": acc,
            "Ablation/NORMAL_F1": f1_norm,
            "Ablation/NORMAL_Recall": rec_norm,
            "Ablation/Macro_F1": f1_mac
        })

    # 4. PLOT RISULTATI FINALI
    print(f"\n{'='*60}\n  RISULTATI ABLATION STUDY\n{'='*60}")
    for i, p in enumerate(percentages):
        print(f"  {p:>3}% Syn | Acc: {results['acc'][i]:.4f} | F1-Norm: {results['f1_normal'][i]:.4f} | Rec-Norm: {results['recall_normal'][i]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(percentages, results['acc'], marker='o', label='Accuracy Totale', linewidth=2)
    ax.plot(percentages, results['f1_normal'], marker='s', label='F1-Score (NORMAL)', linewidth=2)
    ax.plot(percentages, results['recall_normal'], marker='^', label='Recall (NORMAL)', linewidth=2)
    ax.plot(percentages, results['f1_macro'], marker='d', label='Macro F1-Score', linewidth=2, linestyle='--')
    
    ax.set_xlabel('% di Immagini Sintetiche Aggiunte (rispetto al deficit)')
    ax.set_ylabel('Score sul Test Set Reale')
    ax.set_title(f'Ablation Study: Impatto Data Augmentation SNGAN (Epoca {target_epoch})')
    ax.set_xticks(percentages)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Salviamo il grafico sia in locale che sul Drive
    plot_path_local = os.path.join(ABLATION_DIR, 'ablation_curve.png')
    
    # Crea cartella su Drive per i plot se non esiste
    os.makedirs(args.drive_plot_dir, exist_ok=True)
    plot_path_drive = os.path.join(args.drive_plot_dir, f'ablation_curve_ep{target_epoch}.png')
    
    plt.savefig(plot_path_local, dpi=150)
    plt.savefig(plot_path_drive, dpi=150)
    plt.close(fig)
    
    wandb.log({"Ablation_Curve": wandb.Image(plot_path_local)})
    wandb.finish()
    
    print(f"\nGrafico salvato in locale: {plot_path_local}")
    print(f"Grafico salvato su Drive: {plot_path_drive}")
    print("Ablation Study completato con successo!")

if __name__ == '__main__':
    main()
