import os
import shutil
import random
import torch
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURAZIONE PATH (Da Modificare)
# ==============================================================================
# Inserisci qui il path al file .pth del Generatore che vuoi testare
DUMMY_GAN_WEIGHTS_PATH = "/content/drive/MyDrive/ProgettoMLVM/results_SNGAN/sngan_checkpoints/G_epoch_220.pth"

# Se stai usando la SNGAN, decommenta l'import corretto
from models.sngan import SNGenerator as Generator
# ==============================================================================
# IMPORT FUNZIONI PROGETTO
# ==============================================================================
from dataset.loader import setup_dataset, get_dataloaders
from train import train_resnet
from models.resnet import ResNetClassifier # Come richiesto, anche se train_resnet lo usa internamente
from eval import evaluate_on_test, generate_synthetic_images
from utils.seed import set_seed
from config import (
    DATASET_DIR, RESULTS_DIR, METRICS_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_EPOCHS, RESNET_LR,
    GAN_NZ, GAN_N_CLASS, GAN_NC, SEED, SNGAN_D
)

def main():
    # 1. Inizializzazione Seed e Device
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Avvio Ablation Study su {device}")
    
    if not os.path.exists(DUMMY_GAN_WEIGHTS_PATH):
        print(f"ERRORE: Il file {DUMMY_GAN_WEIGHTS_PATH} non esiste.")
        return

    # 2. Setup Dataset
    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res: return
    train_dir, val_dir, test_dir = res

    # 3. Calcolo dello sbilanciamento (Quante immagini mancano a NORMAL?)
    n_train_n = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    n_train_p = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    max_deficit = n_train_p - n_train_n
    
    print(f"\nDataset originale: {n_train_n} NORMAL, {n_train_p} PNEUMONIA")
    print(f"Gap da colmare (100%): {max_deficit} immagini sintetiche")

    # 4. Generazione del Pool di immagini Sintetiche
    pool_dir = os.path.join(RESULTS_DIR, "ablation_synthetic_pool")
    if os.path.exists(pool_dir):
        shutil.rmtree(pool_dir)
    
    print("\nCaricamento Generatore e creazione pool sintetico...")
    # Inizializziamo il Generatore della SNGAN con i parametri centralizzati
    G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=SNGAN_D).to(device)
    G.load_state_dict(torch.load(DUMMY_GAN_WEIGHTS_PATH, map_location=device))
    
    # Generiamo il 100% delle immagini necessarie una sola volta
    generate_synthetic_images(
        G, num_gen_normal=max_deficit, num_gen_pneumonia=0,
        nz=GAN_NZ, n_class=GAN_N_CLASS, device=device, syn_dir=pool_dir
    )
    
    pool_normal_files = os.listdir(os.path.join(pool_dir, 'NORMAL'))
    # Mettiamo in ordine casuale le immagini generate per sicurezza
    random.shuffle(pool_normal_files) 

    # 5. Esecuzione Ablation Study
    percentages = [0, 25, 50, 75, 100]
    results_f1 = []
    
    ablation_base_dir = os.path.join(RESULTS_DIR, "ablation_datasets")
    
    for p in percentages:
        print(f"\n{'='*60}")
        print(f" ESECUZIONE ABLATION: {p}% di Augmentation")
        print(f"{'='*60}")
        
        # Resettiamo il seed prima di ogni ResNet per garantire massima riproducibilità
        set_seed(SEED)
        
        # Quante immagini aggiungere?
        num_to_add = int(max_deficit * (p / 100.0))
        
        # Creiamo una cartella temporanea per questo specifico dataset
        current_aug_dir = os.path.join(ablation_base_dir, f"dataset_{p}pct")
        if os.path.exists(current_aug_dir):
            shutil.rmtree(current_aug_dir)
            
        # Copiamo il dataset reale originale
        shutil.copytree(train_dir, current_aug_dir)
        
        # Aggiungiamo la percentuale di immagini sintetiche alla classe NORMAL
        if num_to_add > 0:
            dest_normal_dir = os.path.join(current_aug_dir, 'NORMAL')
            files_to_copy = pool_normal_files[:num_to_add]
            for f in files_to_copy:
                src_file = os.path.join(pool_dir, 'NORMAL', f)
                dst_file = os.path.join(dest_normal_dir, f)
                shutil.copy(src_file, dst_file)
                
        print(f"Dataset creato: {n_train_n + num_to_add} NORMAL, {n_train_p} PNEUMONIA")

        # Prepariamo i dataloader per questo run
        aug_loader, val_loader, test_loader, classes = get_dataloaders(
            current_aug_dir, val_dir, test_dir, 
            img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE
        )
        
        # Train ResNet (Fisso a 30 epoche per far emergere lo Shortcut Learning)
        tag = f"Ablation_{p}pct"
        resnet_model, _, ckpt_path = train_resnet(
            aug_loader, val_loader, device,
            epochs=30, lr=RESNET_LR, tag=tag
        )
        
        # Valutazione finale
        report, _ = evaluate_on_test(
            resnet_model, ckpt_path, test_loader, classes, device,
            tag=tag, out_dir=METRICS_DIR
        )
        
        # Salviamo la metrica F1 della classe NORMAL
        f1_normal = report['NORMAL']['f1-score']
        results_f1.append(f1_normal)
        print(f"Risultato {p}% -> F1-Score NORMAL: {f1_normal:.4f}")

    # 6. Plot dei risultati
    print("\nGenerazione Grafico Ablation...")
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, results_f1, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title("Ablation Study: Impatto Data Augmentation")
    plt.xlabel("% di Immagini Sintetiche Aggiunte (rispetto al gap)")
    plt.ylabel("F1-Score (Classe NORMAL)")
    plt.xticks(percentages)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(RESULTS_DIR, 'ablation_study_results.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Grafico salvato in: {plot_path}")
    print("\nAblation Study completato con successo!")

if __name__ == '__main__':
    main()
