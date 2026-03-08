"""
Main script che ricrea l'intera evaluation_pipeline.py usando i moduli separati.
"""

import torch
import os
import shutil
from dataset.loader import setup_dataset, get_dataloaders, get_gan_dataloader
from models.wgan import Generator, Critic, compute_gp
from train import train_resnet, train_wgangp
from eval import evaluate_on_test, generate_synthetic_images, plot_comparison

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Avvio pipeline su Device: {device}")

    # --- 1. SETUP DATASET ---
    zip_path = '/content/drive/MyDrive/ProgettoMLVM/chest_xray.zip'
    res = setup_dataset(zip_path, seed=42)
    if not res: return
    train_dir, val_dir, test_dir, n_train_n, n_train_p = res

    # --- 2. DATALOADERS ---
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir, img_size=128, batch_size=32)

    # Creazione cartella per i risultati finali (metriche e plot)
    results_dir = 'evaluation_results'
    os.makedirs(results_dir, exist_ok=True)

    # --- 3. PHASE 1: RESNET BASELINE ---
    model_p1, hist_p1, ckpt_p1 = train_resnet(
        train_loader, val_loader, device, epochs=5, lr=0.001, tag="Phase1")
    
    report_p1, cm_p1 = evaluate_on_test(
        model_p1, ckpt_p1, test_loader, classes, device, tag="Phase1", out_dir=results_dir)

    # --- 4. WGAN-GP TRAINING ---
    gan_loader, gan_classes = get_gan_dataloader(
        train_dir, img_size=128, batch_size=64)
    
    G = Generator(nz=100, n_class=2, nc=1, d=128).to(device)
    D = Critic(nc=1, n_class=2, d=128).to(device)
    
    # Cartella locale dove salvare i sample durante il training
    gan_samples_dir = 'gan_training_samples'

    # Allenamento GAN (modifica le epoche a 5 o 10 per un test veloce)
    G, ckpt_gan = train_wgangp(
        G, D, gan_loader, device, compute_gp, 
        epochs=100, lr=0.0001, n_critic=5, nz=100, n_class=2, out_dir=gan_samples_dir)

    # Copia i sample generati su Google Drive per poterli visualizzare
    drive_samples_dir = '/content/drive/MyDrive/ProgettoMLVM/gan_training_samples'
    if os.path.exists(drive_samples_dir): shutil.rmtree(drive_samples_dir)
    shutil.copytree(gan_samples_dir, drive_samples_dir)
    print(f"Sample GAN salvati in {drive_samples_dir}")

    # --- 5. AUGMENTATION ---
    num_gen_normal = n_train_p - n_train_n     # Colma il gap per NORMAL
    num_gen_pneumonia = 0                      # Nessuna per PNEUMONIA
    syn_dir = 'synthetic_images'
    
    print(f"\n{'='*50}\nBILANCIAMENTO DATASET\n{'='*50}")
    print(f"  Gap da colmare: {num_gen_normal} NORMAL sintetiche")
    generate_synthetic_images(G, num_gen_normal, num_gen_pneumonia, nz=100, n_class=2, device=device, syn_dir=syn_dir)

    # Crea cartella dataset augmented
    aug_dir = 'augmented_dataset'
    aug_train_dir = os.path.join(aug_dir, 'train')
    if os.path.exists(aug_dir): shutil.rmtree(aug_dir)
    shutil.copytree(train_dir, aug_train_dir)
    for cat in ['NORMAL', 'PNEUMONIA']:
        sc = os.path.join(syn_dir, cat)
        if os.path.exists(sc):
            for f in os.listdir(sc):
                shutil.copy(os.path.join(sc, f), os.path.join(aug_train_dir, cat, f))
    
    print(f"Rapporto Augmented: PNEUMONIA:{len(os.listdir(os.path.join(aug_train_dir, 'PNEUMONIA')))} / NORMAL:{len(os.listdir(os.path.join(aug_train_dir, 'NORMAL')))}")

    # --- 6. PHASE 3: RESNET AUGMENTED ---
    aug_train_loader, _, _, _ = get_dataloaders(
        aug_train_dir, val_dir, test_dir, img_size=128, batch_size=32)
    
    model_p3, hist_p3, ckpt_p3 = train_resnet(
        aug_train_loader, val_loader, device, epochs=5, lr=0.001, tag="Phase3")
    
    report_p3, cm_p3 = evaluate_on_test(
        model_p3, ckpt_p3, test_loader, classes, device, tag="Phase3", out_dir=results_dir)

    # --- 7. CONFRONTO FINALE ---
    plot_comparison(hist_p1, hist_p3, cm_p1, cm_p3, classes, report_p1, report_p3, out_dir=results_dir)

    # Copia tutti i risultati su Google Drive
    drive_results_dir = '/content/drive/MyDrive/ProgettoMLVM/evaluation_results'
    if os.path.exists(drive_results_dir): shutil.rmtree(drive_results_dir)
    shutil.copytree(results_dir, drive_results_dir)
    print(f"\n📊 Tutti i grafici e le metriche sono stati salvati su Drive in: {drive_results_dir}")

    print("\n Pipeline completata con successo!")

if __name__ == '__main__':
    main()
