import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import wandb
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import (
    CHECKPOINTS_DIR, RESNET_LR,
    GAN_BETA1, GAN_BETA2, GAN_D_WEIGHT_DECAY,
    GAN_EPSILON_PENALTY_COEFF, GAN_NUM_VIS_SAMPLES,
    GAN_DRIVE_BACKUP_EVERY, GAN_DRIVE_DIR,
)
from models.resnet import ResNetClassifier
from models.wgan import weights_init
from utils.visualization import save_gan_samples

def train_resnet(train_loader, val_loader, device, epochs=10, lr=0.001, tag="Phase1",
                 class_weights=None):
    """
    Allena ResNet18 (Feature Extractor / Classifier).
    Salva il checkpoint migliore in base alla Macro F1 sul validation set.

    Args:
        train_loader:   DataLoader per il training
        val_loader:     DataLoader per la validazione
        device:         torch.device
        epochs:         numero di epoche
        lr:             learning rate
        tag:            stringa identificativa per WandB e checkpoint
        class_weights:  Tensor 1D di forma [num_classes] con pesi per la loss.
                        Se None usa CrossEntropyLoss standard (nessun bilanciamento).
                        Esempio: torch.tensor([w_normal, w_pneumonia], device=device)
    """
    model = ResNetClassifier(num_classes=2)
    model = model.to(device)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"  [Loss] CrossEntropyLoss con class_weights={class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
    # Passiamo all'ottimizzatore solo i parametri che non sono congelati (requires_grad=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_macro_f1 = -1.0
    best_weights = None
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f'best_model_{tag}.pth')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': []}

    print(f"\n{'='*50}\nTraining {tag} ({epochs} epoche)\n{'='*50}")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        rl = 0.0
        
        pbar_train = tqdm(train_loader, desc=f"  Train Ep {epoch+1}/{epochs}", leave=False)
        for x, y in pbar_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            rl += loss.item()
            pbar_train.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_tl = rl / len(train_loader)

        # --- Validation ---
        model.eval()
        vl, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"  Val Ep {epoch+1}/{epochs}", leave=False)
            for x, y in pbar_val:
                x, y = x.to(device), y.to(device)
                out = model(x)
                vl += criterion(out, y).item()
                _, pred = torch.max(out, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        avg_vl = vl / len(val_loader)
        acc = 100 * correct / total
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        history['train_loss'].append(avg_tl)
        history['val_loss'].append(avg_vl)
        history['val_acc'].append(acc)
        history['val_macro_f1'].append(macro_f1)

        wandb.log({
            f"{tag}/train_loss": avg_tl,
            f"{tag}/val_loss": avg_vl,
            f"{tag}/val_acc": acc,
            f"{tag}/val_macro_f1": macro_f1,
            f"{tag}/epoch": epoch + 1
        })
        # Best model selection by Macro F1
        saved = ""
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            saved = " ← best"

        print(f"  Epoch {epoch+1}/{epochs} | TL: {avg_tl:.4f} | VL: {avg_vl:.4f} | "
              f"VA: {acc:.2f}% | Macro F1: {macro_f1:.4f}{saved}")

    # Carica i pesi migliori
    model.load_state_dict(best_weights)
    torch.save(best_weights, ckpt_path)
    print(f"  Best Macro F1: {best_macro_f1:.4f} (salvato in {ckpt_path})")

    return model, history, ckpt_path


def train_wgangp(G, D, gan_loader, device, compute_gp_fn,
                 epochs=100,
                 lr=0.0001, n_critic=5, nz=100, n_class=2,
                 beta1=GAN_BETA1, beta2=GAN_BETA2,
                 d_weight_decay=GAN_D_WEIGHT_DECAY,
                 save_every=20,
                 models_dir='gan_checkpoints',
                 samples_dir='gan_samples',
                 # --- Backup Google Drive ---
                 drive_backup_every=GAN_DRIVE_BACKUP_EVERY,
                 drive_dir=GAN_DRIVE_DIR):
    """
    Allena WGAN-GP.

    Args:
        G, D:               Generator e Critic (già su device)
        gan_loader:         DataLoader per il training GAN
        device:             torch device
        compute_gp_fn:      funzione gradient penalty
        epochs:             numero totale di epoche
        lr:                 learning rate base
        n_critic:           rapporto aggiornamenti critic/generator
        nz, n_class:        config GAN
        save_every:         salva checkpoint ogni N epoche
        models_dir:         cartella checkpoint G/D
        samples_dir:        cartella sample visivi
        drive_backup_every: frequenza backup drive
        drive_dir:          path per backup drive

    Returns:
        (G, g_final_path)
    """
    import os
    import shutil
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)



    # --- Inizializzazione pesi ---
    G.apply(weights_init)
    D.apply(weights_init)

    # --- Ottimizzatori + LR Scheduler ---
    G_opt = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    D_opt = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=d_weight_decay)

    # Decadimento lineare del learning rate verso 0 (come da paper WGAN-GP)
    G_scheduler = optim.lr_scheduler.LinearLR(G_opt, start_factor=1.0, end_factor=0.0, total_iters=epochs)
    D_scheduler = optim.lr_scheduler.LinearLR(D_opt, start_factor=1.0, end_factor=0.0, total_iters=epochs)

    # --- (RESUME — disattivato per ora) ---
    # Per riattivare il resume, aggiungere i parametri start_epoch, g_ckpt_path,
    # d_ckpt_path alla signature e decommentare il blocco seguente.
    # Salvare anche optimizer state_dict nei checkpoint per preservare i momenti di Adam.
    #
    # if g_ckpt_path and os.path.exists(g_ckpt_path):
    #     ckpt = torch.load(g_ckpt_path, map_location=device)
    #     G.load_state_dict(ckpt['model_state'])
    #     G_opt.load_state_dict(ckpt['optimizer_state'])
    # if d_ckpt_path and os.path.exists(d_ckpt_path):
    #     ckpt = torch.load(d_ckpt_path, map_location=device)
    #     D.load_state_dict(ckpt['model_state'])
    #     D_opt.load_state_dict(ckpt['optimizer_state'])

    # --- Label condizionali ---
    img_size = gan_loader.dataset[0][0].shape[1]
    onehot = torch.eye(n_class).view(n_class, n_class, 1, 1).to(device)
    fill = torch.zeros([n_class, n_class, img_size, img_size]).to(device)
    for i in range(n_class):
        fill[i, i, :, :] = 1

    # --- Fixed noise per visualizzare evoluzione ---
    num_vis = GAN_NUM_VIS_SAMPLES
    fixed_z = torch.randn(num_vis * 2, nz, 1, 1).to(device)
    fixed_labels = torch.cat([
        onehot[torch.zeros(num_vis, dtype=torch.long).to(device)],
        onehot[torch.ones(num_vis, dtype=torch.long).to(device)]
    ])



    # --- Training loop ---
    print(f"\nTraining WGAN-GP: {epochs} epoche")
    print(f"  LR: {lr} (LinearLR verso 0), n_critic={n_critic}")


    gan_start = time.time()

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        gp_losses, eps_losses = [], []

        pbar_gan = tqdm(gan_loader, desc=f"  Epoch {epoch}/{epochs}", leave=False)
        for batch_idx, (x_, y_) in enumerate(pbar_gan):
            mb = x_.size(0)
            x_, y_ = x_.to(device), y_.to(device)
            y_fill = fill[y_]

            # --- Train Critic ---
            D.zero_grad()
            D_real = D(x_, y_fill).squeeze().mean()

            z = torch.randn(mb, nz, 1, 1).to(device)
            # CRITICAL FIX: Genera i fake con la stessa label delle immagini reali
            # per non corrompere la matematica della Gradient Penalty
            y_gen_critic = y_
            fake = G(z, onehot[y_gen_critic])
            D_fake = D(fake.detach(), y_fill).squeeze().mean()

            gp, _ = compute_gp_fn(D, x_, fake.detach(), y_fill, y_fill, device)
            # Epsilon drift penalty: evita che i logit del Critic divergano
            epsilon_penalty = GAN_EPSILON_PENALTY_COEFF * (D_real ** 2).mean()
            d_loss = D_fake - D_real + gp + epsilon_penalty
            d_loss.backward()
            D_opt.step()
            d_losses.append(d_loss.item())
            gp_losses.append(gp.item())
            eps_losses.append(epsilon_penalty.item())

            # --- Train Generator ---
            if (batch_idx + 1) % n_critic == 0:
                G.zero_grad()
                z = torch.randn(mb, nz, 1, 1).to(device)
                
                # Approccio simil-BAGAN: Forza il batch del Generatore a essere esattamente 50/50
                half_mb = mb // 2
                zeros = torch.zeros(half_mb, dtype=torch.long)
                ones = torch.ones(mb - half_mb, dtype=torch.long)
                y_gen_unshuffled = torch.cat([zeros, ones])
                y_gen = y_gen_unshuffled[torch.randperm(mb)].to(device)
                
                g_loss = -D(G(z, onehot[y_gen]), fill[y_gen]).squeeze().mean()
                g_loss.backward()
                G_opt.step()
                g_losses.append(g_loss.item())
            
            # Aggiorna progress bar con valori intermedi
            pbar_gan.set_postfix({'D': f"{d_loss.item():.2f}", 'G': f"{g_losses[-1] if g_losses else 0.0:.2f}"})

        # --- LR Scheduler step (dopo ogni epoca) ---
        old_lr = G_opt.param_groups[0]['lr']
        G_scheduler.step()
        D_scheduler.step()
        new_lr = G_opt.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"  [LR decay] Epoca {epoch}: LR {old_lr:.8f} -> {new_lr:.8f}")

        # --- Log + Sample ---
        elapsed = (time.time() - gan_start) / 60
        eta = (elapsed / epoch) * (epochs - epoch)

        # Calcola metriche epoch-level (sempre, per log WandB completo)
        w_dist = -np.mean(d_losses) if d_losses else 0
        g_avg  = np.mean(g_losses)  if g_losses  else 0
        d_avg  = np.mean(d_losses)  if d_losses  else 0
        gp_avg = np.mean(gp_losses) if gp_losses else 0
        eps_avg = np.mean(eps_losses) if eps_losses else 0

        
        log_dict = {
            "GAN_Training/Wasserstein_Dist": w_dist,
            "GAN_Training/Generator_Loss":   g_avg,
            "GAN_Training/Critic_Loss":      d_avg,
            "GAN_Training/Gradient_Penalty": gp_avg,
            "GAN_Training/Epsilon_Penalty":  eps_avg,
            "GAN_Training/LR_Generator":     new_lr,
            "GAN_Training/Epoch":            epoch
        }

        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{epoch}/{epochs}] W_dist: {w_dist:.1f} | "
                  f"G_loss: {g_avg:.1f} | Time: {elapsed:.1f}m | ETA: {eta:.1f}m")
            save_gan_samples(G, fixed_z, fixed_labels, epoch, samples_dir, num_vis)
            sample_img_path = os.path.join(samples_dir, f'samples_epoch_{epoch:03d}.png')
            if os.path.exists(sample_img_path):
                log_dict["GAN_Samples/Generated_Images"] = wandb.Image(sample_img_path, caption=f"Epoch {epoch}")

        wandb.log(log_dict)

        # --- Checkpoint locale (ogni save_every epoche) ---
        if epoch % save_every == 0 or epoch == epochs:
            g_path = os.path.join(models_dir, f'G_epoch_{epoch}.pth')
            d_path = os.path.join(models_dir, f'D_epoch_{epoch}.pth')
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            print(f"  [Checkpoint] G + D salvati localmente (ep.{epoch})")

            # --- Backup su Google Drive (ogni drive_backup_every epoche) ---
            if (drive_dir and drive_backup_every > 0
                    and epoch % drive_backup_every == 0):
                if os.path.isdir(os.path.dirname(drive_dir)) or os.path.exists(drive_dir):
                    import shutil
                    os.makedirs(drive_dir, exist_ok=True)
                    shutil.copy(g_path, os.path.join(drive_dir, f'G_epoch_{epoch}.pth'))
                    shutil.copy(d_path, os.path.join(drive_dir, f'D_epoch_{epoch}.pth'))
                    print(f"  [Drive Backup] G + D copiati su Drive (ep.{epoch}) → {drive_dir}")
                    wandb.log({"GAN_Training/Drive_Backup_Epoch": epoch})
                else:
                    print(f"  [Drive Backup] ⚠️  Drive non montato, skip (ep.{epoch})")



    # --- Fine training ---
    gan_time = (time.time() - gan_start) / 60
    g_final = os.path.join(models_dir, f'G_epoch_{epochs}.pth')
    print(f"\nGAN training completato in {gan_time:.1f} minuti!")
    print(f"  Checkpoint finali: {models_dir}/G_epoch_{epochs}.pth")

    return G, g_final







def train_sngan(
    G, D, gan_loader, device,
    nz=100, n_class=2,
    lr=1e-4,
    beta1=0.0, beta2=0.9,
    epochs=300, n_critic=1,
    samples_dir='./results/sngan_samples',
    models_dir='./results/sngan_checkpoints',
    save_every=10,
    drive_dir=None,
    drive_backup_every=50,
):
    """
    Training loop SNGAN con Hinge Loss.

    Hinge Loss:
      D_loss = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]  → PatchGAN
      G_loss = -E[D(fake)]

    Returns:
        (G, g_final_path)
    """
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)



    # Inizializzazione pesi (solo Generator — Critic gestito da SN)
    from models.sngan import weights_init
    G.apply(weights_init)
    # NON applico weights_init al Critic: la SN gestisce già la scala dei pesi

    # Stesso LR per G e D (niente TTUR — lezione imparata dalla SAGAN)
    G_opt = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    D_opt = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

    # Linear decay verso 0 (come nella WGAN-GP)
    G_sched = optim.lr_scheduler.LinearLR(G_opt, start_factor=1.0, end_factor=0.0, total_iters=epochs)
    D_sched = optim.lr_scheduler.LinearLR(D_opt, start_factor=1.0, end_factor=0.0, total_iters=epochs)

    # Label condizionali (stessa struttura della WGAN-GP)
    img_size = gan_loader.dataset[0][0].shape[1]  # 128
    onehot = torch.eye(n_class).view(n_class, n_class, 1, 1).to(device)
    fill = torch.zeros([n_class, n_class, img_size, img_size]).to(device)
    for i in range(n_class):
        fill[i, i, :, :] = 1

    # Fixed noise per visualizzazione
    num_vis = GAN_NUM_VIS_SAMPLES
    fixed_z = torch.randn(num_vis * 2, nz, 1, 1).to(device)
    fixed_labels = torch.cat([
        onehot[torch.zeros(num_vis, dtype=torch.long).to(device)],
        onehot[torch.ones(num_vis, dtype=torch.long).to(device)]
    ])



    print(f"\nTraining SNGAN: {epochs} epoche (Hinge Loss + Spectral Norm)")
    print(f"  LR: {lr}, n_critic={n_critic}")


    gan_start = time.time()

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []

        pbar = tqdm(gan_loader, desc=f"  Epoch {epoch}/{epochs}", leave=False)
        for batch_idx, (x_, y_) in enumerate(pbar):
            mb = x_.size(0)
            x_, y_ = x_.to(device), y_.to(device)
            y_fill = fill[y_]

            # ═══════════════════════════════════════════
            # Train Critic (Hinge Loss su PatchGAN output)
            # ═══════════════════════════════════════════
            D.zero_grad()

            # Real: max(0, 1 - D(real)) → media sulla griglia PatchGAN
            d_real = D(x_, y_fill)  # [B, 1, 16, 16]
            loss_real = torch.relu(1.0 - d_real).mean()

            # Fake
            z = torch.randn(mb, nz, 1, 1).to(device)
            y_gen_critic = y_
            fake = G(z, onehot[y_gen_critic])
            d_fake = D(fake.detach(), y_fill)
            loss_fake = torch.relu(1.0 + d_fake).mean()

            d_loss = loss_real + loss_fake
            d_loss.backward()
            D_opt.step()
            d_losses.append(d_loss.item())

            # ═══════════════════════════════════════════
            # Train Generator
            # ═══════════════════════════════════════════
            if (batch_idx + 1) % n_critic == 0:
                G.zero_grad()
                z = torch.randn(mb, nz, 1, 1).to(device)

                # 50/50 bilanciamento label
                half = mb // 2
                y_gen = torch.cat([
                    torch.zeros(half, dtype=torch.long),
                    torch.ones(mb - half, dtype=torch.long)
                ])[torch.randperm(mb)].to(device)

                g_out = D(G(z, onehot[y_gen]), fill[y_gen])
                g_loss = -g_out.mean()
                g_loss.backward()
                G_opt.step()
                g_losses.append(g_loss.item())

            pbar.set_postfix({
                'D': f"{d_loss.item():.3f}",
                'G': f"{g_losses[-1] if g_losses else 0:.3f}"
            })

        # LR Scheduler
        G_sched.step()
        D_sched.step()

        # Metriche
        elapsed = (time.time() - gan_start) / 60
        eta = (elapsed / epoch) * (epochs - epoch)
        d_avg = np.mean(d_losses)
        g_avg = np.mean(g_losses) if g_losses else 0
        new_lr = G_opt.param_groups[0]['lr']

        log_dict = {
            "SNGAN/Discriminator_Loss": d_avg,
            "SNGAN/Generator_Loss":     g_avg,
            "SNGAN/LR":                 new_lr,
            "SNGAN/Epoch":              epoch,
        }

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{epoch}/{epochs}] D: {d_avg:.3f} | G: {g_avg:.3f} | "
                  f"Time: {elapsed:.1f}m | ETA: {eta:.1f}m")
            save_gan_samples(G, fixed_z, fixed_labels, epoch, samples_dir, num_vis)
            sample_path = os.path.join(samples_dir, f'samples_epoch_{epoch:03d}.png')
            if os.path.exists(sample_path):
                log_dict["SNGAN/Generated_Images"] = wandb.Image(
                    sample_path, caption=f"Epoch {epoch}")

        wandb.log(log_dict)

        # Checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            g_path = os.path.join(models_dir, f'G_epoch_{epoch}.pth')
            d_path = os.path.join(models_dir, f'D_epoch_{epoch}.pth')
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            print(f"  [Checkpoint] G + D salvati (ep.{epoch})")

            # Drive backup
            if (drive_dir and drive_backup_every > 0
                    and epoch % drive_backup_every == 0):
                if os.path.isdir(os.path.dirname(drive_dir)) or os.path.exists(drive_dir):
                    import shutil
                    os.makedirs(drive_dir, exist_ok=True)
                    shutil.copy(g_path, os.path.join(drive_dir, f'G_sngan_epoch_{epoch}.pth'))
                    shutil.copy(d_path, os.path.join(drive_dir, f'D_sngan_epoch_{epoch}.pth'))
                    print(f"  [Drive Backup] ep.{epoch} → {drive_dir}")



    # Fine training
    gan_time = (time.time() - gan_start) / 60
    g_final = os.path.join(models_dir, f'G_epoch_{epochs}.pth')
    print(f"\nSNGAN training completato in {gan_time:.1f} minuti!")

    return G, g_final



