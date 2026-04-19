import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import wandb
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import CHECKPOINTS_DIR
from models.resnet import ResNetClassifier
from utils.visualization import save_gan_samples

def train_resnet(train_loader, val_loader, device, epochs=10, lr=0.001, tag="Phase1"):
    """
    Allena ResNet18 (Feature Extractor / Classifier).
    Salva il checkpoint migliore in base alla Macro F1 sul validation set.
    """
    model = ResNetClassifier(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
                 lr_milestones=None, lr_gamma=0.2,
                 save_every=20,
                 models_dir='gan_checkpoints',
                 samples_dir='gan_samples',
                 # --- Parametri validazione periodica ---
                 validate_every=15,
                 train_dir=None, val_dir=None, test_dir=None,
                 num_gen_normal=0, num_gen_pneumonia=0,
                 resnet_img_size=128, resnet_batch_size=32, resnet_epochs=5,
                 augmented_dir='./augmented_dataset'):
    """
    Allena WGAN-GP con validazione periodica.

    Ogni `validate_every` epoche:
      1. Genera immagini sintetiche
      2. Crea dataset augmented (train originale + sintetiche)
      3. Allena ResNet veloce e misura Macro F1 su val set
      4. Se è il miglior F1 → conserva il dataset augmented

    Args:
        G, D:               Generator e Critic (già su device)
        gan_loader:         DataLoader per il training GAN
        device:             torch device
        compute_gp_fn:      funzione gradient penalty
        epochs:             numero totale di epoche
        lr:                 learning rate base
        n_critic:           rapporto aggiornamenti critic/generator
        nz, n_class:        config GAN
        lr_milestones:      epoche in cui applicare il decay (default: [60, 80])
        lr_gamma:           fattore decay (default: 0.2)
        save_every:         salva checkpoint ogni N epoche
        models_dir:         cartella checkpoint G/D
        samples_dir:        cartella sample visivi
        validate_every:     frequenza validazione (0 = disabilitata)
        train_dir:          path train set originale (per augmentation)
        val_dir, test_dir:  path val/test set (per DataLoader ResNet)
        num_gen_normal/pneumonia: quante sintetiche generare per classe
        resnet_img_size/batch_size/epochs: config ResNet per validazione
        augmented_dir:      dove salvare il miglior dataset augmented

    Returns:
        (G, g_final_path, best_val_epoch, val_history)
        - best_val_epoch: epoca con miglior Macro F1 (0 se nessuna validazione)
        - val_history: lista di dict {epoch, macro_f1, accuracy}
    """
    import os
    import shutil
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    if lr_milestones is None:
        lr_milestones = [60, 80]

    # Controlla se la validazione periodica è attivabile
    val_enabled = (validate_every > 0 and train_dir is not None
                   and val_dir is not None)
    if not val_enabled and validate_every > 0:
        print("  ⚠️  Validazione periodica disabilitata: train_dir o val_dir non forniti")

    # --- Inizializzazione pesi ---
    G.weight_init(0.0, 0.02)
    D.weight_init(0.0, 0.02)

    # --- Ottimizzatori + LR Scheduler ---
    G_opt = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    D_opt = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

    G_scheduler = optim.lr_scheduler.MultiStepLR(G_opt, milestones=lr_milestones, gamma=lr_gamma)
    D_scheduler = optim.lr_scheduler.MultiStepLR(D_opt, milestones=lr_milestones, gamma=lr_gamma)

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
    num_vis = 6
    fixed_z = torch.randn(num_vis * 2, nz, 1, 1).to(device)
    fixed_labels = torch.cat([
        onehot[torch.zeros(num_vis, dtype=torch.long).to(device)],
        onehot[torch.ones(num_vis, dtype=torch.long).to(device)]
    ])

    # --- Validation tracking ---
    val_history = []
    best_val_f1 = -1.0
    best_val_epoch = 0

    # --- Training loop ---
    print(f"\nTraining WGAN-GP: {epochs} epoche")
    print(f"  LR: {lr}, n_critic={n_critic}")
    print(f"  LR decay: x{lr_gamma} alle epoche {lr_milestones}")
    if val_enabled:
        print(f"  Validazione ogni {validate_every} epoche "
              f"(ResNet {resnet_epochs} ep.)")

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
            y_gen = torch.randint(0, n_class, (mb,)).to(device)
            fake = G(z, onehot[y_gen])
            D_fake = D(fake.detach(), fill[y_gen]).squeeze().mean()

            gp, _ = compute_gp_fn(D, x_, fake.detach(), y_fill, fill[y_gen], device)
            # Epsilon drift penalty: evita che i logit del Critic divergano
            epsilon_penalty = 1e-3 * (D_real ** 2).mean()
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
                y_gen = torch.randint(0, n_class, (mb,)).to(device)
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

        if epoch % 5 == 0 or epoch == 1:
            w_dist = -np.mean(d_losses) if d_losses else 0
            g_avg = np.mean(g_losses) if g_losses else 0
            d_avg = np.mean(d_losses) if d_losses else 0
            gp_avg = np.mean(gp_losses) if gp_losses else 0
            eps_avg = np.mean(eps_losses) if eps_losses else 0
            print(f"  [{epoch}/{epochs}] W_dist: {w_dist:.1f} | "
                  f"G_loss: {g_avg:.1f} | Time: {elapsed:.1f}m | ETA: {eta:.1f}m")
            save_gan_samples(G, fixed_z, fixed_labels, epoch, samples_dir, num_vis)

            log_dict = {
                "GAN_Training/Wasserstein_Dist": w_dist,
                "GAN_Training/Generator_Loss": g_avg,
                "GAN_Training/Critic_Loss": d_avg,
                "GAN_Training/Gradient_Penalty": gp_avg,
                "GAN_Training/Epsilon_Penalty": eps_avg,
                "GAN_Training/LR_Generator": new_lr,
                "GAN_Training/Epoch": epoch
            }
            sample_img_path = os.path.join(samples_dir, f'samples_epoch_{epoch:03d}.png')
            if os.path.exists(sample_img_path):
                log_dict["GAN_Samples/Generated_Images"] = wandb.Image(sample_img_path, caption=f"Epoch {epoch}")
            wandb.log(log_dict)

        # --- Checkpoint ---
        if epoch % save_every == 0 or epoch == epochs:
            g_path = os.path.join(models_dir, f'G_epoch_{epoch}.pth')
            d_path = os.path.join(models_dir, f'D_epoch_{epoch}.pth')
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            print(f"  [Checkpoint] G + D salvati (ep.{epoch})")

        # --- Validazione periodica ---
        if val_enabled and epoch % validate_every == 0:
            val_result = _run_augmented_validation(
                G, epoch, device, nz, n_class,
                train_dir, val_dir,
                num_gen_normal, num_gen_pneumonia,
                resnet_img_size, resnet_batch_size, resnet_epochs,
                augmented_dir, best_val_f1)
            val_history.append(val_result)

            if val_result['macro_f1'] > best_val_f1:
                best_val_f1 = val_result['macro_f1']
                best_val_epoch = epoch

    # --- Fine training ---
    gan_time = (time.time() - gan_start) / 60
    g_final = os.path.join(models_dir, f'G_epoch_{epochs}.pth')
    print(f"\nGAN training completato in {gan_time:.1f} minuti!")
    print(f"  Checkpoint finali: {models_dir}/G_epoch_{epochs}.pth")

    if val_history:
        print(f"\n{'='*50}")
        print(f"Validation Summary")
        print(f"{'='*50}")
        for r in val_history:
            marker = " ← BEST" if r['epoch'] == best_val_epoch else ""
            print(f"  Ep.{r['epoch']:3d} | Macro F1: {r['macro_f1']:.4f} | "
                  f"Acc: {r['accuracy']:.2f}%{marker}")
        print(f"\n  Miglior epoca: {best_val_epoch} (F1: {best_val_f1:.4f})")
        print(f"  Dataset augmented salvato in: {augmented_dir}")

    return G, g_final, best_val_epoch, val_history


def _run_augmented_validation(G, epoch, device, nz, n_class,
                              train_dir, val_dir,
                              num_gen_normal, num_gen_pneumonia,
                              resnet_img_size, resnet_batch_size, resnet_epochs,
                              augmented_dir, current_best_f1):
    """
    Esegue un singolo step di validazione:
      1. Genera sintetiche in cartella temporanea
      2. Crea dataset augmented (train originale + sintetiche)
      3. Allena ResNet e misura Macro F1 su val set
      4. Se è il miglior F1, salva il dataset in augmented_dir
      5. Pulisce i file temporanei
    """
    import os
    import shutil
    from eval import generate_synthetic_images
    from dataset.loader import get_dataloaders

    print(f"\n  {'─'*40}")
    print(f"  [VAL] Validazione epoca {epoch}")
    print(f"  {'─'*40}")

    val_tmp = './_val_tmp'
    tmp_syn = os.path.join(val_tmp, 'synthetic')
    tmp_aug_train = os.path.join(val_tmp, 'augmented', 'train')

    # Pulisci tmp precedente
    if os.path.exists(val_tmp):
        shutil.rmtree(val_tmp)

    # 1. Genera sintetiche
    generate_synthetic_images(
        G, num_gen_normal, num_gen_pneumonia,
        nz=nz, n_class=n_class, device=device, syn_dir=tmp_syn)

    # 2. Crea dataset augmented: copia train originale + sintetiche
    print(f"  [DEBUG] train_dir = {train_dir}")
    print(f"  [DEBUG] contenuto train_dir: {os.listdir(train_dir)}")
    shutil.copytree(train_dir, tmp_aug_train)
    for cat in ['NORMAL', 'PNEUMONIA']:
        syn_cat = os.path.join(tmp_syn, cat)
        if os.path.exists(syn_cat):
            for f in os.listdir(syn_cat):
                shutil.copy(os.path.join(syn_cat, f),
                            os.path.join(tmp_aug_train, cat, f))

    # 3. Allena ResNet su augmented, valida su val set
    aug_loader, val_loader, _, _ = get_dataloaders(
        tmp_aug_train, val_dir, val_dir,  # test_dir non serve, riuso val_dir
        img_size=resnet_img_size, batch_size=resnet_batch_size)

    _, hist, _ = train_resnet(
        aug_loader, val_loader, device,
        epochs=resnet_epochs, lr=0.001, tag=f"AugVal_ep{epoch}")

    # Risultati
    best_f1 = max(hist['val_macro_f1'])
    best_acc = hist['val_acc'][hist['val_macro_f1'].index(best_f1)]
    result = {'epoch': epoch, 'macro_f1': best_f1, 'accuracy': best_acc}

    wandb.log({
        "GAN_Val_TSTR/macro_f1": best_f1,
        "GAN_Val_TSTR/accuracy": best_acc,
        "GAN_Val_TSTR/epoch": epoch
    })
    
    print(f"  [VAL] Ep.{epoch}: Macro F1 = {best_f1:.4f}, Acc = {best_acc:.2f}%")

    # 4. Se è il migliore, conserva il dataset augmented
    if best_f1 > current_best_f1:
        print(f"  [VAL] ★ Nuovo miglior F1! Salvo dataset augmented...")
        if os.path.exists(augmented_dir):
            shutil.rmtree(augmented_dir)
        shutil.copytree(tmp_aug_train, os.path.join(augmented_dir, 'train'))
        print(f"  [VAL] Dataset salvato in {augmented_dir}")

    # 5. Pulisci tmp
    if os.path.exists(val_tmp):
        shutil.rmtree(val_tmp)

    return result


