import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from sklearn.metrics import f1_score

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
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            rl += loss.item()
        avg_tl = rl / len(train_loader)

        # --- Validation ---
        model.eval()
        vl, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
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
                 samples_dir='gan_samples'):
    """
    Allena WGAN-GP per la generazione di immagini.

    Args:
        G, D:           Generator e Critic (già su device)
        gan_loader:     DataLoader per il training
        device:         torch device
        compute_gp_fn:  funzione gradient penalty (da models.wgan)
        epochs:         numero totale di epoche
        lr:             learning rate base
        n_critic:       rapporto aggiornamenti critic/generator
        nz, n_class:    config GAN
        lr_milestones:  lista di epoche in cui applicare il decay (default: [60, 80])
        lr_gamma:       fattore moltiplicativo per il decay (default: 0.2, equivale a /5)
        save_every:     salva checkpoint ogni N epoche
        models_dir:     cartella dove salvare i checkpoint G/D
        samples_dir:    cartella dove salvare le griglie di sample

    Returns:
        (G, g_final_path) — Generator allenato e path dell'ultimo checkpoint
    """
    import os
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    if lr_milestones is None:
        lr_milestones = [60, 80]

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

    # --- Training loop ---
    print(f"\nTraining WGAN-GP: {epochs} epoche")
    print(f"  LR: {lr}, n_critic={n_critic}")
    print(f"  LR decay: x{lr_gamma} alle epoche {lr_milestones}")

    gan_start = time.time()

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []

        for batch_idx, (x_, y_) in enumerate(gan_loader):
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
            d_loss = D_fake - D_real + gp
            d_loss.backward()
            D_opt.step()
            d_losses.append(d_loss.item())

            # --- Train Generator ---
            if (batch_idx + 1) % n_critic == 0:
                G.zero_grad()
                z = torch.randn(mb, nz, 1, 1).to(device)
                y_gen = torch.randint(0, n_class, (mb,)).to(device)
                g_loss = -D(G(z, onehot[y_gen]), fill[y_gen]).squeeze().mean()
                g_loss.backward()
                G_opt.step()
                g_losses.append(g_loss.item())

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
            print(f"  [{epoch}/{epochs}] W_dist: {w_dist:.1f} | "
                  f"G_loss: {g_avg:.1f} | Time: {elapsed:.1f}m | ETA: {eta:.1f}m")
            save_gan_samples(G, fixed_z, fixed_labels, epoch, samples_dir, num_vis)

        # --- Checkpoint ---
        if epoch % save_every == 0 or epoch == epochs:
            g_path = os.path.join(models_dir, f'G_epoch_{epoch}.pth')
            d_path = os.path.join(models_dir, f'D_epoch_{epoch}.pth')
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            print(f"  [Checkpoint] G + D salvati (ep.{epoch})")

    gan_time = (time.time() - gan_start) / 60
    g_final = os.path.join(models_dir, f'G_epoch_{epochs}.pth')
    print(f"\nGAN training completato in {gan_time:.1f} minuti!")
    print(f"  Checkpoint finali: {models_dir}/G_epoch_{epochs}.pth")

    return G, g_final

