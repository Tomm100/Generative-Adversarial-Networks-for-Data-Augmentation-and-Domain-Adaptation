import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import f1_score

from models.resnet import get_resnet_classifier

def train_resnet(train_loader, val_loader, device, epochs=15, lr=0.001, patience=5, tag="Phase1"):
    """
    Allena ResNet18 (Feature Extractor / Classifier).
    Salva il checkpoint migliore in base alla Macro F1 sul validation set.
    Early stopping con patience configurabile.
    """
    model = get_resnet_classifier(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_macro_f1 = -1.0
    best_weights = None
    patience_counter = 0
    ckpt_path = f'best_model_{tag}.pth'
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': []}

    print(f"\n{'='*50}\nTraining {tag} ({epochs} epoche, patience={patience})\n{'='*50}")

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
            patience_counter = 0
            saved = "best model saved"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1}/{epochs} | TL: {avg_tl:.4f} | VL: {avg_vl:.4f} | "
              f"VA: {acc:.2f}% | Macro F1: {macro_f1:.4f}{saved}")

        if patience_counter >= patience:
            print(f"Early stopping (no improvement for {patience} epochs)")
            break

    # Carica i pesi migliori
    model.load_state_dict(best_weights)
    torch.save(best_weights, ckpt_path)
    print(f"  Best Macro F1: {best_macro_f1:.4f} (salvato in {ckpt_path})")

    return model, history, ckpt_path


def train_wgangp(G, D, gan_loader, device, compute_gp_fn,
                 start_epoch=0, end_epoch=100,
                 lr=0.0001, n_critic=5, nz=100, n_class=2,
                 lr_decay_schedule=None,
                 save_every=20,
                 models_dir='gan_checkpoints',
                 samples_dir='gan_samples',
                 g_ckpt_path=None, d_ckpt_path=None):
    """
    Allena WGAN-GP per la generazione di immagini.

    Supporta training da zero o resume da checkpoint:
      - Fresh:  start_epoch=0, end_epoch=100
      - Resume: start_epoch=100, end_epoch=200, g_ckpt_path=..., d_ckpt_path=...

    Args:
        G, D:           Generator e Critic (già su device)
        gan_loader:     DataLoader per il training
        device:         torch device
        compute_gp_fn:  funzione gradient penalty (da models.wgan)
        start_epoch:    epoca di partenza (0 = fresh training)
        end_epoch:      epoca finale
        lr:             learning rate base
        n_critic:       rapporto aggiornamenti critic/generator
        nz, n_class:    config GAN
        lr_decay_schedule: dict {epoch: divisore} per LR decay
                           Default: {60: 5, 80: 5}
        save_every:     salva checkpoint ogni N epoche
        models_dir:     cartella dove salvare i checkpoint G/D
        samples_dir:    cartella dove salvare le griglie di sample
        g_ckpt_path:    path checkpoint Generator per resume (None = fresh)
        d_ckpt_path:    path checkpoint Critic per resume (None = fresh)

    Returns:
        (G, g_final_path) — Generator allenato e path dell'ultimo checkpoint
    """
    import os
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    if lr_decay_schedule is None:
        lr_decay_schedule = {60: 5, 80: 5}

    is_resume = start_epoch > 0

    # --- Caricamento checkpoint per resume ---
    if is_resume:
        if g_ckpt_path and os.path.exists(g_ckpt_path):
            G.load_state_dict(torch.load(g_ckpt_path, map_location=device))
            print(f"  Generator caricato da {g_ckpt_path}")
        else:
            print(f"  ATTENZIONE: g_ckpt_path non fornito o non trovato, uso pesi attuali di G")
        if d_ckpt_path and os.path.exists(d_ckpt_path):
            D.load_state_dict(torch.load(d_ckpt_path, map_location=device))
            print(f"  Critic caricato da {d_ckpt_path}")
        else:
            print(f"  ATTENZIONE: d_ckpt_path non fornito o non trovato, uso pesi attuali di D")
    else:
        # Fresh training: inizializza pesi
        G.weight_init(0.0, 0.02)
        D.weight_init(0.0, 0.02)

    # --- Calcola LR effettivo considerando i decay passati ---
    effective_lr = lr
    for decay_epoch in sorted(lr_decay_schedule.keys()):
        if start_epoch >= decay_epoch:
            effective_lr /= lr_decay_schedule[decay_epoch]

    G_opt = optim.Adam(G.parameters(), lr=effective_lr, betas=(0.0, 0.9))
    D_opt = optim.Adam(D.parameters(), lr=effective_lr, betas=(0.0, 0.9))

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

    def save_gan_samples(epoch):
        G.eval()
        with torch.no_grad():
            imgs = G(fixed_z, fixed_labels)
        G.train()
        fig, axes = plt.subplots(2, num_vis, figsize=(num_vis * 2.5, 5))
        for cls_idx, cls_name in enumerate(['NORMAL', 'PNEUMONIA']):
            for j in range(num_vis):
                idx = cls_idx * num_vis + j
                axes[cls_idx, j].imshow(imgs[idx, 0].cpu().numpy(), cmap='gray')
                axes[cls_idx, j].axis('off')
                if j == 0: axes[cls_idx, j].set_ylabel(cls_name, fontsize=10)
        plt.suptitle(f'GAN Samples — Epoch {epoch}', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f'samples_epoch_{epoch:03d}.png'))
        plt.close(fig)

    # --- Training loop ---
    total_epochs = end_epoch - start_epoch
    resume_str = f" (resume da ep.{start_epoch})" if is_resume else ""
    print(f"\nTraining WGAN-GP: ep.{start_epoch+1} → {end_epoch}{resume_str}")
    print(f"  LR effettivo: {effective_lr:.8f}, n_critic={n_critic}")

    gan_start = time.time()

    for epoch_idx in range(total_epochs):
        epoch = start_epoch + epoch_idx + 1  # epoca assoluta (1-indexed nel log)
        d_losses, g_losses = [], []

        # LR decay (solo per epoche future, non già applicate)
        if epoch in lr_decay_schedule:
            divisor = lr_decay_schedule[epoch]
            for pg in G_opt.param_groups + D_opt.param_groups:
                pg['lr'] /= divisor
            print(f"  [LR decay] Epoca {epoch}: LR → {G_opt.param_groups[0]['lr']:.8f}")

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

        # --- Log + Sample ---
        elapsed = (time.time() - gan_start) / 60
        eta = (elapsed / (epoch_idx + 1)) * (total_epochs - epoch_idx - 1)

        if epoch % 5 == 0 or epoch_idx == 0:
            w_dist = -np.mean(d_losses) if d_losses else 0
            g_avg = np.mean(g_losses) if g_losses else 0
            print(f"  [{epoch}/{end_epoch}] W_dist: {w_dist:.1f} | "
                  f"G_loss: {g_avg:.1f} | Time: {elapsed:.1f}m | ETA: {eta:.1f}m")
            save_gan_samples(epoch)

        # --- Checkpoint ---
        if epoch % save_every == 0 or epoch == end_epoch:
            g_path = os.path.join(models_dir, f'G_epoch_{epoch}.pth')
            d_path = os.path.join(models_dir, f'D_epoch_{epoch}.pth')
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            print(f"  [Checkpoint] G + D salvati (ep.{epoch})")

    gan_time = (time.time() - gan_start) / 60
    g_final = os.path.join(models_dir, f'G_epoch_{end_epoch}.pth')
    print(f"\nGAN training completato in {gan_time:.1f} minuti!")
    print(f"  Checkpoint finali: {models_dir}/G_epoch_{end_epoch}.pth")

    return G, g_final

