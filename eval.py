import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from torchvision import transforms

def evaluate_on_test(model, ckpt_path, test_loader, class_names, device, tag="Phase1", out_dir='./results'):
    """
    Valuta il modello sul test set e salva i grafici e le metriche in `out_dir`.
    Restituisce il classification report e la confusion matrix.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            _, pred = torch.max(model(x), 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print(f"\n{'='*50}\nRISULTATI {tag}\n{'='*50}")
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=class_names)
    print(report_str)

    # Salva il report in formato testuale
    with open(os.path.join(out_dir, f'report_{tag}.txt'), 'w') as f:
        f.write(f"RISULTATI {tag}\n{'='*50}\n")
        f.write(report_str)

    # Plot e salvataggio Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {tag}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'cm_{tag}.png'))
    plt.close(fig) # Non usiamo plt.show() in esecuzione batch su server

    return report_dict, cm

def generate_synthetic_images(G, num_gen_normal, num_gen_pneumonia, nz, n_class, device, syn_dir='synthetic_images'):
    """
    Genera immagini sintetiche per bilanciare il dataset originale.
    """
    os.makedirs(os.path.join(syn_dir, 'NORMAL'), exist_ok=True)
    os.makedirs(os.path.join(syn_dir, 'PNEUMONIA'), exist_ok=True)

    onehot = torch.eye(n_class).view(n_class, n_class, 1, 1).to(device)
    
    G.eval()
    with torch.no_grad():
        for cls_idx, cls_name, num_gen in [(0, 'NORMAL', num_gen_normal),
                                            (1, 'PNEUMONIA', num_gen_pneumonia)]:
            if num_gen <= 0:
                print(f"  {cls_name}: nessuna generazione necessaria")
                continue

            generated = 0
            while generated < num_gen:
                batch_sz = min(64, num_gen - generated)
                z = torch.randn(batch_sz, nz, 1, 1).to(device)
                labels = onehot[torch.full((batch_sz,), cls_idx, dtype=torch.long).to(device)]
                fakes = G(z, labels)

                for i in range(batch_sz):
                    img = fakes[i].cpu()
                    img = (img + 1) / 2.0  # [-1,1] -> [0,1]
                    img_pil = transforms.ToPILImage()(img)
                    img_rgb = img_pil.convert('RGB') # 128x128 
                    img_rgb.save(os.path.join(syn_dir, cls_name, f'syn_{cls_name}_{generated+i}.png'))

                generated += batch_sz

            print(f"  {cls_name}: generate {num_gen} immagini sintetiche")
    print("Generazione completata! ✅")

def plot_comparison(hist_p1, hist_p3, cm_p1, cm_p3, classes, report_p1, report_p3, out_dir='./results'):
    """
    Stampa le metriche a confronto e plotta le curve e matrici side-by-side.
    Salva i plot e il report finale in `out_dir`.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    comp_text = []
    comp_text.append("\n" + "=" * 60)
    comp_text.append("CONFRONTO FINALE: Phase 1 (Baseline) vs Phase 3 (Augmented)")
    comp_text.append("=" * 60)

    metrics = ['precision', 'recall', 'f1-score']
    for cls in classes:
        comp_text.append(f"\n  {cls}:")
        for m in metrics:
            v1 = report_p1[cls][m]
            v3 = report_p3[cls][m]
            diff = v3 - v1
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
            comp_text.append(f"    {m:12s}: {v1:.4f} → {v3:.4f}  ({arrow} {abs(diff):.4f})")

    acc_p1, acc_p3 = report_p1['accuracy'], report_p3['accuracy']
    diff_acc = acc_p3 - acc_p1
    arrow = "↑" if diff_acc > 0 else "↓"
    comp_text.append(f"\n  Overall Acc: {acc_p1:.4f} → {acc_p3:.4f}  ({arrow} {abs(diff_acc):.4f})")

    full_output = "\n".join(comp_text)
    print(full_output)
    
    with open(os.path.join(out_dir, 'comparison_report.txt'), 'w') as f:
        f.write(full_output)

    # Plot confronto curve
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, metric in enumerate(['train_loss', 'val_loss', 'val_acc']):
        axes[i].plot(hist_p1[metric], label='Phase 1', marker='o')
        axes[i].plot(hist_p3[metric], label='Phase 3', marker='s')
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].legend()
        axes[i].grid(True)
    plt.suptitle('Phase 1 vs Phase 3 — Training Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison_p1_vs_p3.png'))
    plt.close(fig)

    # Matrici
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_p1, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title('Phase 1 (Baseline)')
    sns.heatmap(cm_p3, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title('Phase 3 (Augmented)')
    plt.suptitle('Confusion Matrix Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cm_comparison.png'))
    plt.close(fig)
