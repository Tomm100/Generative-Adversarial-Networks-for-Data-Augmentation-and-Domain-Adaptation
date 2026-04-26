"""
Utility di logging per WandB.
Contiene funzioni per loggare confronti tra esperimenti.
"""

import os
import numpy as np
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def log_dann_comparison(report_pre, report_post, cm_pre, cm_post, class_names, out_dir):
    """
    Logga su WandB il confronto completo PRE-DANN vs POST-DANN:
      - Tabella riassuntiva con tutte le metriche
      - Bar chart per-classe (Precision, Recall, F1)
      - Confusion matrices side-by-side
      - Metriche scalari e delta
    """
    os.makedirs(out_dir, exist_ok=True)
    metrics = ['precision', 'recall', 'f1-score']

    # ── 1. Tabella riassuntiva WandB ──
    columns = ["Metric", "PRE-DANN", "POST-DANN", "Δ"]
    table_data = []

    # Accuracy & Macro F1
    for label, key in [("Accuracy", 'accuracy'),
                       ("Macro F1", ('macro avg', 'f1-score'))]:
        if isinstance(key, tuple):
            v_pre = report_pre[key[0]][key[1]]
            v_post = report_post[key[0]][key[1]]
        else:
            v_pre = report_pre[key]
            v_post = report_post[key]
        delta = v_post - v_pre
        table_data.append([label, round(v_pre, 4), round(v_post, 4), round(delta, 4)])

    # Per-class metrics
    for cls in class_names:
        for m in metrics:
            v_pre = report_pre[cls][m]
            v_post = report_post[cls][m]
            delta = v_post - v_pre
            table_data.append([f"{cls}/{m}", round(v_pre, 4), round(v_post, 4), round(delta, 4)])

    wandb_table = wandb.Table(columns=columns, data=table_data)

    # ── 2. Bar chart per-classe ──
    fig_bar, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(class_names))
    width = 0.35

    for i, m in enumerate(metrics):
        vals_pre = [report_pre[cls][m] for cls in class_names]
        vals_post = [report_post[cls][m] for cls in class_names]
        axes[i].bar(x - width/2, vals_pre, width, label='PRE-DANN', color='#5B9BD5')
        axes[i].bar(x + width/2, vals_post, width, label='POST-DANN', color='#ED7D31')
        axes[i].set_ylabel(m.capitalize())
        axes[i].set_title(m.capitalize())
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(class_names)
        axes[i].set_ylim(0, 1.05)
        axes[i].legend()
        axes[i].grid(axis='y', alpha=0.3)

    plt.suptitle('PRE-DANN vs POST-DANN — Per-Class Metrics', fontsize=14)
    plt.tight_layout()
    bar_path = os.path.join(out_dir, 'comparison_pre_vs_post_dann.png')
    plt.savefig(bar_path, dpi=150)
    plt.close(fig_bar)

    # ── 3. Confusion matrices side-by-side ──
    fig_cm, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_pre, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('PRE-DANN (ResNet Phase3)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    sns.heatmap(cm_post, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('POST-DANN')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    plt.suptitle('Confusion Matrix — PRE vs POST Domain Adaptation', fontsize=14)
    plt.tight_layout()
    cm_cmp_path = os.path.join(out_dir, 'cm_comparison_pre_vs_post_dann.png')
    plt.savefig(cm_cmp_path, dpi=150)
    plt.close(fig_cm)

    # ── 4. Log tutto su WandB ──
    acc_pre = report_pre['accuracy']
    acc_post = report_post['accuracy']
    f1_pre = report_pre['macro avg']['f1-score']
    f1_post = report_post['macro avg']['f1-score']

    log_dict = {
        # Tabella e immagini
        "Comparison/Summary_Table": wandb_table,
        "Comparison/Per_Class_Metrics": wandb.Image(bar_path),
        "Comparison/Confusion_Matrices": wandb.Image(cm_cmp_path),
        # Scalari
        "Comparison/PreDANN_Accuracy": acc_pre,
        "Comparison/PostDANN_Accuracy": acc_post,
        "Comparison/PreDANN_Macro_F1": f1_pre,
        "Comparison/PostDANN_Macro_F1": f1_post,
        "Comparison/Delta_Accuracy": acc_post - acc_pre,
        "Comparison/Delta_Macro_F1": f1_post - f1_pre,
    }

    # Per-class scalari
    for cls in class_names:
        for m in metrics:
            log_dict[f"Comparison/{cls}_PreDANN_{m}"] = report_pre[cls][m]
            log_dict[f"Comparison/{cls}_PostDANN_{m}"] = report_post[cls][m]
            log_dict[f"Comparison/{cls}_Delta_{m}"] = report_post[cls][m] - report_pre[cls][m]

    wandb.log(log_dict)

    # ── 5. Stampa riepilogo console ──
    print(f"\n{'═'*60}")
    print(f"  CONFRONTO PRE-DANN vs POST-DANN (loggato su WandB)")
    print(f"{'═'*60}")
    print(f"  Accuracy:  {acc_pre:.4f} → {acc_post:.4f}  (Δ {acc_post - acc_pre:+.4f})")
    print(f"  Macro F1:  {f1_pre:.4f} → {f1_post:.4f}  (Δ {f1_post - f1_pre:+.4f})")
    print(f"  📊 Plot salvati in: {out_dir}")
    print(f"  📊 Tutti i risultati loggati su WandB")
