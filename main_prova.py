import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
import shutil
import wandb
import re
from tqdm import tqdm
from sklearn.metrics import f1_score
import config

from config import (
    DATASET_DIR,
    RESNET_IMG_SIZE, RESNET_BATCH_SIZE, RESNET_EPOCHS, RESNET_LR,
    GAN_NZ, GAN_N_CLASS, GAN_NC, GAN_D,
    SEED
)
from dataset.loader import setup_dataset, get_dataloaders
from models.wgan import Generator
from eval import evaluate_on_test, generate_synthetic_images, plot_comparison
from utils.seed import set_seed

# ==============================================================================
# ⚙️ IMPOSTAZIONI MANUALI
# ==============================================================================
CUSTOM_GAN_WEIGHTS_PATH = "/content/drive/MyDrive/ProgettoMLVM/results_BAGAN/gan_checkpoints/G_epoch_200.pth"
DRIVE_EVAL_BASE_DIR = "/content/drive/MyDrive/ProgettoMLVM/evaluation_gan_epoch"

# Aggiungi qui i classificatori che vuoi testare in loop!
MODELS_TO_TEST = ["resnet18", "densenet121", "efficientnet_b0", "vgg16", "alexnet", "googlenet"]
# ==============================================================================

def get_classifier(name="resnet18", num_classes=2):
    """Factory per istanziare diversi modelli pre-addestrati."""
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "googlenet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Modello {name} non supportato")
    return model

def train_classifier(model_name, train_loader, val_loader, device, epochs, lr, tag, ckpt_dir):
    """Logica di training sganciata da train.py per supportare vari modelli."""
    model = get_classifier(model_name, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_macro_f1 = -1.0
    best_weights = None
    ckpt_path = os.path.join(ckpt_dir, f'best_{model_name}_{tag}.pth')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': []}
    
    for epoch in range(epochs):
        model.train()
        rl = 0.0
        pbar_train = tqdm(train_loader, desc=f"  Train Ep {epoch+1}/{epochs} ({model_name})", leave=False)
        for x, y in pbar_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            rl += loss.item()
            pbar_train.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_tl = rl / len(train_loader)

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
            f"{model_name}/{tag}/train_loss": avg_tl,
            f"{model_name}/{tag}/val_loss": avg_vl,
            f"{model_name}/{tag}/val_macro_f1": macro_f1,
            "epoch": epoch + 1
        })

        saved = ""
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            saved = " ← best"

        print(f"  [{model_name}] Ep {epoch+1}/{epochs} | TL: {avg_tl:.4f} | VL: {avg_vl:.4f} | VA: {acc:.2f}% | F1: {macro_f1:.4f}{saved}")

    model.load_state_dict(best_weights)
    torch.save(best_weights, ckpt_path)
    return model, history, ckpt_path

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}")
    print(f"  ESPERIMENTO: Confronto Multi-Modello con Data Augmentation (GAN)")
    print(f"  Modelli in test: {MODELS_TO_TEST}")
    print(f"  Device: {device}")
    print(f"{'='*80}")

    if not os.path.exists(CUSTOM_GAN_WEIGHTS_PATH):
        print(f"\n❌ ERRORE: Il file specificato non esiste:\n   {CUSTOM_GAN_WEIGHTS_PATH}")
        return

    match = re.search(r'G_epoch_(\d+)', CUSTOM_GAN_WEIGHTS_PATH)
    epoch_num = int(match.group(1)) if match else "Custom"

    epoch_dir = os.path.join(DRIVE_EVAL_BASE_DIR, f"epoca{epoch_num}")
    EXP_SYNTHETIC_DIR = os.path.join(epoch_dir, "synthetic_images")
    EXP_AUGMENTED_DIR = os.path.join(epoch_dir, "augmented_dataset")
    
    os.makedirs(epoch_dir, exist_ok=True)

    wandb.init(
        project="gan-chest-xray-augmentation",
        entity="MachineLearningForVisionAndMultimedia",
        name=f"multi_model_eval_epoca_{epoch_num}",
        config={
            "models_tested": MODELS_TO_TEST,
            "seed": SEED,
            "epochs": RESNET_EPOCHS,
            "gan_weights_path": CUSTOM_GAN_WEIGHTS_PATH
        }
    )

    res = setup_dataset(dataset_dir=DATASET_DIR)
    if not res: return
    train_dir, val_dir, test_dir = res

    n_train_n = len([f for f in os.listdir(os.path.join(train_dir, 'NORMAL')) if not f.startswith('.')])
    n_train_p = len([f for f in os.listdir(os.path.join(train_dir, 'PNEUMONIA')) if not f.startswith('.')])
    num_gen_normal = max(0, n_train_p - n_train_n)
    num_gen_pneumonia = max(0, n_train_n - n_train_p)

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir, img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    # --- 1. CARICA GAN E CREA DATASET AUGMENTED (Fatto una sola volta) ---
    print(f"\n[1/2] Generazione immagini sintetiche e dataset aumentato...")
    G = Generator(nz=GAN_NZ, n_class=GAN_N_CLASS, nc=GAN_NC, d=GAN_D).to(device)
    G.load_state_dict(torch.load(CUSTOM_GAN_WEIGHTS_PATH, map_location=device))
    G.eval()

    if os.path.exists(EXP_SYNTHETIC_DIR): shutil.rmtree(EXP_SYNTHETIC_DIR)
    generate_synthetic_images(G, num_gen_normal, num_gen_pneumonia, nz=GAN_NZ, n_class=GAN_N_CLASS, device=device, syn_dir=EXP_SYNTHETIC_DIR)

    aug_train_dir = os.path.join(EXP_AUGMENTED_DIR, 'train')
    if os.path.exists(EXP_AUGMENTED_DIR): shutil.rmtree(EXP_AUGMENTED_DIR)
    shutil.copytree(train_dir, aug_train_dir)

    for cat in ['NORMAL', 'PNEUMONIA']:
        syn_cat = os.path.join(EXP_SYNTHETIC_DIR, cat)
        if os.path.exists(syn_cat):
            for f in os.listdir(syn_cat):
                shutil.copy(os.path.join(syn_cat, f), os.path.join(aug_train_dir, cat, f))

    aug_train_loader, _, _, _ = get_dataloaders(
        aug_train_dir, val_dir, test_dir, img_size=RESNET_IMG_SIZE, batch_size=RESNET_BATCH_SIZE)

    # --- 2. LOOP SUI MODELLI ---
    print(f"\n[2/2] Avvio training per i modelli: {MODELS_TO_TEST}")
    
    for model_name in MODELS_TO_TEST:
        print(f"\n" + "🚀"*30)
        print(f"🚀 INIZIO VALUTAZIONE: {model_name.upper()}")
        print("🚀"*30)

        model_dir = os.path.join(epoch_dir, model_name)
        EXP_METRICS_DIR = os.path.join(model_dir, "metrics")
        EXP_CHECKPOINTS_DIR = os.path.join(model_dir, "checkpoints")
        os.makedirs(EXP_METRICS_DIR, exist_ok=True)
        os.makedirs(EXP_CHECKPOINTS_DIR, exist_ok=True)

        # FASE 1
        print(f"\n--- {model_name.upper()} | Phase 1 (Baseline) ---")
        model_p1, hist_p1, ckpt_p1 = train_classifier(
            model_name, train_loader, val_loader, device, 
            epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Phase1", ckpt_dir=EXP_CHECKPOINTS_DIR)
        report_p1, cm_p1 = evaluate_on_test(
            model_p1, ckpt_p1, test_loader, classes, device,
            tag=f"{model_name}_Phase1", out_dir=EXP_METRICS_DIR)

        # FASE 3
        print(f"\n--- {model_name.upper()} | Phase 3 (Augmented) ---")
        model_p3, hist_p3, ckpt_p3 = train_classifier(
            model_name, aug_train_loader, val_loader, device, 
            epochs=RESNET_EPOCHS, lr=RESNET_LR, tag="Phase3", ckpt_dir=EXP_CHECKPOINTS_DIR)
        report_p3, cm_p3 = evaluate_on_test(
            model_p3, ckpt_p3, test_loader, classes, device,
            tag=f"{model_name}_Phase3", out_dir=EXP_METRICS_DIR)

        # CONFRONTO
        plot_comparison(hist_p1, hist_p3, cm_p1, cm_p3, classes,
                        report_p1, report_p3, out_dir=EXP_METRICS_DIR)
        
        print(f"✅ Valutazione {model_name.upper()} completata. Risultati in {model_dir}")

    print(f"\n🎉 ESPERIMENTO MULTI-MODELLO CONCLUSO CON SUCCESSO! 🎉")
    wandb.finish()

if __name__ == '__main__':
    main()
