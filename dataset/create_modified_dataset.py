"""
Script da eseguire UNA SOLA VOLTA su Google Colab.

Crea il 'modified_dataset' in LOCALE con:
- train/  → copia intatta del train set originale
- val/    → 12.5% stratificato dal test set originale + 16 img validation originale
- test/   → test set originale MENO le immagini spostate in val

Poi lo copi tu su Drive manualmente:
  !cp -r ./modified_dataset /content/drive/MyDrive/ProgettoMLVM/modified_dataset

Le immagini sono selezionate con seed fisso (42) per riproducibilità.
Un file split_info.json viene salvato per documentare lo split.
"""

import os
import shutil
import random
import json
import zipfile


def create_modified_dataset(
    zip_path='/content/drive/MyDrive/ProgettoMLVM/chest_xray.zip',
    extract_path='./original_dataset',
    dest='./modified_dataset',
    val_ratio=0.125,
    seed=42
):
    # Controlla se già completo
    info_path = os.path.join(dest, 'split_info.json')
    if os.path.exists(dest) and os.path.exists(info_path):
        print(f"✅ modified_dataset già presente in {dest}")
        with open(info_path) as f:
            info = json.load(f)
        print(f"   Val:  {info['val_normal']} NORMAL + {info['val_pneumonia']} PNEUMONIA")
        print(f"   Test: {info['test_normal']} NORMAL + {info['test_pneumonia']} PNEUMONIA")
        return dest

    # Se cartella incompleta, rimuovi e ricrea
    if os.path.exists(dest):
        print(f"⚠️  Cartella incompleta trovata. Ricreo da zero...")
        shutil.rmtree(dest)

    random.seed(seed)

    # 1. Estrazione ZIP
    if not os.path.exists(extract_path) and os.path.exists(zip_path):
        print(f"Estrazione dataset da {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Dataset estratto!")
    elif not os.path.exists(extract_path):
        print(f"ERRORE: {extract_path} non trovato e ZIP ({zip_path}) non presente.")
        return None

    # 2. Percorsi originali
    orig_train = os.path.join(extract_path, 'chest_xray/train')
    orig_test  = os.path.join(extract_path, 'chest_xray/test')
    orig_val   = os.path.join(extract_path, 'chest_xray/val')

    for p in [orig_train, orig_test]:
        if not os.path.exists(p):
            print(f"ERRORE: {p} non trovato!")
            return None

    # 3. Crea struttura
    train_dest = os.path.join(dest, 'train')
    val_dest   = os.path.join(dest, 'val')
    test_dest  = os.path.join(dest, 'test')

    print(f"\nCreazione modified_dataset in {dest}...")

    # 3a. Copia train intatto
    print("  Copiando train set (intatto)...")
    shutil.copytree(orig_train, train_dest)
    print("  Train copiato ✓")

    # 3b. Split stratificato del test set
    split_info = {'seed': seed, 'val_ratio': val_ratio, 'val_files': {}, 'test_files': {}}

    for cat in ['NORMAL', 'PNEUMONIA']:
        os.makedirs(os.path.join(val_dest, cat), exist_ok=True)
        os.makedirs(os.path.join(test_dest, cat), exist_ok=True)

        orig_cat_dir = os.path.join(orig_test, cat)
        all_files = sorted([
            f for f in os.listdir(orig_cat_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        random.shuffle(all_files)

        n_val = int(len(all_files) * val_ratio)
        val_files  = all_files[:n_val]
        test_files = all_files[n_val:]

        for f in val_files:
            shutil.copy2(os.path.join(orig_cat_dir, f), os.path.join(val_dest, cat, f))
        for f in test_files:
            shutil.copy2(os.path.join(orig_cat_dir, f), os.path.join(test_dest, cat, f))

        split_info['val_files'][cat] = val_files
        split_info['test_files'][cat] = test_files
        print(f"  {cat}: {len(val_files)} → val, {len(test_files)} → test (totale: {len(all_files)})")

    # 3c. Aggiungi le 16 immagini della validation originale al val set
    n_orig_val = 0
    split_info['orig_val_files'] = {}
    for cat in ['NORMAL', 'PNEUMONIA']:
        orig_val_cat = os.path.join(orig_val, cat)
        copied = []
        if os.path.exists(orig_val_cat):
            for f in os.listdir(orig_val_cat):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy2(os.path.join(orig_val_cat, f), os.path.join(val_dest, cat, f))
                    copied.append(f)
                    n_orig_val += 1
        split_info['orig_val_files'][cat] = copied
    print(f"  + {n_orig_val} immagini dalla validation originale aggiunte al val set")

    # Conteggi
    split_info['val_normal']     = len(split_info['val_files']['NORMAL']) + len(split_info['orig_val_files']['NORMAL'])
    split_info['val_pneumonia']  = len(split_info['val_files']['PNEUMONIA']) + len(split_info['orig_val_files']['PNEUMONIA'])
    split_info['test_normal']    = len(split_info['test_files']['NORMAL'])
    split_info['test_pneumonia'] = len(split_info['test_files']['PNEUMONIA'])

    train_n = len([f for f in os.listdir(os.path.join(train_dest, 'NORMAL')) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    train_p = len([f for f in os.listdir(os.path.join(train_dest, 'PNEUMONIA')) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    split_info['train_normal']    = train_n
    split_info['train_pneumonia'] = train_p

    # Salva split_info.json (ULTIMO STEP = prova di completamento)
    with open(os.path.join(dest, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\n{'='*50}")
    print(f"modified_dataset creato con successo!")
    print(f"{'='*50}")
    print(f"  Train: {train_n} NORMAL + {train_p} PNEUMONIA = {train_n + train_p}")
    print(f"  Val:   {split_info['val_normal']} NORMAL + {split_info['val_pneumonia']} PNEUMONIA = {split_info['val_normal'] + split_info['val_pneumonia']}")
    print(f"  Test:  {split_info['test_normal']} NORMAL + {split_info['test_pneumonia']} PNEUMONIA = {split_info['test_normal'] + split_info['test_pneumonia']}")
    print(f"\n  Salvato in: {dest}")
    print(f"\n  Per copiarlo su Drive:")
    print(f"  !cp -r {dest} /content/drive/MyDrive/ProgettoMLVM/modified_dataset")

    return dest


if __name__ == '__main__':
    create_modified_dataset()
