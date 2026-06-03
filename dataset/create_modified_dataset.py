"""
Script da eseguire UNA SOLA VOLTA per creare il modified_dataset.
Il test set originale non viene toccato.
Train e Val vengono creati unendo le immagini dei vecchi train e val.
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
    seed=42
):
    info_path = os.path.join(dest, 'split_info.json')
    if os.path.exists(dest) and os.path.exists(info_path):
        print(f"modified_dataset gia presente in {dest}")
        with open(info_path) as f:
            info = json.load(f)
        print(f"   Train: {info['train_normal']} NORMAL + {info['train_pneumonia']} PNEUMONIA")
        print(f"   Val:   {info['val_normal']} NORMAL + {info['val_pneumonia']} PNEUMONIA")
        print(f"   Test:  {info['test_normal']} NORMAL + {info['test_pneumonia']} PNEUMONIA")
        return dest

    if os.path.exists(dest):
        print(f"Cartella incompleta trovata. Ricreo da zero...")
        shutil.rmtree(dest)

    random.seed(seed)

    if not os.path.exists(extract_path) and os.path.exists(zip_path):
        print(f"Estrazione dataset da {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Dataset estratto!")
    elif not os.path.exists(extract_path):
        print(f"ERRORE: {extract_path} non trovato e ZIP ({zip_path}) non presente.")
        return None

    orig_train = os.path.join(extract_path, 'chest_xray/train')
    orig_test  = os.path.join(extract_path, 'chest_xray/test')
    orig_val   = os.path.join(extract_path, 'chest_xray/val')

    print(f"\nCreazione modified_dataset in {dest}...")

    print("  Copiando test set originale (intatto)...")
    test_dest = os.path.join(dest, 'test')
    shutil.copytree(orig_test, test_dest)
    print("  Test copiato")
    
    test_normal = [f for f in os.listdir(os.path.join(test_dest, 'NORMAL')) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    test_pneumonia = [f for f in os.listdir(os.path.join(test_dest, 'PNEUMONIA')) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    avail_normal = []
    avail_pneumonia = []

    for orig_dir in [orig_train, orig_val]:
        if os.path.exists(orig_dir):
            for cat, target_list in [('NORMAL', avail_normal), ('PNEUMONIA', avail_pneumonia)]:
                cat_dir = os.path.join(orig_dir, cat)
                if os.path.exists(cat_dir):
                    for f in os.listdir(cat_dir):
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            target_list.append(os.path.join(cat_dir, f))

    avail_normal.sort()
    avail_pneumonia.sort()
    random.shuffle(avail_normal)
    random.shuffle(avail_pneumonia)

    target_val_n = 205
    target_val_p = 380

    val_normal = avail_normal[:target_val_n]
    val_pneumonia = avail_pneumonia[:target_val_p]

    train_normal = avail_normal[target_val_n:]
    train_pneumonia = avail_pneumonia[target_val_p:]

    splits = [
        ('train', 'NORMAL', train_normal),
        ('train', 'PNEUMONIA', train_pneumonia),
        ('val', 'NORMAL', val_normal),
        ('val', 'PNEUMONIA', val_pneumonia)
    ]

    for split_name, cat, files in splits:
        out_dir = os.path.join(dest, split_name, cat)
        os.makedirs(out_dir, exist_ok=True)
        for src_file in files:
            filename = os.path.basename(src_file)
            shutil.copy2(src_file, os.path.join(out_dir, filename))

    split_info = {
        'seed': seed,
        'train_normal': len(train_normal),
        'train_pneumonia': len(train_pneumonia),
        'val_normal': len(val_normal),
        'val_pneumonia': len(val_pneumonia),
        'test_normal': len(test_normal),
        'test_pneumonia': len(test_pneumonia)
    }

    with open(os.path.join(dest, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"\n{'='*50}")
    print(f"modified_dataset creato con successo!")
    print(f"{'='*50}")
    print(f"  Train: {len(train_normal)} NORMAL + {len(train_pneumonia)} PNEUMONIA = {len(train_normal) + len(train_pneumonia)}")
    print(f"  Val:   {len(val_normal)} NORMAL + {len(val_pneumonia)} PNEUMONIA = {len(val_normal) + len(val_pneumonia)}")
    print(f"  Test:  {len(test_normal)} NORMAL + {len(test_pneumonia)} PNEUMONIA = {len(test_normal) + len(test_pneumonia)}")
    print(f"\n  Salvato in: {dest}")

    return dest

if __name__ == '__main__':
    create_modified_dataset()
