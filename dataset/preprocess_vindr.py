"""
Preprocessing VinDr -> Kermany per la Cross-Hospital Domain Adaptation.

Obiettivo: ridurre il domain shift *a monte* (prima della DANN) rendendo le
immagini VinDr visivamente simili a quelle Kermany.

Differenze osservate tra i due ospedali:
  1. POLARITA' INVERTITA (la piu' importante): in Kermany i polmoni sono SCURI e
     l'osso/mediastino CHIARO (radiografia standard, MONOCHROME2); in VinDr la
     polarita' e' invertita (polmoni chiari, resto scuro, tipo negativo).
     -> L'histogram matching NON puo' correggerla (e' monotono): va invertita PRIMA.
  2. Luminosita'/contrasto diversi (VinDr piu' chiaro e piatto).
  3. Campo visivo piu' ampio in VinDr (spalle/collo/bordi visibili).
  4. Marker radiopachi ("R") e bordi di collimazione.

Pipeline per ogni immagine VinDr:
     grayscale -> [center-crop FOV] -> correzione polarita' (auto vs Kermany)
     -> histogram matching a Kermany -> [CLAHE] -> [crop margini] -> resize -> save

Output: stessa struttura di VinDr ({train,val,test}/{NORMAL,PNEUMONIA}/*.png),
salvato in OUTPUT_ROOT. Poi in main_dann_cross_hospital.py imposta
    TARGET_ROOT = OUTPUT_ROOT
e rilancia l'esperimento.

Uso:  python dataset/preprocess_vindr.py
Dipendenze: numpy, pillow  (scikit-image serve SOLO se attivi DO_CLAHE)
"""

import os
import glob
import random
import numpy as np
from PIL import Image

# ============================ CONFIG ============================
KERMANY_ROOT = "./data/modified_dataset"        # source (riferimento di stile)
VINDR_ROOT   = "./VinDr_Target_DA_Ready"         # target da preprocessare
OUTPUT_ROOT  = "./VinDr_preProcessed"            # dove salvare il risultato

SPLITS  = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]

# --- Operazioni (attiva/disattiva) ---
DO_INVERT       = "auto"   # "auto" = decide per-immagine confrontando con Kermany
                           # True = inverti sempre | False = non invertire mai
DO_HISTMATCH    = True     # allinea luminosita'/contrasto a Kermany
DO_CENTER_CROP  = True     # ritaglia il centro per avvicinare il FOV a Kermany
CENTER_CROP_FRAC = 0.85    # frazione centrale mantenuta (1.0 = nessun crop)
DO_CLAHE        = False    # contrasto locale adattivo (piu' aggressivo)
CLAHE_CLIP      = 0.01
DO_MARGIN_CROP  = False    # ritaglia i margini per attenuare marker "R"/bordi
MARGIN_FRAC     = 0.06     # frazione di bordo rimossa per lato

OUT_SIZE        = None     # None = mantieni dimensione nativa; oppure es. 256
REF_SAMPLE      = 300      # n. immagini Kermany campionate per il riferimento
REF_SIZE        = 256      # dimensione a cui uniformare il riferimento
SEED            = 42
SAVE_QC         = True     # salva un montaggio prima/dopo per controllo visivo
QC_SAMPLES      = 6
# ===============================================================

random.seed(SEED)
np.random.seed(SEED)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".bmp", ".tif", ".tiff")


# --------------------------- utility base ---------------------------
def load_gray(path):
    """Carica un'immagine come float32 grayscale in [0,1]."""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def save_gray(arr, path):
    """Salva un array float [0,1] come PNG grayscale a 8 bit."""
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def match_histograms(source, reference):
    """Histogram matching quantile-based (equivalente a skimage), puro numpy.
    source, reference: array 1D di intensita'. Restituisce source rimappato in
    modo che la sua distribuzione cumulativa combaci con quella di reference.
    Nota: e' una trasformazione MONOTONA -> preserva l'ordine chiaro/scuro,
    per questo NON puo' correggere l'inversione di polarita' (va fatta prima).
    """
    s_vals, s_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    r_vals, r_counts = np.unique(reference, return_counts=True)
    s_q = np.cumsum(s_counts).astype(np.float64) / source.size
    r_q = np.cumsum(r_counts).astype(np.float64) / reference.size
    interp = np.interp(s_q, r_q, r_vals)
    return interp[s_idx]


def center_crop(arr, frac):
    """Ritaglia la porzione centrale (frac in (0,1]) e la restituisce."""
    if frac >= 0.999:
        return arr
    h, w = arr.shape
    ch, cw = int(h * frac), int(w * frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return arr[y0:y0 + ch, x0:x0 + cw]


def margin_crop(arr, frac):
    """Rimuove una cornice di larghezza frac*lato su ogni bordo."""
    if frac <= 0.0:
        return arr
    h, w = arr.shape
    my, mx = int(h * frac), int(w * frac)
    return arr[my:h - my, mx:w - mx]


def polarity_stat(arr):
    """Statistica di polarita': media della striscia centrale (mediastino/colonna)
    meno media delle strisce laterali (polmoni).
    In polarita' Kermany (osso chiaro, polmoni scuri) il centro e' PIU' CHIARO
    -> statistica > 0. Se invertita -> statistica < 0.
    """
    h, w = arr.shape
    c0, c1 = int(w * 0.40), int(w * 0.60)     # striscia centrale (colonna/mediastino)
    center = arr[:, c0:c1].mean()
    left   = arr[:, :int(w * 0.25)].mean()    # polmone sx
    right  = arr[:, int(w * 0.75):].mean()    # polmone dx
    return center - 0.5 * (left + right)


# --------------------------- riferimento Kermany ---------------------------
def build_kermany_reference():
    """Campiona immagini Kermany e costruisce:
       - ref_stack: array impilato per l'histogram matching (distribuzione aggregata)
       - kermany_sign: segno atteso della statistica di polarita' (+1 tipicamente).
    """
    paths = []
    for split in SPLITS:
        for cls in CLASSES:
            d = os.path.join(KERMANY_ROOT, split, cls)
            if os.path.isdir(d):
                for e in IMG_EXTS:
                    paths.extend(glob.glob(os.path.join(d, f"*{e}")))
    if not paths:
        raise RuntimeError(f"Nessuna immagine Kermany trovata sotto {KERMANY_ROOT}")

    random.shuffle(paths)
    paths = paths[:REF_SAMPLE]

    stack, signs = [], []
    for p in paths:
        g = load_gray(p)
        g_rs = np.asarray(
            Image.fromarray((g * 255).astype(np.uint8)).resize((REF_SIZE, REF_SIZE)),
            dtype=np.float32) / 255.0
        stack.append(g_rs)
        signs.append(polarity_stat(g_rs))

    ref_stack = np.concatenate([s.reshape(-1) for s in stack]).reshape(-1, 1)
    kermany_sign = 1.0 if np.median(signs) >= 0 else -1.0
    print(f"  Riferimento Kermany: {len(paths)} img | "
          f"statistica polarita' mediana = {np.median(signs):+.4f} "
          f"(segno atteso {kermany_sign:+.0f})")
    return ref_stack.squeeze(), kermany_sign


# --------------------------- pipeline per immagine ---------------------------
def process_image(arr, ref_flat, kermany_sign, stats):
    """Applica l'intera pipeline a una singola immagine grayscale [0,1]."""
    # 1) center-crop (FOV) prima di tutto, cosi' la statistica di polarita' e' piu' pulita
    if DO_CENTER_CROP:
        arr = center_crop(arr, CENTER_CROP_FRAC)

    # 2) correzione polarita'
    invert = False
    if DO_INVERT is True:
        invert = True
    elif DO_INVERT == "auto":
        invert = (np.sign(polarity_stat(arr)) != kermany_sign)
    if invert:
        arr = 1.0 - arr
        stats["inverted"] += 1

    # 3) histogram matching alla distribuzione Kermany
    if DO_HISTMATCH:
        matched = match_histograms(arr.reshape(-1), ref_flat)
        arr = matched.reshape(arr.shape).astype(np.float32)

    # 4) CLAHE (opzionale, richiede scikit-image)
    if DO_CLAHE:
        try:
            from skimage import exposure
        except ImportError:
            raise ImportError("DO_CLAHE=True richiede scikit-image "
                              "(pip install scikit-image) oppure imposta DO_CLAHE=False")
        arr = exposure.equalize_adapthist(np.clip(arr, 0, 1), clip_limit=CLAHE_CLIP)

    # 5) crop margini (opzionale, per marker/bordi)
    if DO_MARGIN_CROP:
        arr = margin_crop(arr, MARGIN_FRAC)

    # 6) resize finale (opzionale)
    if OUT_SIZE is not None:
        arr = np.asarray(
            Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((OUT_SIZE, OUT_SIZE)),
            dtype=np.float32) / 255.0
    return arr


# --------------------------- driver ---------------------------
def main():
    print("=" * 60)
    print("  Preprocessing VinDr -> stile Kermany")
    print("=" * 60)
    print(f"  Input : {VINDR_ROOT}")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"  Operazioni: invert={DO_INVERT} histmatch={DO_HISTMATCH} "
          f"center_crop={DO_CENTER_CROP}({CENTER_CROP_FRAC}) "
          f"clahe={DO_CLAHE} margin_crop={DO_MARGIN_CROP}")

    ref_flat, kermany_sign = build_kermany_reference()

    stats = {"processed": 0, "inverted": 0}
    qc_pairs = []

    for split in SPLITS:
        for cls in CLASSES:
            src_dir = os.path.join(VINDR_ROOT, split, cls)
            if not os.path.isdir(src_dir):
                print(f"  [skip] cartella assente: {src_dir}")
                continue
            out_dir = os.path.join(OUTPUT_ROOT, split, cls)
            os.makedirs(out_dir, exist_ok=True)

            files = [f for f in os.listdir(src_dir) if f.endswith(IMG_EXTS)]
            for i, fn in enumerate(files):
                src = os.path.join(src_dir, fn)
                orig = load_gray(src)
                out = process_image(orig, ref_flat, kermany_sign, stats)

                stem = os.path.splitext(fn)[0]
                save_gray(out, os.path.join(out_dir, stem + ".png"))
                stats["processed"] += 1

                if SAVE_QC and len(qc_pairs) < QC_SAMPLES and i % 7 == 0:
                    qc_pairs.append((orig, out, f"{split}/{cls}/{fn}"))

            print(f"  {split}/{cls}: {len(files)} immagini processate")

    print("-" * 60)
    print(f"  Totale processate: {stats['processed']}")
    inv_pct = 100.0 * stats["inverted"] / max(stats["processed"], 1)
    print(f"  Invertite (polarita'): {stats['inverted']} ({inv_pct:.1f}%)")

    if SAVE_QC and qc_pairs:
        _save_qc(qc_pairs)

    print(f"\n  Fatto. Ora in main_dann_cross_hospital.py imposta:")
    print(f"      TARGET_ROOT = \"{OUTPUT_ROOT}\"")
    print(f"  e rilancia:  python experiments/main_dann_cross_hospital.py")


def _save_qc(pairs):
    """Salva un montaggio prima/dopo (2 righe: originale sopra, processata sotto)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  [QC] matplotlib assente, salto il montaggio.")
        return
    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, (orig, out, name) in enumerate(pairs):
        axes[0, j].imshow(orig, cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(name, fontsize=7)
        axes[0, j].axis("off")
        axes[1, j].imshow(out, cmap="gray", vmin=0, vmax=1)
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("originale")
    axes[1, 0].set_ylabel("processata")
    plt.suptitle("VinDr: originale (sopra) vs preprocessata stile Kermany (sotto)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    qc_path = os.path.join(OUTPUT_ROOT, "_qc_before_after.png")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    plt.savefig(qc_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [QC] Montaggio salvato: {qc_path}  (controlla che la polarita' sia corretta!)")


if __name__ == "__main__":
    main()
