# Generative Adversarial Networks for Data Augmentation and Domain Adaptation

## 1. Dataset

- **Source**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Content**: 5.856 radiografie toraciche pediatriche in scala di grigi
- **Classes**: NORMAL (classe minoritaria) e PNEUMONIA (classe maggioritaria)

### Distribuzione originale

| Classe | Train | Validation | Test | Totale |
|---|---|---|---|---|
| NORMAL | 1.341 | 8 | 234 | 1.583 |
| PNEUMONIA | 3.875 | 8 | 390 | 4.273 |
| **Totale** | 5.216 | 16 | 624 | 5.856 |

La suddivisione originale presenta un validation set di sole 16 immagini (8 per classe), statisticamente insufficiente per una model selection affidabile.

### Distribuzione ripartizionata (adottata nel progetto)

Per ovviare a questa limitazione, una porzione del training set originale viene trasferita al validation set tramite campionamento stratificato, preservando il rapporto tra le classi. Il test set rimane invariato.

| Classe | Train | Validation | Test | Totale |
|---|---|---|---|---|
| NORMAL | 1.145 | 205 | 234 | 1.584 |
| PNEUMONIA | 3.502 | 380 | 390 | 4.272 |
| **Totale** | 4.647 | 585 | 624 | 5.856 |
| *Percentuale* | 79,35% | 9,99% | 10,66% | 100% |

Lo sbilanciamento risultante nel training set è di circa **1:3** (NORMAL:PNEUMONIA), con un gap di **2.357 immagini** da colmare per la classe minoritaria.

---

## 2. Architettura del Classificatore

| Componente | Architettura | Dettagli |
|---|---|---|
| **Classificatore** | ResNet-18 (pre-trained su ImageNet) | Fine-tuning parziale: solo `layer3`, `layer4` e il layer fully-connected finale (sostituito con un classificatore binario) vengono scongelati. I layer inferiori preservano le feature generiche a basso livello apprese su ImageNet. |

Le immagini, nativamente in scala di grigi, vengono caricate in formato RGB (3 canali) duplicando il canale di luminanza per garantire la compatibilità con il backbone pre-addestrato.

---

## 3. Modelli Generativi

Vengono implementate e confrontate **due architetture GAN condizionali** complementari:

### 3.1 Conditional Deep Convolutional WGAN-GP (cDC-WGAN-GP)

| Proprietà | Valore |
|---|---|
| **Risoluzione** | 128×128 grayscale |
| **Stabilizzazione** | Wasserstein Loss + Gradient Penalty (λ=10) |
| **Critic** | Instance Normalization (no Batch Norm per preservare la GP) + Epsilon Drift Penalty |
| **n_critic** | 5 (5 aggiornamenti Critic per ogni aggiornamento Generator) |

### 3.2 Conditional Deep Convolutional SNGAN (cDC-SNGAN)

| Proprietà | Valore |
|---|---|
| **Risoluzioni testate** | 128×128 e 256×256 grayscale |
| **Stabilizzazione** | Spectral Normalization su tutti i layer del Critic |
| **Loss** | Hinge Loss |
| **n_critic** | 1 (un solo aggiornamento Critic per step, reso possibile dalla Spectral Norm) |

### 3.3 Tecniche Comuni a Entrambe le Architetture

#### PatchGAN Discriminator
Il discriminatore produce una griglia spaziale 16×16 anziché un singolo scalare. Ogni cella valuta l'autenticità di un patch locale dell'immagine, forzando il Generatore a riprodurre texture biologiche ad alta frequenza (strutture costali, opacità polmonari, profilo diaframmatico) in ogni regione della radiografia.

#### Strategia di Bilanciamento Generativo (simil-BAGAN)
Il Generatore viene forzato a produrre batch perfettamente bilanciati tra le classi (50% NORMAL, 50% PNEUMONIA), mentre il Critic viene addestrato sulla distribuzione reale sbilanciata del dataset. Questa asimmetria impedisce al Generatore di collassare sulla classe maggioritaria.

---

## 4. Pipeline Sperimentale

### Phase 1 — Baseline ResNet-18
Addestramento del classificatore sul dataset reale sbilanciato, per stabilire un benchmark di riferimento.

### Phase 2 — Training GAN
Addestramento della GAN condizionale (WGAN-GP o SNGAN) sul training set reale per generare immagini sintetiche della classe minoritaria (NORMAL).

### Phase 3 — Classificatore su Dataset Augmented
Re-addestramento del classificatore sul dataset arricchito con campioni sintetici. Lo **studio di ablazione** varia la percentuale di immagini sintetiche inserite (25%, 50%, 75%, 100%) per individuare il rapporto ottimale reale/sintetico.

### Phase 4 — Analisi del Domain Shift
Estrazione delle feature a 512 dimensioni dal penultimo layer della ResNet-18 e proiezione in uno spazio bidimensionale tramite **PCA**, **t-SNE** e **UMAP** per visualizzare e quantificare il domain gap tra campioni reali e sintetici.

### Phase 5 — Domain Adaptation via DANN
Addestramento di una Domain-Adversarial Neural Network (DANN) per allineare le distribuzioni latenti reale e sintetica tramite un Gradient Reversal Layer (GRL), rendendo il classificatore invariante rispetto al dominio di provenienza.

---

## 5. Studio di Ablazione

L'ablation study è strutturato su **due dimensioni ortogonali**:

### 5.1 Ablazione sull'architettura GAN
Per ciascuna delle tre configurazioni (WGAN 128, SNGAN 128, SNGAN 256), vengono confrontate due varianti:
- **NO PG BG**: Discriminatore standard DCGAN (output scalare 1×1) + distribuzione reale sbilanciata per il Generator
- **SI PG BG**: Discriminatore PatchGAN (output 16×16) + bilanciamento simil-BAGAN (50/50)

Totale: **6 configurazioni** GAN.

### 5.2 Ablazione sulla percentuale di augmentation
Per ciascuna delle 6 configurazioni, il classificatore viene ri-addestrato con il 25%, 50%, 75% e 100% del gap colmato da immagini sintetiche.

### 5.3 Confronto con Baseline tradizionali
I risultati GAN vengono confrontati con:
- **Baseline** (solo dati reali, nessun bilanciamento)
- **Class Weighting** (pesi inversi nella loss function)
- **Random Oversampling** (duplicazione dei campioni minoritari)

---

## 6. Metriche di Valutazione

### Classificazione (downstream)

| Metrica | Motivazione |
|---|---|
| **Macro F1-Score** | Metrica primaria: media armonica non pesata tra Precision e Recall per ogni classe, impedisce al modello di ignorare la classe minoritaria |
| **Accuracy** | Metrica complementare per il confronto generale |
| **Precision per classe** | Misura la purezza delle predizioni per classe |
| **Recall per classe** | Misura la capacità di individuare tutti i campioni positivi |
| **Confusion Matrix** | Strumento diagnostico per analizzare gli errori di classificazione |

### Qualità Generativa

| Metrica | Descrizione |
|---|---|
| **Fréchet Inception Distance (FID)** | Distanza tra le distribuzioni delle feature Inception-v3 per reali e sintetici. Valori inferiori = maggiore somiglianza. |
| **Kernel Inception Distance (KID)** | Stimatore non distorto basato su MMD, particolarmente affidabile con dataset di dimensioni limitate. |

---

## 7. Interpretabilità e Analisi delle Feature

Per comprendere come il classificatore organizza le rappresentazioni interne e quantificare il domain shift, vengono estratte le feature dal penultimo layer della ResNet-18 (spazio a 512 dimensioni) e proiettate in 2D tramite:

| Tecnica | Tipo | Scopo |
|---|---|---|
| **PCA** | Lineare | Identifica le componenti di massima varianza globale |
| **t-SNE** | Non-lineare | Preserva le relazioni di vicinanza locale, evidenzia cluster |
| **UMAP** | Non-lineare | Bilancia topologia locale e globale, evidenzia la struttura a livello di manifold |

Queste visualizzazioni permettono di verificare se i campioni sintetici risiedono in cluster separati da quelli reali e motivano l'applicazione della Domain Adaptation.

---

## 8. Domain Adaptation Avversariale (DANN)

Per mitigare il domain shift residuo tra immagini reali e sintetiche, viene implementata una **Domain-Adversarial Neural Network (DANN)**:

| Componente | Architettura | Ruolo |
|---|---|---|
| **Feature Extractor (G_f)** | ResNet-18 backbone → avgpool → 512-dim | Estrae rappresentazioni latenti |
| **Label Predictor (G_y)** | FC(512 → 2) | Classifica Normal vs Pneumonia (task principale) |
| **Domain Discriminator (G_d)** | MLP(512 → 256 → 128 → 1) + Dropout | Distingue reale da sintetico |
| **Gradient Reversal Layer (GRL)** | Identità in forward, −λ in backward | Forza l'invarianza di dominio |

La loss combinata è:

$$L = L_{task}^{real} + \alpha_{synth} \cdot L_{task}^{synth} - \lambda \cdot L_{domain}$$

dove λ cresce sigmoidalmente da 0 a 1 durante il training (scheduling di Ganin et al., 2016) e α_synth controlla il peso della supervisione sui campioni sintetici (Supervised Domain Adaptation).

---

## 9. Risultati Attesi

1. Le GAN con PatchGAN e simil-BAGAN produrranno campioni di qualità superiore in termini di texture locale
2. L'augmentation generativa migliorerà il Macro F1-Score rispetto alle baseline tradizionali (Class Weighting, Oversampling)
3. Esiste una percentuale ottimale di augmentation (non necessariamente il 100%) che massimizza le performance
4. L'analisi PCA/t-SNE/UMAP rivelerà un domain shift residuo tra reali e sintetici
5. Il DANN ridurrà il domain gap, migliorando ulteriormente la robustezza del classificatore su dati reali
