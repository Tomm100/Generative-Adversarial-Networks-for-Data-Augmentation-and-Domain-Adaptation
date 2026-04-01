### Dataset

- **Source**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Content**: 5,863 chest X-ray images, organized into train, validation, and test splits
- **Classes**: NORMAL (1,341 train) and PNEUMONIA (3,875 train) — naturally imbalanced (~1:3 ratio)
- **Subset strategy**: 20% of train split is added to validation split, while the remaining 80% is used for training. Test split remains unchanged.

### Architecture

| Component | Architecture | Details |
|---|---|---|
| **Classifier** | ResNet-18 (pre-trained on ImageNet) | Final fully-connected layer replaced with a binary classification head |
| **GAN** | WGAN-GP | Conditional on class label; 5 transposed conv layers; outputs 64×64 grayscale images |

### Training Setup

#### Phase 1 — Classifier Baseline
- **Method**: Fine-tuning of the pre-trained ResNet-18

#### Phase 2 — GAN Training
- **Method**: Training WGAN-GP on the train split

#### Phase 3 — Classifier with Augmented Data
- **Method**: Fine-tuning of the pre-trained ResNet-18 on the train split with augmented data

### Evaluation Metrics

| Metric |
|---|
| **Accuracy** |
| **Precision** |
| **Recall** |
| **F1-Score** |

A **Confusion Matrix** will also be used as a diagnostic tool to provide a detailed breakdown of classification errors per class.


Certo, va benissimo anche via mail. Le allego dunque quali sono i miei dubbi a riguardo

1) Posso utilizzare come classificatore una ResNet18 pre-addestrata su ImageNet?

Per quanto riguarda la validazione durante il training, quale metrica mi conviene guardare oltre alla loss per scegliere l'epoca migliore? 

2) Ho scelto il dataset chest x-ray pneumonia (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Questo è composto nel seguente modo:
- train: 5216 immagini (3875 PNEUMONIA, 1341 NORMAL)
- val: 16 immagini (8 PNEUMONIA, 8 NORMAL)
- test: 624 immagini (390 PNEUMONIA, 234 NORMAL)

Lo split di validazione è sufficiente? Oppure devo splittare da una parte del train per creare un validation set più grande? e in tal caso, questo split non deve essere usato per addestrare la GAN, corretto?

Nella traccia viene specificato che il dataset dev'essere ridotto/sbilanciato. Questo è leggermente sbilanciato, ma non so se sia sufficiente. Inoltre è corretto che la classe negativa(Normal) sia quella minoritaria?

3) Per quanto riguarda la GAN, ho pensato di utilizzare una WGAN-GP con le seguenti caratteristiche:
- Conditional on class label
- 6 conv layers
- Outputs 128x128 grayscale images

È una buona scelta? Oppure mi consiglia di utilizzare un'altra architettura?

Per valutare la qualità delle immagini prodotte dal GAN, pensavo di addestrare il classificatore sulle immagini sintetiche del gan ed utilizzare il test set composto da immagini reali. È una strategia corretta o mi consiglia altre metriche per valuare le immagini del gan?

Nel caso in cui l'augmentation non porti a miglioramenti significativi, ha senso testare su quale percentuale del dataset (es prendendo solo il 20-30% del dataset originale) si ottengono miglioramenti? (la gan però sarebbe sempre addestrata su tutto il dataset, in quanto il 20-30% del dataset originale non sarebbe sufficiente per addestrarla)

Mi scuso in anticipo se mi sono dilungato troppo e le auguro una buona serata.






