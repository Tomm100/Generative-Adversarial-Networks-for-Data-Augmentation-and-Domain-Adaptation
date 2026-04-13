"""
Utilità per la riproducibilità degli esperimenti.
"""

import torch
import numpy as np
import random


def set_seed(seed=42):
    """
    Fissa il seed per tutti i generatori di numeri casuali
    per garantire riproducibilità tra diverse esecuzioni.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Rende le operazioni CUDA deterministiche (può rallentare leggermente)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Seed fissato a {seed}")
