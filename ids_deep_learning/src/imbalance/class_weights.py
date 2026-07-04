import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def compute_balanced_class_weights(y, device=None):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    tensor = torch.tensor(weights, dtype=torch.float32)
    return tensor.to(device) if device is not None else tensor

