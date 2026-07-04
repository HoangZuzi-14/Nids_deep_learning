import numpy as np


def controlled_undersample(X, y, max_samples_per_class: int = 300000, seed: int = 42):
    rng = np.random.default_rng(seed)
    selected = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        if len(idx) > max_samples_per_class:
            idx = rng.choice(idx, size=max_samples_per_class, replace=False)
        selected.append(idx)
    selected_idx = np.concatenate(selected)
    rng.shuffle(selected_idx)
    return X[selected_idx], y[selected_idx]

