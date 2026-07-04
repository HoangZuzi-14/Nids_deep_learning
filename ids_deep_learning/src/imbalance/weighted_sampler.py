import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(y):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    class_weights = {cls: 1.0 / count for cls, count in zip(classes, counts)}
    sample_weights = torch.DoubleTensor([class_weights[label] for label in y])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

