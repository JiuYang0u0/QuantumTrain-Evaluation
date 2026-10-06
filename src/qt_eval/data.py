"""Fixed, stratified digits splits; no fitted preprocessing."""
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def load_digits_splits(split_seed=2026, smoke=False):
    digits = load_digits()
    indices = np.arange(len(digits.target))
    train, rest = train_test_split(indices, test_size=0.4, random_state=split_seed,
                                   stratify=digits.target)
    val, test = train_test_split(rest, test_size=0.5, random_state=split_seed,
                                stratify=digits.target[rest])
    result = {'indices': {}}
    for key, idx in zip(('train', 'validation', 'test'), (train, val, test)):
        if smoke:
            idx, _ = train_test_split(idx, train_size=40, random_state=split_seed,
                                      stratify=digits.target[idx])
        result['indices'][key] = idx.tolist()
        result[key] = TensorDataset(torch.tensor(digits.data[idx]/16, dtype=torch.float64),
                                    torch.tensor(digits.target[idx], dtype=torch.long))
    return result
