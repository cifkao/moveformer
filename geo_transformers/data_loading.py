import abc
from typing import Any, Dict

import numpy as np
import torch


class DataProcessor:
    @abc.abstractmethod
    def encode(self, example: Dict[str, Any]) -> Dict[str, np.ndarray]:
        pass


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_raw, preprocess_fn):
        self.dataset_raw = dataset_raw
        self.preprocess_fn = preprocess_fn

    def __getitem__(self, idx):
        return self.preprocess_fn(self.dataset_raw[idx])

    def __len__(self):
        return len(self.dataset_raw)
