import abc
from typing import Any, Dict

import numpy as np


class DataProcessor:
    @abc.abstractmethod
    def encode(self, example: Dict[str, Any]) -> Dict[str, np.ndarray]:
        pass
