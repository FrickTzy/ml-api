from typing import Union
from sklearn.base import BaseEstimator
from typing import Protocol
import numpy as np


class Model(Protocol):
    def score(self, input_array: np.ndarray, output_array: np.ndarray) -> float:
        pass

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        pass

    def fit(self, input_array: np.ndarray, output_array: np.ndarray) -> None:
        pass


BaseModel = Union[BaseEstimator, Model]
