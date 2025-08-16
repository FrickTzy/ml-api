import numpy as np
from abc import ABC, abstractmethod
from interface import BaseModel
from typing import Union


class _ModelWrapper(ABC):
    """
    Represents a model
    A wrapper class to unify different model types
    """
    def __init__(self, model: BaseModel):
        self.model = model


class PredictionWrapper(_ModelWrapper):
    """
    Represents a model
    A wrapper class to unify different model types
    """

    @abstractmethod
    def predict(self, input_array: np.ndarray) -> np.ndarray:
        ...


class NeighborWrapper(_ModelWrapper):
    """
    Represents a model
    A wrapper class to unify different model types
    """

    @abstractmethod
    def neighbors(self, input_array: np.ndarray | list, k_neighbors: int) -> np.ndarray:
        ...


ModelWrapper = Union[PredictionWrapper, NeighborWrapper]
