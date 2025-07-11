from typing import Union
from sklearn.base import BaseEstimator
from tensorflow import lite
from typing import Protocol
import numpy as np


class ModelInterpreter:
    def __init__(self, interpreter: lite.Interpreter):
        self.interpreter = interpreter
        self.input = interpreter.get_input_details()
        self.output = interpreter.get_output_details()


class Model(Protocol):
    def score(self, input_array: np.ndarray, output_array: np.ndarray) -> float:
        pass

    def predict(self, input_array: np.ndarray | list) -> np.ndarray:
        pass

    def fit(self, input_array: np.ndarray, output_array: np.ndarray) -> None:
        pass


BaseModel = Union[BaseEstimator, Model, ModelInterpreter]
