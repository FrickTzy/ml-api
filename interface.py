import pathlib
from typing import Union
from sklearn.base import BaseEstimator
from tensorflow import lite
from typing import Protocol
import numpy as np


class ModelInterpreter:
    """Model Wrapper for TFLite Models"""

    def __init__(self, model_path: pathlib.Path) -> None:
        self.interpreter = lite.Interpreter(str(model_path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()
        self.output = self.interpreter.get_output_details()

    def predict(self, input_array: np.ndarray | list) -> np.ndarray:
        self.interpreter.set_tensor(self.input[0]['index'], input_array)
        self.interpreter.invoke()

        probabilities = self.interpreter.get_tensor(self.output[0]['index'])
        return probabilities


class Model(Protocol):
    def score(self, input_array: np.ndarray, output_array: np.ndarray) -> float:
        pass

    def predict(self, input_array: np.ndarray | list) -> np.ndarray:
        pass

    def fit(self, input_array: np.ndarray, output_array: np.ndarray) -> None:
        pass


BaseModel = Union[BaseEstimator, Model, ModelInterpreter]
