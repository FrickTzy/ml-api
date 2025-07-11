import numpy as np
from abc import ABC, abstractmethod
from interface import BaseModel


class ModelWrapper(ABC):
    """
    Represents a model
    A wrapper class to unify different model types
    """
    def __init__(self, model: BaseModel):
        self.model = model

    @abstractmethod
    def predict(self, input_array: np.ndarray) -> np.ndarray:
        ...


class TFModel(ModelWrapper):
    def predict(self, input_array: np.ndarray) -> np.ndarray:
        input_array = input_array / 255.0
        input_array = input_array.reshape(1, 28, 28, 1)
        probabilities = self.model.predict(input_array)
        if probabilities.ndim > 1 and probabilities.shape[0] == 1:
            probabilities = probabilities[0]
        predicted_class = np.argmax(probabilities)
        return np.array([predicted_class])


class TFLiteModel(ModelWrapper):
    def predict(self, input_array: np.ndarray) -> np.ndarray:
        input_array = input_array / 255.0
        input_array = input_array.astype(np.float32)
        input_array = input_array.reshape(1, 28, 28, 1)

        self.model.interpreter.set_tensor(self.model.input[0]['index'], input_array)
        self.model.interpreter.invoke()

        probabilities = self.model.interpreter.get_tensor(self.model.output[0]['index'])

        predicted_class = np.argmax(probabilities)
        return np.array([predicted_class])


class SKLearnModel(ModelWrapper):
    def predict(self, input_array: np.ndarray) -> np.ndarray:
        return np.array(self.model.predict(input_array))

