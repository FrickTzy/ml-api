from model.model_wrapper import PredictionWrapper
import numpy as np
from typing import Dict, Type
from model.suffix import PICKLE_SUFFIX, KERAS_SUFFIX, TFLITE_SUFFIX


class TFModel(PredictionWrapper):
    def predict(self, input_array: np.ndarray) -> np.ndarray:
        input_array = input_array / 255.0
        input_array = input_array.reshape(1, 28, 28, 1)
        probabilities = self.model.predict(input_array)
        if probabilities.ndim > 1 and probabilities.shape[0] == 1:
            probabilities = probabilities[0]
        predicted_class = np.argmax(probabilities)
        return np.array([predicted_class])


class TFLiteModel(PredictionWrapper):
    def predict(self, input_array: np.ndarray) -> np.ndarray:
        input_array = input_array / 255.0
        input_array = input_array.astype(np.float32)
        input_array = input_array.reshape(1, 28, 28, 1)

        probabilities = self.model.predict(input_array)

        predicted_class = np.argmax(probabilities)
        return np.array([predicted_class])


class SKLearnModel(PredictionWrapper):
    def predict(self, input_array: np.ndarray) -> np.ndarray:
        return np.array(self.model.predict(input_array))


digit_detection_models: Dict[str, Type[PredictionWrapper]] = {
    PICKLE_SUFFIX: SKLearnModel,
    KERAS_SUFFIX: TFModel,
    TFLITE_SUFFIX: TFLiteModel,
}