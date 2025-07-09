from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict
import numpy as np
import pickle
from tensorflow import keras

import config
from interface import BaseModel

PICKLE_SUFFIX = ".pkl"
KERAS_SUFFIX = ".keras"


class ModelWrapper:
    """
    Represents a model
    A wrapper class to unify different model types
    """
    def __init__(self, model: BaseModel, model_type: str):
        self.model = model
        self.model_type = model_type

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        if self.model_type == "keras":
            # keras.models.Sequential returns probabilities for classification
            probs = self.model.predict(input_array)
            if probs.ndim > 1 and probs.shape[0] == 1:
                probs = probs[0]
            predicted_class = np.argmax(probs)
            return np.array([predicted_class])
        elif self.model_type == "sklearn":
            return np.array(self.model.predict(input_array))
        else:
            # Default fallback
            return np.array(self.model.predict(input_array))


class ModelManager:
    def __init__(self, models_folder: str):
        self.path = Path(models_folder)
        self.models: Dict[str, Dict[str, ModelWrapper]] = defaultdict(dict)
        self.loaders: Dict[str, Callable[[Path], BaseModel]] = {
            PICKLE_SUFFIX: self._load_pickle_model,
            KERAS_SUFFIX: self._load_keras_model,
        }
        self.model_type_map: Dict[str, str] = {
            PICKLE_SUFFIX: "sklearn",
            KERAS_SUFFIX: "keras",
        }

    def _load_pickle_model(self, model_path: Path) -> BaseModel:
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def _load_keras_model(self, model_path: Path) -> BaseModel:
        return keras.models.load_model(model_path)

    def init_models(self) -> None:
        for model_type_dir in self.path.iterdir():
            if not model_type_dir.is_dir():
                continue
            current_dict = self.models[model_type_dir.name]
            for model_file in model_type_dir.iterdir():
                for suffix, loader in self.loaders.items():
                    if model_file.name.endswith(suffix):
                        model_name = model_file.name.removesuffix(suffix)
                        loaded_model = loader(model_file)
                        wrapped_model = ModelWrapper(loaded_model, self.model_type_map[suffix])
                        current_dict[model_name] = wrapped_model
                        break

    def get_model(self, model_name: str, model_type: str) -> ModelWrapper:
        return self.models[model_type][model_name]

    def get_models_via_type(self, model_type: str) -> list[str]:
        return list(self.models[model_type].keys())

    def get_all_models(self) -> dict[str, list[str]]:
        return {model_type: list(models.keys()) for model_type, models in self.models.items()}


if __name__ == '__main__':
    model_manager = ModelManager(models_folder=config.models_folder)
    model_manager.init_models()
    print(model_manager.get_all_models())
