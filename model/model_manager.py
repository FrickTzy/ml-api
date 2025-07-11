from pathlib import Path
from collections import defaultdict
from typing import Dict, Type
from .model_wrapper import ModelWrapper, SKLearnModel, TFModel, TFLiteModel
from .loader import Loader, SKLearnLoader, TFLoader, TFLiteLoader
import config

PICKLE_SUFFIX = ".pkl"
KERAS_SUFFIX = ".keras"
TFLITE_SUFFIX = ".tflite"


class ModelManager:
    def __init__(self, models_folder: str):
        self.path = Path(models_folder)
        self.models: Dict[str, Dict[str, ModelWrapper]] = defaultdict(dict)
        self.loaders: Dict[str, Type[Loader]] = {
            PICKLE_SUFFIX: SKLearnLoader,
            KERAS_SUFFIX: TFLoader,
            TFLITE_SUFFIX: TFLiteLoader,
        }
        self.model_wrapper: Dict[str, Type[ModelWrapper]] = {
            PICKLE_SUFFIX: SKLearnModel,
            KERAS_SUFFIX: TFModel,
            TFLITE_SUFFIX: TFLiteModel,
        }

    def init_models(self) -> None:
        for model_type_dir in self.path.iterdir():
            if not model_type_dir.is_dir():
                continue
            current_dict = self.models[model_type_dir.name]
            for model_file in model_type_dir.iterdir():
                for suffix, loader in self.loaders.items():
                    if model_file.name.endswith(suffix):
                        model_name = model_file.name.removesuffix(suffix)
                        loaded_model = loader().load_model(model_file)
                        wrapped_model = self.model_wrapper[suffix](loaded_model)
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
