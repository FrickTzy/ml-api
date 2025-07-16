from pathlib import Path
from collections import defaultdict
from typing import Dict, Type
from .model_wrapper import ModelWrapper, SKLearnModel, TFModel, TFLiteModel
from .loader import Loader, SKLearnLoader, TFLoader, TFLiteLoader
import config

PICKLE_SUFFIX = ".pkl"
KERAS_SUFFIX = ".keras"
TFLITE_SUFFIX = ".tflite"

PRIORITY_MODELS = {"Neural Network", "SVC"}


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
        for project_path in self.path.iterdir():
            if not project_path.is_dir():
                continue
            current_dict = self.models[project_path.name]
            for model_file in project_path.iterdir():
                for suffix, loader in self.loaders.items():
                    if model_file.name.endswith(suffix):
                        model_name = model_file.name.removesuffix(suffix)
                        loaded_model = loader().load_model(model_file)
                        wrapped_model = self.model_wrapper[suffix](loaded_model)
                        current_dict[model_name] = wrapped_model
                        break

    def get_model(self, model_name: str, project_name: str) -> ModelWrapper:
        return self.models[project_name][model_name]

    def get_project_models(self, project_name: str) -> list[str]:
        models = list(self.models[project_name].keys())

        # prioritize priority models, so it's at the start of the list
        models = sorted(models, key=lambda x: 0 if x in PRIORITY_MODELS else 1)

        return models

    def get_all_models(self) -> dict[str, list[str]]:
        return {project_name: list(models.keys()) for project_name, models in self.models.items()}

    def get_projects(self) -> list[str]:
        return list(self.models.keys())


if __name__ == '__main__':
    model_manager = ModelManager(models_folder=config.models_folder)
    model_manager.init_models()
    print(model_manager.get_all_models())
