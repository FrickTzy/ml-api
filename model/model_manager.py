from pathlib import Path
from typing import Dict
from .model_wrapper import ModelWrapper
from .projects.digit_detection import digit_detection_models
from model.projects.anime_recommendation import anime_recommendation_models
from .loader import loaders
import config

PRIORITY_MODELS = {"Neural Network", "SVC"}

project_to_models = {"anime_recommendation": anime_recommendation_models,
                     "digit_detection": digit_detection_models}


class ModelManager:
    def __init__(self, models_folder: str):
        self.path = Path(models_folder)

        self.models: Dict[str, Dict[str, ModelWrapper]] = {}

    def init_models(self) -> None:
        for project_path in self.path.iterdir():
            if not project_path.is_dir():
                continue
            project_name = project_path.name
            if project_name not in self.models:
                self.models[project_name] = {}

            models = project_to_models[project_name]
            current_project = self.models[project_name]

            for model_file in project_path.iterdir():
                for suffix, loader in loaders.items():
                    if not model_file.name.endswith(suffix):
                        continue

                    model_name = model_file.name.removesuffix(suffix)
                    loaded_model = loader().load_model(model_file)
                    wrapped_model = models[suffix](loaded_model)
                    current_project[model_name] = wrapped_model
                    break

    def get_project_models(self, project_name: str) -> list[str]:
        models = list(self.models[project_name].keys())

        # prioritize priority get_models, so it's at the start of the list
        models = sorted(models, key=lambda x: 0 if x in PRIORITY_MODELS else 1)

        return models

    def get_model(self, project_name: str, model_name: str) -> ModelWrapper:
        return self.models[project_name][model_name]

    def get_all_models(self) -> dict[str, list[str]]:
        return {project_name: list(models.keys()) for project_name, models in self.models.items()}

    def get_projects(self) -> list[str]:
        return list(self.models.keys())


if __name__ == '__main__':
    model_manager = ModelManager(models_folder=config.models_folder)
    model_manager.init_models()
    print(model_manager.models)
