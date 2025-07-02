from pathlib import Path
from config import models_folder
from interface import BaseModel
from collections import defaultdict
import pickle

path = Path(models_folder)
models = defaultdict(dict)

PICKLE_SUFFIX = ".pkl"


def _load_model(model_path: Path) -> BaseModel:
    with open(model_path, "rb") as file:
        return pickle.load(file)


def init_models() -> None:
    model_types = [folder for folder in path.iterdir()]
    for model_path in model_types:
        current_dict = models[model_path.name]
        for model_file in model_path.iterdir():
            model_name = model_file.name.removesuffix(PICKLE_SUFFIX)
            current_dict[model_name] = _load_model(model_file)


def get_model(model_name: str, model_type: str) -> BaseModel:
    return models[model_type][model_name]


def get_models_via_type(model_type: str) -> list[str]:
    return list(models[model_type].keys())


def get_all_models() -> dict[str, list[str]]:
    return {current_model_type: list(models[current_model_type].keys()) for current_model_type in models}

