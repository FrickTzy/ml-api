from pathlib import Path
import pickle
from tensorflow import keras
import tensorflow as tf
from abc import ABC, abstractmethod
from interface import BaseModel, ModelInterpreter

PICKLE_SUFFIX = ".pkl"
KERAS_SUFFIX = ".keras"


class Loader(ABC):
    @abstractmethod
    def load_model(self, model_path: Path) -> BaseModel:
        ...


class SKLearnLoader(Loader):
    def load_model(self, model_path: Path) -> BaseModel:
        with open(model_path, "rb") as f:
            return pickle.load(f)


class TFLoader(Loader):
    def load_model(self, model_path: Path) -> BaseModel:
        return keras.models.load_model(model_path)


class TFLiteLoader(Loader):
    def load_model(self, model_path: Path) -> BaseModel:
        return ModelInterpreter(model_path)
