from model.model_wrapper import NeighborWrapper
import numpy as np
from typing import Dict, Type
from model.suffix import PICKLE_SUFFIX
import pickle


class SKLearnModel(NeighborWrapper):
    def neighbors(self, input_array: np.ndarray | list, k_neighbors: int) -> np.ndarray:
        _, indices = self.model.kneighbors(input_array, n_neighbors=k_neighbors)
        all_indices = indices[0]

        return all_indices


anime_recommendation_models: Dict[str, Type[NeighborWrapper]] = {
    PICKLE_SUFFIX: SKLearnModel,
}