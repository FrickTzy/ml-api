from flask import jsonify, request, Response, current_app
from config import ANIME_RECOMMENDATION, MODEL_NAME, MODEL_MANAGER
from pathlib import Path
from sklearn.preprocessing import Normalizer
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
import numpy as np
import pickle
from flask import Blueprint
from typing import TypedDict

anime_recommendation_blueprint = Blueprint("anime_recommendation", __name__)

CURRENT_DIRECTORY = Path(__file__).resolve().parent

with open(CURRENT_DIRECTORY / "anime_info.pkl", "rb") as file:
    anime_df = pickle.load(file)

with open(CURRENT_DIRECTORY / "vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

GENRE = "Genres"
ID = "ID"
NAME = "Name"
TYPE = "Type"
RATE = "Rate"
SCORE = "Score"
SYNOPSIS = "Synopsis"
IMAGE = "Image URL"

genres = ['Genres_Action', 'Genres_Adventure', 'Genres_Avant Garde', 'Genres_Award Winning', 'Genres_Boys Love',
          'Genres_Comedy', 'Genres_Drama', 'Genres_Ecchi', 'Genres_Erotica', 'Genres_Fantasy', 'Genres_Girls Love',
          'Genres_Gourmet', 'Genres_Hentai', 'Genres_Horror', 'Genres_Mystery', 'Genres_Romance', 'Genres_Sci-Fi',
          'Genres_Slice of Life', 'Genres_Sports', 'Genres_Supernatural', 'Genres_Suspense', 'Genres_UNKNOWN']

types = ['Type_Movie', 'Type_Music', 'Type_ONA', 'Type_OVA', 'Type_Special', 'Type_TV', 'Type_UNKNOWN', 'Type_nan']

max_words = 2000

normalizer = Normalizer()


class AnimeInfo(TypedDict):
    ID: int
    Genres: str
    Score: float
    Synopsis: str
    Type: str


def _get_input_array(favorite_genre: list[str], favorite_anime: list[AnimeInfo], favorite_type: str, anime_rating: int) -> np.ndarray:
    input_vector = []
    input_vector.extend([1 if genre.removeprefix("Genres_") in favorite_genre else 0 for genre in genres])
    input_vector.extend([1 if anime_type.removeprefix("Type_") == favorite_type else 0 for anime_type in types])
    input_vector.append(anime_rating)

    favorite_anime_weight = 0.5
    if len(favorite_anime) == 0:
        combined_vector = input_vector
        mean_synopsis = csr_matrix(np.zeros((1, max_words)))
    else:
        favorite_genres = []
        favorite_types = []
        favorite_score = []
        favorite_synopsis = []
        for anime in favorite_anime:
            target_genres = set([genre.strip() for genre in anime[GENRE].split(",")])
            target_type = anime[TYPE]
            favorite_genres.append([1 if genre.replace("Genres_", "") in target_genres else 0 for genre in genres])
            favorite_types.append([1 if anime_type.removeprefix("Type_") == target_type else 0 for anime_type in types])
            favorite_score.append(anime[SCORE])
            favorite_synopsis.append(anime[SYNOPSIS])

        mean_genre_vector = np.array(favorite_genres).mean(axis=0) * favorite_anime_weight
        mean_type_vector = np.array(favorite_types).mean(axis=0) * favorite_anime_weight

        mean_score = np.array(favorite_score).mean() * favorite_anime_weight

        mean_vector = np.concatenate([mean_genre_vector, mean_type_vector, [mean_score]]).reshape(1, -1)

        favorite_synopsis = vectorizer.transform(favorite_synopsis)

        mean_synopsis = favorite_synopsis.mean(axis=0)

        combined_vector = input_vector + mean_vector
        combined_vector = np.clip(combined_vector, 0, 1)
        combined_vector = normalizer.transform(combined_vector)

    input_vector_sparse = csr_matrix(combined_vector)
    input_combined = hstack([mean_synopsis, input_vector_sparse])

    return input_combined


@anime_recommendation_blueprint.route(f"/{ANIME_RECOMMENDATION}", methods=["POST"])
def anime_recommendation() -> tuple[Response, int] | Response:
    # two ways, either skip the anime that isn't in the dataset, or make it so that the anime info is put in the json
    # there's also two keys in id, so maybe rename it to be the same

    model_name = request.args.get(MODEL_NAME, None)

    if model_name is None:
        return jsonify({"success": False, "error": "Incomplete argument."}), 400

    model_manager = current_app.config[MODEL_MANAGER]
    model = model_manager.get_model(ANIME_RECOMMENDATION, model_name)

    input_dict = request.get_json()
    if not isinstance(input_dict, dict):
        return jsonify({"success": False, "error": "Incomplete Input."}), 400

    favorite_genre = input_dict.get("favorite_genre", [])

    favorite_anime: list[AnimeInfo] = input_dict.get("favorite_anime", [])
    # this contains anime id, anime genre, anime synopsis, anime type, anime rate

    neighbors = input_dict.get("neighbors", 10) + len(favorite_anime)

    favorite_id = [anime[ID] for anime in favorite_anime]

    anime_type = input_dict.get("anime_type", "")
    anime_rating = input_dict.get("anime_rating", 1)  # should be 0 - 1

    input_array = _get_input_array(favorite_genre, favorite_anime, anime_type, anime_rating)

    try:
        indices = model.neighbors(input_array, k_neighbors=neighbors).tolist()
        favorite_id_set = set(favorite_id)

        similar_anime = anime_df.iloc[indices].copy()
        similar_anime = similar_anime[~similar_anime[ID].isin(favorite_id_set)]
        similar_anime = similar_anime.sort_values(by=SCORE, ascending=False)
        similar_anime = similar_anime[:neighbors]

        labels = [ID, RATE, NAME, IMAGE, GENRE]
        anime_ids = similar_anime[labels].to_dict(orient='records')

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({
        "success": True,
        "project_name": ANIME_RECOMMENDATION,
        "model": model_name,
        "neighbors": anime_ids
    })