from flask_cors import CORS
from flask import Flask, jsonify, request, Response
from config import CLASSIFICATION, DEFAULT_REGRESSION, DEFAULT_CLASSIFICATION, models_folder
from model import ModelManager
import numpy as np

app = Flask(__name__)
CORS(app)

model_manager = ModelManager(models_folder)
model_manager.init_models()


MODEL_TYPE = "modelType"
MODEL_NAME = "modelName"


@app.route("/")
def home() -> Response:
    return jsonify({"API Routes": ["/predict", "/get_models"], "ARGS": [MODEL_TYPE, MODEL_NAME],
                    "JSON": "Used for data input."})


@app.route("/predict", methods=["POST"])
def predict() -> tuple[Response, int] | Response:
    model_type = request.args.get(MODEL_TYPE, CLASSIFICATION)
    model_name = request.args.get(
        MODEL_NAME,
        DEFAULT_CLASSIFICATION if model_type == CLASSIFICATION else DEFAULT_REGRESSION
    )

    model = model_manager.get_model(model_name, model_type)

    input_array = request.get_json()
    if not isinstance(input_array, list):
        return jsonify({"success": False, "error": "Input must be a list of lists."}), 400

    try:
        prediction = model.predict(np.array(input_array)).tolist()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({
        "success": True,
        "type": model_type,
        "model": model_name,
        "prediction": prediction
    })


@app.route("/models", methods=["GET"])
def get_models():
    model_type = request.args.get(MODEL_TYPE, None)
    response = model_manager.get_all_models() if model_type is None else model_manager.get_models_via_type(model_type)

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
