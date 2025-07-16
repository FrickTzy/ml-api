from flask_cors import CORS
from flask import Flask, jsonify, request, Response
from config import models_folder
from model import ModelManager
import numpy as np

app = Flask(__name__)
CORS(app)

model_manager = ModelManager(models_folder)
model_manager.init_models()
projects = model_manager.get_projects()

PROJECT_NAME = "projectName"
MODEL_NAME = "modelName"


@app.route("/")
def home() -> Response:
    return jsonify({"API Routes": ["/predict", "/get_models"], "ARGS": [PROJECT_NAME, MODEL_NAME],
                    "JSON": "Used for data input.", "Projects": projects})


@app.route("/predict", methods=["POST"])
def predict() -> tuple[Response, int] | Response:
    project_name = request.args.get(PROJECT_NAME, None)
    model_name = request.args.get(MODEL_NAME, None)

    if project_name is None or model_name is None:
        return jsonify({"success": False, "error": "Incomplete argument."}), 400

    model = model_manager.get_model(model_name, project_name)

    input_array = request.get_json()
    if not isinstance(input_array, list):
        return jsonify({"success": False, "error": "Input must be a list of lists."}), 400

    try:
        prediction = model.predict(np.array(input_array)).tolist()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({
        "success": True,
        "project_name": project_name,
        "model": model_name,
        "prediction": prediction
    })


@app.route("/models", methods=["GET"])
def get_models():
    project_name = request.args.get(PROJECT_NAME, None)
    response = model_manager.get_all_models() if project_name is None else model_manager.get_project_models(project_name)

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
