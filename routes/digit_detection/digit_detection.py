from flask import jsonify, request, Response, current_app
from config import DIGIT_DETECTION, MODEL_NAME, MODEL_MANAGER
import numpy as np
from flask import Blueprint

digit_detection_blueprint = Blueprint("digit_detection", __name__)


@digit_detection_blueprint.route(f"/{DIGIT_DETECTION}", methods=["POST"])
def digit_detection() -> tuple[Response, int] | Response:
    model_manager = current_app.config[MODEL_MANAGER]
    model_name = request.args.get(MODEL_NAME, None)

    if model_name is None:
        return jsonify({"success": False, "error": "Incomplete argument."}), 400

    model = model_manager.get_model(DIGIT_DETECTION, model_name)

    input_array = request.get_json()
    if not isinstance(input_array, list):
        return jsonify({"success": False, "error": "Input must be a list of lists."}), 400

    try:
        prediction = model.predict(np.array(input_array)).tolist()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({
        "success": True,
        "project_name": DIGIT_DETECTION,
        "model": model_name,
        "prediction": prediction
    })
