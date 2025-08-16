from flask import jsonify, request, current_app
from config import PROJECT_NAME, MODEL_MANAGER
from flask import Blueprint

model_blueprint = Blueprint("models", __name__)


@model_blueprint.route("/models", methods=["GET"])
def get_models():
    model_manager = current_app.config[MODEL_MANAGER]
    project_name = request.args.get(PROJECT_NAME, None)
    response = model_manager.get_all_models() if project_name is None else model_manager.get_project_models(project_name)

    return jsonify(response)

