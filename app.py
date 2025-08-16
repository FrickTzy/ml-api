from flask_cors import CORS
from flask import Flask, jsonify, Response
from config import models_folder, MODEL_NAME, MODEL_MANAGER
from model import ModelManager
from routes import blueprints

app = Flask(__name__)
CORS(app)

model_manager = ModelManager(models_folder)
model_manager.init_models()
projects = model_manager.get_projects()

app.config[MODEL_MANAGER] = model_manager

for blueprint in blueprints:
    app.register_blueprint(blueprint)


@app.route("/")
def home() -> Response:
    return jsonify({"API Routes": ["/models"] + ["/" + project for project in projects],
                    "ARGS": [MODEL_NAME]})


if __name__ == '__main__':
    app.run(debug=True)
