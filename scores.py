

scores = {"SVC": 0.9836983698369837,
          "RadiusNeighborsClassifier": 0.9190919091909191,
          "KNeighborsClassifier": 0.9687968796879688}


def get_score(model_name: str) -> float:
    return scores[model_name]