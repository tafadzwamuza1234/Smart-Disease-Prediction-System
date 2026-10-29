"""
predict.py
----------
Stage 13: Prediction pipeline for new users.

Loads the saved model bundle (final ensemble model + feature names +
class labels) and exposes a single function, predict_disease(), that:

    * accepts a list of symptom names typed/selected by a user
    * silently ignores/reports symptoms that are not part of the
      training vocabulary instead of crashing
    * builds the exact same 0/1 feature vector used during training
    * returns the predicted disease, its confidence, and the top-3
      most likely diseases with probabilities

This module has no dependency on the training code (train_models.py)
so the Streamlit app only needs this file + the saved model bundle to
run -- it never re-trains anything.
"""

import os

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "ensemble_model.pkl"
)


def load_model_bundle(path=DEFAULT_MODEL_PATH):
    bundle = joblib.load(path)
    return bundle


def normalize_symptom(s):
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def predict_disease(symptom_list, bundle, top_k=3):
    """
    Parameters
    ----------
    symptom_list : list[str]
        Symptoms as typed/selected by the user, e.g.
        ["headache", "fever", "nausea"]. Names are matched against the
        training feature names after light normalization (lowercase,
        spaces -> underscores) so the function accepts either
        "sore throat" or "sore_throat".
    bundle : dict
        The object returned by load_model_bundle().
    top_k : int
        How many top predictions (with probabilities) to return.

    Returns
    -------
    dict with keys:
        predicted_disease, confidence, top_k_predictions,
        recognized_symptoms, unrecognized_symptoms
    """
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    feature_index = {normalize_symptom(f): i for i, f in enumerate(feature_names)}

    recognized = []
    unrecognized = []
    vector = np.zeros((1, len(feature_names)), dtype=int)

    for raw in symptom_list:
        key = normalize_symptom(raw)
        if key in feature_index:
            vector[0, feature_index[key]] = 1
            recognized.append(feature_names[feature_index[key]])
        else:
            unrecognized.append(raw)

    if len(recognized) == 0:
        return {
            "predicted_disease": None,
            "confidence": None,
            "top_k_predictions": [],
            "recognized_symptoms": recognized,
            "unrecognized_symptoms": unrecognized,
            "message": (
                "None of the entered symptoms were recognized. Please "
                "check spelling or choose symptoms from the provided list."
            ),
        }

    if hasattr(model, "predict_proba"):
        vector_df = pd.DataFrame(vector, columns=feature_names)
        proba = model.predict_proba(vector_df)[0]
        classes = model.classes_
        order = np.argsort(proba)[::-1][:top_k]
        top_predictions = [
            {"disease": classes[i], "probability": float(round(proba[i], 4))}
            for i in order
        ]
        predicted_disease = top_predictions[0]["disease"]
        confidence = top_predictions[0]["probability"]
    else:
        predicted_disease = model.predict(vector)[0]
        confidence = None
        top_predictions = [{"disease": predicted_disease, "probability": None}]

    return {
        "predicted_disease": predicted_disease,
        "confidence": confidence,
        "top_k_predictions": top_predictions,
        "recognized_symptoms": recognized,
        "unrecognized_symptoms": unrecognized,
        "message": None,
    }


if __name__ == "__main__":
    # Small manual smoke test when running: python src/predict.py
    bundle = load_model_bundle()
    example = ["headache", "nausea", "not_a_real_symptom"]
    result = predict_disease(example, bundle)
    print(result)
