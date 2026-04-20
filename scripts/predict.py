import numpy as np
import pandas as pd
import pickle
import json
import os
import joblib

_MODEL = None
_MODEL_NAME = None
_FEAT_COLS = None
_TARGETS = ['avg_d_mbps', 'avg_u_mbps', 'avg_lat_ms']


def _load_artifacts():
    global _MODEL, _MODEL_NAME, _FEAT_COLS

    if _MODEL is not None:
        return

    base = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')

    with open(os.path.join(base, 'feature_cols2.json'), 'r') as f:
        _FEAT_COLS = json.load(f)

    with open(os.path.join(base, 'best_model_name.txt'), 'r') as f:
        _MODEL_NAME = f.read().strip()

    if _MODEL_NAME == 'Gradient Boosting':
        _MODEL = joblib.load(os.path.join(base, 'gb_models.pkl'))
    elif _MODEL_NAME == 'Random Forest':
        _MODEL = joblib.load(os.path.join(base, 'rf_models.pkl'))
    else:
        raise ValueError("An error occurred")


def predict_speeds(input_dict: dict) -> dict:
    _load_artifacts()

    X = pd.DataFrame([input_dict])

    missing_cols = [col for col in _FEAT_COLS if col not in X.columns]
    if missing_cols:
        raise ValueError(f"Missing required input features: {missing_cols}")

    X = X[_FEAT_COLS].astype(float)

    preds = {}
    for target in _TARGETS:
        pred_log = _MODEL[target].predict(X)[0]
        pred = float(np.expm1(pred_log))
        preds[target] = round(float(np.clip(pred, 0, None)), 2)

    return preds