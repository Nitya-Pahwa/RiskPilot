"""
Model registry: train, save, load, and version the fraud detection model.
Bridges the existing AutoML pipeline to the RiskPilot fraud use-case.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import joblib
import numpy as np

ARTIFACT_DIR    = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
MODEL_PATH      = os.path.join(ARTIFACT_DIR, "fraud_model_best.joblib")
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "fraud_model_preprocessor.joblib")
REGISTRY_PATH   = os.path.join(ARTIFACT_DIR, "fraud_model_registry.json")

# All columns the preprocessor was trained on (includes ID/categorical cols)
# These must match the dataset columns minus target
ALL_COLUMNS = [
    "transaction_id",
    "merchant_id",
    "customer_id",
    "amount",
    "currency",
    "payment_method",
    "timestamp",
    "customer_age_days",
    "customer_txn_count",
    "customer_avg_amount",
    "device_id",
    "device_change_count",
    "ip_risk_score",
    "location_distance",
    "failed_attempts_24h",
    "transactions_1h",
    "transactions_24h",
    "refund_count",
    "chargeback_history",
    "amount_vs_avg_ratio",
    "is_new_device",
    "velocity_risk",
    "high_ip_risk",
    "new_customer",
]

# Default values for ID / categorical columns not present in a live transaction
_COLUMN_DEFAULTS = {
    "transaction_id": "txn_000000",
    "merchant_id":    "mrc_0001",
    "customer_id":    "cust_000000",
    "currency":       "INR",
    "payment_method": "upi",
    "timestamp":      "2024-01-01 00:00:00",
    "device_id":      "dev_000000",
}

# Features used for explainability (numeric/derived signals only)
FEATURE_COLUMNS = [
    "amount",
    "customer_age_days",
    "customer_txn_count",
    "customer_avg_amount",
    "device_change_count",
    "ip_risk_score",
    "location_distance",
    "failed_attempts_24h",
    "transactions_1h",
    "transactions_24h",
    "refund_count",
    "chargeback_history",
    "amount_vs_avg_ratio",
    "is_new_device",
    "velocity_risk",
    "high_ip_risk",
    "new_customer",
]
TARGET_COLUMN = "is_fraud"



def _load_registry() -> dict:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def _save_registry(data: dict) -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_model_version() -> str:
    reg = _load_registry()
    return reg.get("model_version", "none")


def get_model_metrics() -> dict:
    return _load_registry().get("metrics", {})


def is_model_trained() -> bool:
    return os.path.exists(MODEL_PATH)


def train_fraud_model(
    dataset_path: str,
    optuna_trials: int = 30,
    max_retries: int = 1,
    progress_callback=None,
    verbose: bool = True,
) -> dict:
    """
    Train the fraud detection model using the existing AutoML pipeline.
    Returns final metrics dictionary.
    """
    import sys
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    from automl_agent.core.graph import run_pipeline

    if verbose:
        print(f"Starting AutoML fraud model training on: {dataset_path}")
        print(f"Optuna trials: {optuna_trials} | Max retries: {max_retries}")

    final_state = run_pipeline(
        dataset_path=dataset_path,
        target_column=TARGET_COLUMN,
        dataset_name="fraud_model",
        experiment_name="riskpilot_fraud_detection",
        max_retries=max_retries,
        progress_callback=progress_callback,
    )

    if final_state.error:
        raise RuntimeError(f"AutoML pipeline failed: {final_state.error}")

    # ── post-process: load model, attach preprocessor, save to registry ──
    src_model_path = os.path.abspath(final_state.training_result.model_path)
    src_prep_path  = os.path.abspath(final_state.training_result.preprocessor_path)
    dst_model_path = os.path.abspath(MODEL_PATH)
    dst_prep_path  = os.path.abspath(PREPROCESSOR_PATH)

    import shutil
    if src_model_path != dst_model_path and os.path.exists(src_model_path):
        shutil.copy2(src_model_path, dst_model_path)
    elif src_model_path == dst_model_path:
        pass  # already in the right place

    if src_prep_path != dst_prep_path and os.path.exists(src_prep_path):
        shutil.copy2(src_prep_path, dst_prep_path)
    elif src_prep_path == dst_prep_path:
        pass  # already in the right place


    version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics = final_state.evaluation_result.metrics if final_state.evaluation_result else {}
    architecture = final_state.ml_config.architecture if final_state.ml_config else "unknown"

    registry_data = {
        "model_version":  version,
        "architecture":   architecture,
        "trained_at":     datetime.now().isoformat(),
        "dataset_path":   dataset_path,
        "optuna_trials":  optuna_trials,
        "metrics":        metrics,
        "best_params":    final_state.training_result.best_params if final_state.training_result else {},
        "model_path":     MODEL_PATH,
        "preprocessor_path": PREPROCESSOR_PATH,
    }
    _save_registry(registry_data)

    if verbose:
        print(f"\nModel trained successfully!")
        print(f"Version:      {version}")
        print(f"Architecture: {architecture}")
        print(f"Metrics:      {json.dumps(metrics, indent=2)}")

    return registry_data


def load_model():
    """Load trained fraud classifier. Returns (model, preprocessor_dict, registry)."""
    if not is_model_trained():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run `python run_riskpilot.py --train` first."
        )
    model     = joblib.load(MODEL_PATH)
    prep_data = joblib.load(PREPROCESSOR_PATH)
    registry  = _load_registry()
    return model, prep_data, registry


def score_transaction(features_dict: dict) -> tuple[float, bool]:
    """
    Score a single transaction dict.
    Returns (risk_score: float, model_healthy: bool).
    Gracefully returns (-1.0, False) if model is unavailable.
    """
    try:
        import warnings
        import pandas as pd
        model, prep_data, _ = load_model()
        preprocessor = prep_data["preprocessor"]

        # Build a full row with all preprocessor-expected columns
        row = {**_COLUMN_DEFAULTS}
        for col in ALL_COLUMNS:
            if col in features_dict:
                row[col] = features_dict[col]

        X = pd.DataFrame([row])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_t = preprocessor.transform(X)
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X_t)[0][1])
            else:
                pred = int(model.predict(X_t)[0])
                prob = float(pred)

        return prob, True

    except FileNotFoundError:
        return -1.0, False
    except Exception:
        return -1.0, False



def get_feature_importances() -> dict[str, float]:
    """Return feature → importance mapping from the trained model."""
    try:
        model, _, _ = load_model()
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            return dict(zip(FEATURE_COLUMNS, [float(v) for v in imp]))
        if hasattr(model, "coef_"):
            coef = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            total = coef.sum() or 1.0
            return dict(zip(FEATURE_COLUMNS, [float(v / total) for v in coef]))
    except Exception:
        pass
    return {col: 1.0 / len(FEATURE_COLUMNS) for col in FEATURE_COLUMNS}
