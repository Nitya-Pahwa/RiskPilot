"""
Drift monitor: detects distribution shift between training data and recent transactions.
Uses Population Stability Index (PSI) — the industry standard for model monitoring.

PSI < 0.1   → No significant drift
PSI 0.1–0.2 → Moderate drift — monitor closely
PSI > 0.2   → Significant drift — consider retraining
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "fraud_transactions.csv"
)

MONITOR_FEATURES = [
    "amount",
    "ip_risk_score",
    "transactions_1h",
    "device_change_count",
    "customer_age_days",
    "amount_vs_avg_ratio",
]

PSI_THRESHOLD_WARN   = 0.10
PSI_THRESHOLD_ALERT  = 0.20


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions."""
    # Use expected distribution to define bins
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.unique(np.percentile(expected, percentiles))
    if len(bins) < 2:
        return 0.0

    exp_counts = np.histogram(expected, bins=bins)[0]
    act_counts = np.histogram(actual,   bins=bins)[0]

    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)

    # Avoid log(0)
    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def compute_drift(recent_transactions: List[dict]) -> Dict[str, dict]:
    """
    Compare recent transactions against training data distribution.
    Returns PSI per feature with status and interpretation.
    """
    if not os.path.exists(DATASET_PATH):
        return {}

    try:
        train_df = pd.read_csv(DATASET_PATH)
    except Exception:
        return {}

    if not recent_transactions:
        return {}

    recent_df = pd.DataFrame(recent_transactions)

    results = {}
    for feat in MONITOR_FEATURES:
        if feat not in train_df.columns or feat not in recent_df.columns:
            continue
        try:
            train_vals  = train_df[feat].dropna().values.astype(float)
            recent_vals = recent_df[feat].dropna().values.astype(float)
            if len(recent_vals) < 10:
                continue

            psi_val = _psi(train_vals, recent_vals)
            if psi_val < PSI_THRESHOLD_WARN:
                status = "stable"
                label  = "✅ Stable"
            elif psi_val < PSI_THRESHOLD_ALERT:
                status = "warning"
                label  = "⚠️ Moderate drift"
            else:
                status = "alert"
                label  = "🚨 Significant drift"

            results[feat] = {
                "psi":        round(psi_val, 4),
                "status":     status,
                "label":      label,
                "train_mean": round(float(np.mean(train_vals)), 3),
                "recent_mean":round(float(np.mean(recent_vals)), 3),
            }
        except Exception:
            continue

    return results


def overall_drift_status(drift_results: Dict[str, dict]) -> Tuple[str, str]:
    """
    Returns (status, message) based on worst-case feature drift.
    status: 'stable' | 'warning' | 'alert'
    """
    if not drift_results:
        return "stable", "No drift data available."

    statuses = [v["status"] for v in drift_results.values()]
    if "alert" in statuses:
        n_alert = statuses.count("alert")
        return "alert", (
            f"{n_alert} feature(s) show significant distribution shift. "
            "Model performance may degrade — retraining recommended."
        )
    if "warning" in statuses:
        n_warn = statuses.count("warning")
        return "warning", (
            f"{n_warn} feature(s) show moderate drift. Monitor closely."
        )
    return "stable", "All monitored features are stable. Model distribution is healthy."
