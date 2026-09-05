"""
Explanation agent: generates feature-based explanations for risk decisions.

Architecture (safe for financial use):
1. Compute per-feature contribution using model feature importances × feature values
2. Select top contributing features
3. Pass structured signals to LLM → human-readable narrative

The LLM never invents reasons — it only reformats structured signals.
"""
from __future__ import annotations

import os
import sys
from typing import List, Dict, Any

from riskpilot.core.fraud_state import FraudPipelineState, RiskExplanation
from riskpilot.core.model_registry import get_feature_importances, FEATURE_COLUMNS


# ─── feature display helpers ──────────────────────────────────────────────────

FEATURE_LABELS = {
    "amount":               "Transaction amount",
    "customer_age_days":    "Customer account age",
    "customer_txn_count":   "Customer transaction history count",
    "customer_avg_amount":  "Customer average transaction amount",
    "device_change_count":  "Number of device changes",
    "ip_risk_score":        "IP address risk score",
    "location_distance":    "Location distance from usual (km)",
    "failed_attempts_24h":  "Failed payment attempts (24h)",
    "transactions_1h":      "Transactions in last 1 hour",
    "transactions_24h":     "Transactions in last 24 hours",
    "refund_count":         "Recent refund count",
    "chargeback_history":   "Chargeback history count",
    "amount_vs_avg_ratio":  "Amount vs customer average ratio",
    "is_new_device":        "New device detected",
    "velocity_risk":        "High transaction velocity",
    "high_ip_risk":         "High-risk IP detected",
    "new_customer":         "New customer account",
}

# Thresholds for flagging a feature as suspicious
SUSPICIOUS_THRESHOLDS = {
    "ip_risk_score":        0.6,
    "location_distance":    200,
    "failed_attempts_24h":  3,
    "transactions_1h":      5,
    "device_change_count":  3,
    "refund_count":         2,
    "chargeback_history":   1,
    "amount_vs_avg_ratio":  3.0,
    "is_new_device":        0.5,
    "velocity_risk":        0.5,
    "high_ip_risk":         0.5,
    "new_customer":         0.5,
}


def _compute_feature_scores(
    txn_dict: dict,
    importances: dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Compute contribution score per feature:
      contribution = importance × normalized_feature_value
    Returns list sorted by contribution (descending).
    """
    scores = []
    for feat in FEATURE_COLUMNS:
        val   = txn_dict.get(feat, 0)
        imp   = importances.get(feat, 0.0)
        thresh = SUSPICIOUS_THRESHOLDS.get(feat)

        is_suspicious = False
        if thresh is not None:
            is_suspicious = float(val) > float(thresh)

        scores.append({
            "feature":        feat,
            "label":          FEATURE_LABELS.get(feat, feat),
            "value":          val,
            "importance":     round(imp, 4),
            "is_suspicious":  is_suspicious,
            "contribution":   round(imp * (1.0 if is_suspicious else 0.1), 5),
        })

    scores.sort(key=lambda x: x["contribution"], reverse=True)
    return scores


def _rule_based_narrative(top_features: List[Dict[str, Any]], decision: str) -> str:
    suspicious = [f for f in top_features if f["is_suspicious"]]
    if not suspicious:
        return f"Transaction risk assessment: {decision}. All behavioral signals are within normal parameters."

    reasons = []
    for f in suspicious[:5]:
        label = f["label"]
        val   = f["value"]
        if isinstance(val, float):
            reasons.append(f"• {label}: {val:.2f}")
        else:
            reasons.append(f"• {label}: {val}")

    return (
        f"Transaction flagged as {decision} based on {len(suspicious)} "
        f"suspicious signal(s):\n" + "\n".join(reasons)
    )


def _llm_narrative(
    top_features: List[Dict[str, Any]],
    risk_score: float,
    decision: str,
    txn: dict,
) -> str:
    """Generate LLM narrative from structured signals (LLM cannot invent reasons)."""
    try:
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        if root not in sys.path:
            sys.path.insert(0, root)
        from automl_agent.core.llm import get_llm
        llm = get_llm()
        if llm is None:
            return _rule_based_narrative(top_features, decision)

        suspicious = [f for f in top_features if f["is_suspicious"]]
        signal_lines = "\n".join(
            f"- {f['label']}: {f['value']} (importance: {f['importance']:.3f})"
            for f in suspicious[:6]
        )
        if not signal_lines:
            signal_lines = "- No strongly suspicious signals detected"

        prompt = f"""You are a fraud analyst writing a concise, factual risk explanation.

Transaction details:
- Amount: ₹{txn.get('amount', 0):,.2f}
- Risk score: {risk_score:.2f}/1.0
- Decision: {decision}
- Customer age: {txn.get('customer_age_days', 0)} days
- Transactions in last 1h: {txn.get('transactions_1h', 0)}

Top suspicious signals (from ML model feature analysis):
{signal_lines}

Write a 2-3 sentence plain-English explanation of WHY this transaction was assessed as {decision}.
Be specific about the signals. Do NOT invent new reasons beyond what is listed above.
Do NOT use markdown. Just plain text."""

        return llm.invoke(prompt).content.strip()

    except Exception:
        return _rule_based_narrative(top_features, decision)


# ─── main agent node ──────────────────────────────────────────────────────────

def explanation_agent(state: FraudPipelineState) -> FraudPipelineState:
    """
    Node: generate structured + narrative explanation for the risk decision.
    Uses model feature importances — LLM only reformats, never invents.
    """
    state.add_log("ExplanationAgent: generating explanation.")
    try:
        importances  = get_feature_importances()
        txn_dict     = state.transaction.model_dump()
        feature_scores = _compute_feature_scores(txn_dict, importances)

        # top suspicious reasons as strings (for audit)
        suspicious_features = [
            f["feature"] for f in feature_scores if f["is_suspicious"]
        ][:5]

        narrative = _llm_narrative(
            top_features=feature_scores,
            risk_score=state.risk_score,
            decision=state.decision,
            txn=txn_dict,
        )

        state.explanation = RiskExplanation(
            top_features=feature_scores[:8],
            narrative=narrative,
            feature_scores={
                f["feature"]: f["contribution"]
                for f in feature_scores
            },
        )
        state.add_log(f"ExplanationAgent: {len(suspicious_features)} suspicious features identified.")

    except Exception as exc:
        # Explanation failure is non-critical — continue pipeline
        state.add_log(f"ExplanationAgent: skipped ({exc})")
        state.explanation = RiskExplanation(
            narrative="Explanation generation failed — see audit log for raw score.",
        )

    return state
