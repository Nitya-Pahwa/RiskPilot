"""
Risk scoring agent: loads the trained fraud model and scores a transaction.
Includes graceful fallback when model is unavailable.
"""
from __future__ import annotations

from riskpilot.core.fraud_state import FraudPipelineState
from riskpilot.core.model_registry import score_transaction, _load_registry


def risk_agent(state: FraudPipelineState) -> FraudPipelineState:
    """
    Node: score the transaction using the trained fraud model.
    Sets state.risk_score and state.model_healthy.
    If model is unavailable, risk_score = -1.0 and model_healthy = False.
    """
    state.add_log("RiskAgent: scoring transaction.")
    try:
        txn_dict = state.transaction.model_dump()
        risk_score, healthy = score_transaction(txn_dict)

        state.risk_score   = risk_score
        state.model_healthy = healthy

        # model version from registry
        reg = _load_registry()
        state.model_version = reg.get("model_version", "unknown")

        if healthy:
            state.add_log(f"RiskAgent: risk_score={risk_score:.4f}, model={state.model_version}")
        else:
            state.add_log("RiskAgent: model unavailable — risk_score set to -1.0")

    except Exception as exc:
        state.risk_score    = -1.0
        state.model_healthy = False
        state.add_log(f"RiskAgent: exception — {exc}")

    return state
