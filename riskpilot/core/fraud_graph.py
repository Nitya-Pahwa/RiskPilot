"""
LangGraph fraud detection workflow.

Graph:
  START → transaction_analyzer → risk_model → policy_checker
        → explanation_agent → audit_logger → END

Graceful degradation:
- If risk_model fails → model_healthy=False → policy routes to MANUAL_REVIEW
- If explanation_agent fails → continues (non-critical)
- If audit_logger fails → logs but continues
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from riskpilot.core.fraud_state import FraudPipelineState, TransactionFeatures
from riskpilot.agents.risk_agent import risk_agent
from riskpilot.agents.explanation_agent import explanation_agent
from riskpilot.agents.audit_logger import audit_logger_agent
from riskpilot.agents.policy_engine import apply_policy


# ─── helpers ──────────────────────────────────────────────────────────────────

def _to_state(state_dict: dict | FraudPipelineState) -> FraudPipelineState:
    if isinstance(state_dict, FraudPipelineState):
        return state_dict
    return FraudPipelineState(**state_dict)


# ─── node wrappers ─────────────────────────────────────────────────────────────

def _transaction_analyzer(state_dict: dict) -> dict:
    """Validate and enrich transaction features."""
    state = _to_state(state_dict)
    state.add_log("TransactionAnalyzer: validating transaction.")
    txn = state.transaction
    # Compute derived features if not already set
    if txn.customer_avg_amount > 0:
        txn.amount_vs_avg_ratio = round(txn.amount / txn.customer_avg_amount, 3)
    txn.is_new_device  = int(txn.device_change_count >= 3)
    txn.velocity_risk  = int(txn.transactions_1h > 5)
    txn.high_ip_risk   = int(txn.ip_risk_score > 0.6)
    txn.new_customer   = int(txn.customer_age_days < 30)
    state.transaction  = txn
    state.add_log("TransactionAnalyzer: features enriched.")
    return state.model_dump()


def _risk_model_node(state_dict: dict) -> dict:
    state = _to_state(state_dict)
    result = risk_agent(state)
    return result.model_dump()


def _policy_node(state_dict: dict) -> dict:
    state  = _to_state(state_dict)
    policy = apply_policy(
        risk_score=state.risk_score,
        amount=state.transaction.amount,
        model_healthy=state.model_healthy,
    )
    state.risk_label       = policy.risk_label
    state.decision         = policy.decision
    state.policy_triggered = policy.policy_triggered
    state.add_log(
        f"PolicyEngine: {policy.decision} | policy={policy.policy_triggered} | note={policy.note}"
    )
    return state.model_dump()


def _explanation_node(state_dict: dict) -> dict:
    state  = _to_state(state_dict)
    result = explanation_agent(state)
    return result.model_dump()


def _audit_node(state_dict: dict) -> dict:
    state  = _to_state(state_dict)
    result = audit_logger_agent(state)
    return result.model_dump()


# ─── graph construction ────────────────────────────────────────────────────────

def build_fraud_graph():
    # Use dict as state type so LangGraph passes the full dict through nodes
    graph = StateGraph(dict)

    graph.add_node("transaction_analyzer", _transaction_analyzer)
    graph.add_node("risk_model",           _risk_model_node)
    graph.add_node("policy_checker",       _policy_node)
    graph.add_node("explanation_agent",    _explanation_node)
    graph.add_node("audit_logger",         _audit_node)

    graph.set_entry_point("transaction_analyzer")
    graph.add_edge("transaction_analyzer", "risk_model")
    graph.add_edge("risk_model",           "policy_checker")
    graph.add_edge("policy_checker",       "explanation_agent")
    graph.add_edge("explanation_agent",    "audit_logger")
    graph.add_edge("audit_logger",         END)

    return graph.compile()


# ─── public API ───────────────────────────────────────────────────────────────

_COMPILED_GRAPH = None


def get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_fraud_graph()
    return _COMPILED_GRAPH


def evaluate_transaction(transaction: TransactionFeatures) -> FraudPipelineState:
    """
    Run a single transaction through the full fraud detection graph.
    Returns final FraudPipelineState with risk_score, decision, explanation, audit_entry.
    """
    initial_state = FraudPipelineState(transaction=transaction)
    # Pass as plain dict — LangGraph will route it through each node
    initial_dict = initial_state.model_dump()

    graph = get_graph()
    final_dict = initial_dict.copy()

    for event in graph.stream(initial_dict):
        for _, node_output in event.items():
            if isinstance(node_output, dict):
                final_dict.update(node_output)

    return FraudPipelineState(**final_dict)

