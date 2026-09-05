"""
Audit logger: writes every risk decision to a JSONL audit trail.
Every automated decision is recorded with full context — no silent failures.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from riskpilot.core.fraud_state import FraudPipelineState, AuditEntry

AUDIT_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "artifacts", "audit_log.jsonl"
)


def _ensure_dir():
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)


def audit_logger_agent(state: FraudPipelineState) -> FraudPipelineState:
    """
    Node: persist the risk decision to the JSONL audit log.
    Runs at the end of every transaction evaluation — never skipped.
    """
    state.add_log("AuditLogger: recording decision.")
    _ensure_dir()

    try:
        top_reasons = []
        if state.explanation:
            top_reasons = [
                f["feature"]
                for f in state.explanation.top_features
                if f.get("is_suspicious")
            ][:5]

        narrative = ""
        if state.explanation and state.explanation.narrative:
            narrative = state.explanation.narrative

        entry = AuditEntry(
            transaction_id=  state.transaction.transaction_id,
            merchant_id=     state.transaction.merchant_id,
            amount=          state.transaction.amount,
            risk_score=      round(state.risk_score, 4),
            risk_label=      state.risk_label,
            decision=        state.decision,
            policy_triggered=state.policy_triggered,
            top_reasons=     top_reasons,
            model_version=   state.model_version,
            model_healthy=   state.model_healthy,
            timestamp=       datetime.now().isoformat(),
            explanation=     narrative,
        )
        state.audit_entry = entry

        # Append to JSONL (one JSON object per line)
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.model_dump()) + "\n")

        state.add_log(
            f"AuditLogger: logged → {entry.decision} | "
            f"score={entry.risk_score} | policy={entry.policy_triggered}"
        )

    except Exception as exc:
        state.add_log(f"AuditLogger: failed to write audit entry ({exc})")

    return state


# ─── read helpers ─────────────────────────────────────────────────────────────

def load_audit_log(n: int = 500) -> list[dict]:
    """Load last n audit entries. Returns empty list if no log exists."""
    _ensure_dir()
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    entries = []
    with open(AUDIT_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries[-n:]


def clear_audit_log() -> None:
    """Clear the audit log (for demo resets)."""
    if os.path.exists(AUDIT_LOG_PATH):
        os.remove(AUDIT_LOG_PATH)
