"""
Deterministic policy engine.
The ML model recommends; the policy engine constrains what actions can be taken.
All actions are defense-only (no offensive capabilities).
"""
from __future__ import annotations
from dataclasses import dataclass


# ─── thresholds (easily tuneable) ─────────────────────────────────────────────
ALLOW_THRESHOLD  = 0.30
REVIEW_THRESHOLD = 0.70
HIGH_AMOUNT_INR  = 50_000.0    # extra human-approval gate


@dataclass
class PolicyDecision:
    risk_label:       str   # LOW / MEDIUM / HIGH
    decision:         str   # ALLOW / MANUAL_REVIEW / FLAG
    policy_triggered: str
    note:             str = ""


def apply_policy(risk_score: float, amount: float, model_healthy: bool) -> PolicyDecision:
    """
    Apply deterministic risk policy to an ML risk score.

    Rules (in priority order):
    1. Model unavailable → always MANUAL REVIEW
    2. risk < 0.30       → ALLOW
    3. risk < 0.70       → MANUAL REVIEW
    4. risk ≥ 0.70       → FLAG
    5. FLAG + amount > ₹50k → note: requires human approval

    All outcomes are defensive — no automated blocking of funds.
    The system can RECOMMEND blocking but a human must confirm.
    """
    if not model_healthy:
        return PolicyDecision(
            risk_label="UNKNOWN",
            decision="MANUAL_REVIEW",
            policy_triggered="MODEL_UNAVAILABLE",
            note="Risk model health check failed. Automated decision suspended. No automated action taken.",
        )

    if risk_score < ALLOW_THRESHOLD:
        return PolicyDecision(
            risk_label="LOW",
            decision="ALLOW",
            policy_triggered="LOW_RISK_V1",
        )

    if risk_score < REVIEW_THRESHOLD:
        return PolicyDecision(
            risk_label="MEDIUM",
            decision="MANUAL_REVIEW",
            policy_triggered="MEDIUM_RISK_V1",
            note="Send to analyst queue for review.",
        )

    # HIGH RISK
    note = "High-risk transaction flagged for investigation."
    if amount > HIGH_AMOUNT_INR:
        note += f" Amount ₹{amount:,.0f} exceeds ₹{HIGH_AMOUNT_INR:,.0f} threshold — requires explicit human approval before any action."

    return PolicyDecision(
        risk_label="HIGH",
        decision="FLAG",
        policy_triggered="HIGH_RISK_V1",
        note=note,
    )


def get_policy_summary() -> dict:
    """Return human-readable policy configuration (for dashboard display)."""
    return {
        "allow_threshold":    ALLOW_THRESHOLD,
        "review_threshold":   REVIEW_THRESHOLD,
        "high_amount_gate":   HIGH_AMOUNT_INR,
        "version":            "V1",
        "defense_only":       True,
        "rules": [
            f"score < {ALLOW_THRESHOLD}  → ALLOW",
            f"{ALLOW_THRESHOLD} ≤ score < {REVIEW_THRESHOLD} → MANUAL REVIEW",
            f"score ≥ {REVIEW_THRESHOLD}  → FLAG",
            f"FLAG + amount > ₹{HIGH_AMOUNT_INR:,.0f} → Requires human approval",
            "Model unavailable → MANUAL REVIEW always",
        ],
    }
