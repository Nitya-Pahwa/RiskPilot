"""Fraud pipeline Pydantic state models."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """Raw + derived features for a single transaction."""
    transaction_id:       str
    merchant_id:          str
    customer_id:          str
    amount:               float
    currency:             str   = "INR"
    payment_method:       str   = "upi"
    timestamp:            str   = ""
    customer_age_days:    int   = 0
    customer_txn_count:   int   = 0
    customer_avg_amount:  float = 0.0
    device_change_count:  int   = 0
    ip_risk_score:        float = 0.0
    location_distance:    float = 0.0
    failed_attempts_24h:  int   = 0
    transactions_1h:      int   = 0
    transactions_24h:     int   = 0
    refund_count:         int   = 0
    chargeback_history:   int   = 0
    # derived
    amount_vs_avg_ratio:  float = 0.0
    is_new_device:        int   = 0
    velocity_risk:        int   = 0
    high_ip_risk:         int   = 0
    new_customer:         int   = 0


class RiskExplanation(BaseModel):
    """Structured explanation for a risk decision."""
    top_features:      List[Dict[str, Any]] = Field(default_factory=list)
    narrative:         str = ""          # LLM-generated human-readable text
    feature_scores:    Dict[str, float]  = Field(default_factory=dict)


class AuditEntry(BaseModel):
    """Single audit-log record."""
    transaction_id:   str
    merchant_id:      str
    amount:           float
    risk_score:       float
    risk_label:       str    # LOW / MEDIUM / HIGH
    decision:         str    # ALLOW / MANUAL_REVIEW / FLAG
    policy_triggered: str
    top_reasons:      List[str]         = Field(default_factory=list)
    model_version:    str               = "unknown"
    model_healthy:    bool              = True
    timestamp:        str               = ""
    explanation:      str               = ""


class FraudPipelineState(BaseModel):
    """State object that flows through the LangGraph fraud detection graph."""
    # input
    transaction:     TransactionFeatures

    # intermediate
    risk_score:      float                  = -1.0
    risk_label:      str                    = "UNKNOWN"   # LOW / MEDIUM / HIGH
    decision:        str                    = "UNKNOWN"   # ALLOW / MANUAL_REVIEW / FLAG
    policy_triggered:str                    = ""
    explanation:     Optional[RiskExplanation] = None
    model_healthy:   bool                   = True
    model_version:   str                    = "unknown"

    # output
    audit_entry:     Optional[AuditEntry]   = None

    # meta
    error:           Optional[str]          = None
    logs:            List[str]              = Field(default_factory=list)

    def add_log(self, msg: str) -> "FraudPipelineState":
        self.logs.append(msg)
        return self
