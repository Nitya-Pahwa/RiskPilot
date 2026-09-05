"""
Razorpay Connector — Pluggable Integration Stub
================================================
Currently in SIMULATION MODE.
The system works fully offline using synthetic transactions.

To enable live Razorpay Test Mode integration later:
  1. Set USE_RAZORPAY=true in .env
  2. Add RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET to .env
  3. The same evaluate_transaction() API will be called — no other code changes needed.

Razorpay API docs: https://razorpay.com/docs/api/payments/
"""
from __future__ import annotations

import os
from typing import Optional

# ── config ────────────────────────────────────────────────────────────────────
USE_RAZORPAY    = os.getenv("USE_RAZORPAY", "false").lower() == "true"
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")


def is_razorpay_enabled() -> bool:
    """Returns True if Razorpay integration is configured and enabled."""
    return USE_RAZORPAY and bool(RAZORPAY_KEY_ID) and bool(RAZORPAY_SECRET)


def get_razorpay_client():
    """
    Returns an initialized Razorpay client, or None in simulation mode.
    Requires `razorpay` package: pip install razorpay
    """
    if not is_razorpay_enabled():
        return None
    try:
        import razorpay
        return razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_SECRET))
    except ImportError:
        print("razorpay package not installed. Run: pip install razorpay")
        return None


def fetch_payment(payment_id: str) -> Optional[dict]:
    """
    Fetch a payment from Razorpay Test Mode.
    Returns None in simulation mode.

    Example live payload:
      {"id": "pay_xxx", "amount": 5000, "currency": "INR", "method": "upi", ...}
    """
    client = get_razorpay_client()
    if client is None:
        return None
    try:
        return client.payment.fetch(payment_id)
    except Exception as exc:
        print(f"Razorpay fetch_payment error: {exc}")
        return None


def razorpay_payment_to_transaction_features(payment: dict) -> dict:
    """
    Convert a Razorpay payment object to RiskPilot TransactionFeatures dict.
    Called when USE_RAZORPAY=true; maps Razorpay fields → RiskPilot fields.

    ┌─────────────────────────────┬──────────────────────────┐
    │ Razorpay field              │ RiskPilot feature        │
    ├─────────────────────────────┼──────────────────────────┤
    │ id                          │ transaction_id           │
    │ amount / 100                │ amount (INR)             │
    │ method                      │ payment_method           │
    │ contact (hashed)            │ customer_id              │
    │ created_at (unix)           │ timestamp                │
    └─────────────────────────────┴──────────────────────────┘

    Note: Fields like ip_risk_score and location_distance are not
    directly available from Razorpay — they should come from your
    fraud enrichment pipeline (GeoIP lookup, velocity counter, etc.)
    """
    from datetime import datetime
    amount_inr = float(payment.get("amount", 0)) / 100.0
    ts = datetime.fromtimestamp(
        payment.get("created_at", 0)
    ).isoformat() if payment.get("created_at") else ""

    return {
        "transaction_id":    payment.get("id", ""),
        "merchant_id":       payment.get("merchant_id", "rzp_merchant"),
        "customer_id":       str(abs(hash(payment.get("contact", ""))) % 100000),
        "amount":            amount_inr,
        "currency":          payment.get("currency", "INR"),
        "payment_method":    payment.get("method", "upi"),
        "timestamp":         ts,
        # ── these need enrichment from your velocity/geo pipeline ──
        "customer_age_days":   0,
        "customer_txn_count":  0,
        "customer_avg_amount": amount_inr,
        "device_change_count": 0,
        "ip_risk_score":       0.0,
        "location_distance":   0.0,
        "failed_attempts_24h": 0,
        "transactions_1h":     1,
        "transactions_24h":    1,
        "refund_count":        0,
        "chargeback_history":  0,
    }


def integration_status() -> dict:
    """Return integration status for dashboard display."""
    return {
        "mode":     "live" if is_razorpay_enabled() else "simulation",
        "enabled":  is_razorpay_enabled(),
        "key_set":  bool(RAZORPAY_KEY_ID),
        "message":  (
            "✅ Razorpay Test Mode active"
            if is_razorpay_enabled()
            else "🔄 Simulation mode — set USE_RAZORPAY=true in .env to enable live integration"
        ),
    }
