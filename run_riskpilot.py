"""
RiskPilot entry point.

Usage:
  python run_riskpilot.py                     # full setup: generate data + train model
  python run_riskpilot.py --generate-only     # only generate dataset
  python run_riskpilot.py --train-only        # only train (dataset must exist)
  python run_riskpilot.py --demo              # run a quick demo evaluation
  streamlit run riskpilot/ui/dashboard.py     # launch dashboard (after training)
"""
import argparse
import os
import sys

# Force UTF-8 output on Windows to handle unicode in log messages
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATASET_PATH = os.path.join(ROOT, "data", "fraud_transactions.csv")


def step_generate(n_rows: int = 50_000):
    print("\n" + "=" * 60)
    print("  STEP 1: Generating Synthetic Fraud Dataset")
    print("=" * 60)
    from riskpilot.data.generate_dataset import generate_dataset
    generate_dataset(n_total=n_rows, verbose=True)


def step_train(optuna_trials: int = 30):
    print("\n" + "=" * 60)
    print("  STEP 2: Training Fraud Model via AutoML Pipeline")
    print("=" * 60)
    from riskpilot.core.model_registry import train_fraud_model

    def progress(node_name, state):
        if state.logs:
            print(f"  [{node_name:20s}] {state.logs[-1]}")

    result = train_fraud_model(
        dataset_path=DATASET_PATH,
        optuna_trials=optuna_trials,
        max_retries=1,
        progress_callback=progress,
        verbose=True,
    )
    return result


def step_demo():
    print("\n" + "=" * 60)
    print("  STEP 3: Demo - Evaluating Sample Transactions")
    print("=" * 60)
    from riskpilot.core.fraud_state import TransactionFeatures
    from riskpilot.core.fraud_graph import evaluate_transaction

    demo_transactions = [
        {
            "label": "Low-risk (normal transaction)",
            "transaction_id": "txn_demo_001",
            "merchant_id": "mrc_0042",
            "customer_id": "cust_001234",
            "amount": 850.0,
            "payment_method": "upi",
            "customer_age_days": 365,
            "customer_txn_count": 87,
            "customer_avg_amount": 920.0,
            "device_change_count": 1,
            "ip_risk_score": 0.08,
            "location_distance": 5.0,
            "failed_attempts_24h": 0,
            "transactions_1h": 1,
            "transactions_24h": 3,
            "refund_count": 0,
            "chargeback_history": 0,
        },
        {
            "label": "High-risk (suspicious transaction)",
            "transaction_id": "txn_demo_002",
            "merchant_id": "mrc_0042",
            "customer_id": "cust_001235",
            "amount": 48_000.0,
            "payment_method": "card",
            "customer_age_days": 3,
            "customer_txn_count": 2,
            "customer_avg_amount": 500.0,
            "device_change_count": 5,
            "ip_risk_score": 0.89,
            "location_distance": 1800.0,
            "failed_attempts_24h": 8,
            "transactions_1h": 12,
            "transactions_24h": 35,
            "refund_count": 3,
            "chargeback_history": 2,
        },
    ]

    for demo in demo_transactions:
        print(f"\n  Transaction: {demo['label']}")
        print(f"  Amount: INR {demo['amount']:,.2f} | IP Risk: {demo['ip_risk_score']}")

        txn = TransactionFeatures(
            **{k: v for k, v in demo.items() if k != "label"}
        )
        result = evaluate_transaction(txn)

        label = {"ALLOW": "[ALLOW]", "MANUAL_REVIEW": "[REVIEW]", "FLAG": "[FLAG]"}.get(result.decision, "[?]")
        print(f"  -> Risk Score: {result.risk_score:.4f}")
        print(f"  -> Decision:   {label} {result.decision}")
        print(f"  -> Policy:     {result.policy_triggered}")
        if result.explanation and result.explanation.narrative:
            print(f"  -> Reason:     {result.explanation.narrative[:120]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="RiskPilot Setup & Demo")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--train-only",    action="store_true")
    parser.add_argument("--demo",          action="store_true")
    parser.add_argument("--rows",  type=int, default=50_000, help="Dataset rows")
    parser.add_argument("--trials",type=int, default=30,     help="Optuna trials")
    args = parser.parse_args()

    print("=" * 62)
    print("     RiskPilot -- AI Merchant Risk Agent")
    print("     Razorpay Buildathon 2026 - Track 02")
    print("=" * 62)


    if args.generate_only:
        step_generate(args.rows)
    elif args.train_only:
        if not os.path.exists(DATASET_PATH):
            print(f"❌ Dataset not found: {DATASET_PATH}")
            print("   Run without --train-only to generate first.")
            sys.exit(1)
        step_train(args.trials)
    elif args.demo:
        step_demo()
    else:
        # Full setup
        step_generate(args.rows)
        step_train(args.trials)
        step_demo()
        print("\n" + "═" * 60)
        print("  ✅ Setup complete!")
        print("  Launch dashboard:")
        print("    streamlit run riskpilot/ui/dashboard.py")
        print("═" * 60)


if __name__ == "__main__":
    main()
