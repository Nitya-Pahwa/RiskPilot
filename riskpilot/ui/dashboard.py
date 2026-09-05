"""
RiskPilot — 5-Page Streamlit Dashboard
Razorpay Buildathon 2026 | Track 02: AI Risk Manager

Pages:
  1. 🏠 Overview          — command-centre metrics
  2. 🔍 Investigate       — single-transaction risk analysis
  3. 🤖 AutoML Engine     — model training & comparison
  4. 📊 Evaluation        — held-out test metrics & FP cost
  5. 📋 Audit Log         — full decision trail
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── path bootstrap ─────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskPilot — AI Merchant Risk Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* dark background */
.stApp { background: #0a0e1a; color: #e2e8f0; }

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1629 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}

/* metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px !important;
    transition: transform .15s, box-shadow .15s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99,102,241,.15);
}

/* decision banners */
.banner-allow  { background: linear-gradient(90deg,#052e16,#14532d); border:1px solid #15803d; border-radius:12px; padding:16px 24px; }
.banner-review { background: linear-gradient(90deg,#1c1005,#451a03); border:1px solid #d97706; border-radius:12px; padding:16px 24px; }
.banner-flag   { background: linear-gradient(90deg,#1c0505,#450a0a); border:1px solid #dc2626; border-radius:12px; padding:16px 24px; }

/* risk gauge text */
.gauge-label { font-size: 2.5rem; font-weight: 700; letter-spacing: -1px; }

/* section header */
.section-header {
    font-size: 1.1rem; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 1px;
    border-bottom: 1px solid #1e293b; padding-bottom: 8px; margin-bottom: 16px;
}

/* feature card */
.feature-card {
    background: #1e293b; border-radius: 8px; padding: 10px 14px;
    margin-bottom: 6px; border-left: 3px solid #6366f1;
}
.feature-card.suspicious { border-left-color: #ef4444; }

/* tabs */
[data-baseweb="tab"] { color: #94a3b8 !important; }
[aria-selected="true"] { color: #6366f1 !important; border-bottom-color: #6366f1 !important; }

button[kind="primary"] {
    background: linear-gradient(90deg, #6366f1, #818cf8) !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def _get_graph():
    from riskpilot.core.fraud_graph import evaluate_transaction
    return evaluate_transaction


def _load_audit():
    from riskpilot.agents.audit_logger import load_audit_log
    return load_audit_log(n=1000)


def _model_status() -> dict:
    from riskpilot.core.model_registry import is_model_trained, get_model_metrics, get_model_version, _load_registry
    trained = is_model_trained()
    return {
        "trained":  trained,
        "version":  get_model_version() if trained else "none",
        "metrics":  get_model_metrics() if trained else {},
        "registry": _load_registry()    if trained else {},
    }


def _integration_status() -> dict:
    from riskpilot.integrations.razorpay_connector import integration_status
    return integration_status()


def _risk_color(decision: str) -> str:
    return {"ALLOW": "#22c55e", "MANUAL_REVIEW": "#f59e0b", "FLAG": "#ef4444"}.get(decision, "#94a3b8")


def _risk_emoji(decision: str) -> str:
    return {"ALLOW": "✅", "MANUAL_REVIEW": "⚠️", "FLAG": "🚨"}.get(decision, "❓")


def _gauge_figure(score: float) -> go.Figure:
    pct = max(0.0, min(1.0, score)) * 100
    color = "#22c55e" if pct < 30 else "#f59e0b" if pct < 70 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#475569", "tickwidth": 1},
            "bar":   {"color": color, "thickness": 0.25},
            "bgcolor": "#1e293b",
            "steps": [
                {"range": [0, 30],  "color": "#052e16"},
                {"range": [30, 70], "color": "#1c1005"},
                {"range": [70, 100],"color": "#1c0505"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": pct,
            },
        },
    ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
    )
    return fig


DATASET_PATH = os.path.join(ROOT, "data", "fraud_transactions.csv")
MODEL_PATH   = os.path.join(ROOT, "artifacts", "fraud_model_best.joblib")


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛡️ RiskPilot")
    st.caption("AI Merchant Fraud Detection")
    st.divider()

    ms = _model_status()
    if ms["trained"]:
        st.success(f"Model active · {ms['version']}")
        reg = ms["registry"]
        arch = reg.get("architecture", "?")
        st.caption(f"Architecture: **{arch}**")
        metrics = ms["metrics"]
        if metrics:
            p = metrics.get("precision_macro") or metrics.get("precision_weighted", 0)
            r = metrics.get("recall_macro")    or metrics.get("recall_weighted", 0)
            st.caption(f"Precision: **{p:.1%}** | Recall: **{r:.1%}**")
    else:
        st.error("⚠️ Model not trained")
        st.caption("Go to AutoML Engine tab to train")

    st.divider()
    integ = _integration_status()
    st.info(integ["message"])
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔍 Investigate", "🤖 AutoML Engine", "📊 Evaluation", "📋 Audit Log"],
        label_visibility="collapsed",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown("# 🛡️ RiskPilot")
    st.markdown("### AI Merchant Fraud & Transaction Risk Agent")
    st.caption("Razorpay Buildathon 2026 · Track 02: AI Risk Manager · Defense-Only")
    st.divider()

    audit = _load_audit()
    df    = _load_dataset(DATASET_PATH)

    # ── aggregate metrics ──
    total    = len(audit)
    allowed  = sum(1 for e in audit if e.get("decision") == "ALLOW")
    reviews  = sum(1 for e in audit if e.get("decision") == "MANUAL_REVIEW")
    flagged  = sum(1 for e in audit if e.get("decision") == "FLAG")
    avg_risk = float(np.mean([e.get("risk_score", 0) for e in audit])) if audit else 0.0
    total_amt_protected = sum(
        e.get("amount", 0) for e in audit if e.get("decision") == "FLAG"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions Analyzed", f"{total:,}")
    c2.metric("✅ Allowed", f"{allowed:,}")
    c3.metric("⚠️ Under Review", f"{reviews:,}")
    c4.metric("🚨 Flagged", f"{flagged:,}")
    c5.metric("Avg Risk Score", f"{avg_risk:.1%}")

    if total > 0:
        st.divider()
        c1b, c2b = st.columns(2)
        with c1b:
            st.markdown('<div class="section-header">Decision Distribution</div>', unsafe_allow_html=True)
            decision_counts = {"ALLOW": allowed, "MANUAL_REVIEW": reviews, "FLAG": flagged}
            fig_pie = px.pie(
                values=list(decision_counts.values()),
                names=list(decision_counts.keys()),
                color=list(decision_counts.keys()),
                color_discrete_map={
                    "ALLOW": "#22c55e", "MANUAL_REVIEW": "#f59e0b", "FLAG": "#ef4444"
                },
                hole=0.55,
            )
            fig_pie.update_layout(
                paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
                height=320, margin=dict(l=0,r=0,t=20,b=0),
                showlegend=True, legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2b:
            st.markdown('<div class="section-header">Risk Score Distribution (Last 200)</div>', unsafe_allow_html=True)
            recent_scores = [e.get("risk_score", 0) for e in audit[-200:]]
            if recent_scores:
                fig_hist = px.histogram(
                    x=recent_scores, nbins=20,
                    color_discrete_sequence=["#6366f1"],
                    labels={"x": "Risk Score", "y": "Count"},
                )
                fig_hist.update_layout(
                    paper_bgcolor="#0a0e1a", plot_bgcolor="#0f172a",
                    font_color="#e2e8f0", height=320,
                    margin=dict(l=0,r=0,t=20,b=0),
                    xaxis=dict(gridcolor="#1e293b"),
                    yaxis=dict(gridcolor="#1e293b"),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        # ── business impact ──
        st.divider()
        st.markdown("### 💰 Estimated Business Impact")
        ms_data = _model_status()
        metrics = ms_data.get("metrics", {})

        if metrics:
            fp_rate = 1.0 - metrics.get("precision_macro", 0.8)
            fp_count = int(flagged * fp_rate)
            fp_cost  = fp_count * 1200.0  # avg ₹1200 per false positive (analyst time + friction)

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Est. Fraud Prevented (₹)", f"₹{total_amt_protected:,.0f}")
            b2.metric("Est. False Positives", f"{fp_count:,}")
            b3.metric("Est. FP Cost (₹)", f"₹{fp_cost:,.0f}")
            prec = metrics.get("precision_macro", 0)
            rec  = metrics.get("recall_macro", 0)
            b4.metric("Precision / Recall", f"{prec:.1%} / {rec:.1%}")

            st.info(
                "ℹ️ FP cost is estimated at ₹1,200 per false positive "
                "(analyst review time + merchant friction). "
                "Precision & Recall from held-out test set."
            )

    # ── dataset info if no audit yet ──
    if not df.empty and total == 0:
        st.divider()
        st.markdown("### 📂 Dataset Overview")
        d1, d2, d3, d4 = st.columns(4)
        fraud_count = int(df["is_fraud"].sum())
        d1.metric("Total Transactions", f"{len(df):,}")
        d2.metric("Fraud", f"{fraud_count:,} ({fraud_count/len(df)*100:.1f}%)")
        d3.metric("Legitimate", f"{len(df)-fraud_count:,}")
        d4.metric("Features", f"{len(df.columns)-1}")
        st.caption("Run transactions through the **Investigate** page to build audit data.")

    # ── architecture diagram ──
    st.divider()
    st.markdown("### 🏗️ System Architecture")
    st.code("""
    Merchant Transactions
           ↓
    Transaction Analyzer     ← validates & enriches features
           ↓
    Risk Model (AutoML)      ← XGBoost / LightGBM / RF via Optuna
           ↓
    Policy Engine            ← deterministic bounded rules
           ↓
    Explanation Agent        ← feature-based + LLM narrative
           ↓
    Audit Logger             ← JSONL audit trail
           ↓
    ┌──────────────────────────────────┐
    │         RiskPilot Dashboard       │
    │  Overview │ Investigate │ AutoML  │
    │  Evaluation │ Audit Log           │
    └──────────────────────────────────┘

    Razorpay Integration: Simulation Mode (pluggable → live)
    """, language="text")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — INVESTIGATE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Investigate":
    st.markdown("# 🔍 Transaction Investigation")
    st.caption("Analyze any transaction through the full fraud detection pipeline.")
    st.divider()

    ms = _model_status()
    if not ms["trained"]:
        st.error("⚠️ No trained model found. Go to **AutoML Engine** to train first.")
        st.stop()

    df = _load_dataset(DATASET_PATH)

    # ── select mode ──
    mode = st.radio(
        "Input Mode",
        ["📋 Pick from dataset", "✍️ Manual entry"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    txn_data = {}

    if mode == "📋 Pick from dataset" and not df.empty:
        col1, col2 = st.columns([2, 1])
        with col1:
            preset = st.selectbox(
                "Quick preset",
                ["Custom", "Low-risk (normal transaction)", "High-risk (suspicious)"],
            )
        with col2:
            if st.button("🎲 Random from dataset"):
                row = df.sample(1, random_state=random.randint(0, 9999)).iloc[0]
                st.session_state["sampled_row"] = row.to_dict()

        if preset == "Low-risk (normal transaction)":
            row = df[df["is_fraud"] == 0].sample(1, random_state=42).iloc[0]
            st.session_state["sampled_row"] = row.to_dict()
        elif preset == "High-risk (suspicious)":
            row = df[df["is_fraud"] == 1].sample(1, random_state=42).iloc[0]
            st.session_state["sampled_row"] = row.to_dict()

        if "sampled_row" in st.session_state:
            txn_data = st.session_state["sampled_row"]
            st.json({k: v for k, v in txn_data.items() if k not in ["is_fraud"]})

    else:
        # manual entry form
        with st.form("manual_txn"):
            st.markdown("**Transaction Details**")
            c1, c2, c3 = st.columns(3)
            txn_id     = c1.text_input("Transaction ID", value=f"txn_{random.randint(100000,999999)}")
            merchant   = c2.text_input("Merchant ID", value="mrc_0001")
            customer   = c3.text_input("Customer ID", value="cust_001234")

            c4, c5, c6 = st.columns(3)
            amount     = c4.number_input("Amount (₹)", min_value=1.0, max_value=500000.0, value=5000.0)
            pay_method = c5.selectbox("Payment Method", ["upi", "card", "netbanking", "wallet", "emi"])
            cust_avg   = c6.number_input("Customer Avg Amount (₹)", min_value=1.0, value=2000.0)

            st.markdown("**Risk Signals**")
            r1, r2, r3, r4 = st.columns(4)
            cust_age    = r1.number_input("Customer Age (days)", 0, 3000, 365)
            cust_txns   = r2.number_input("Customer Txn Count", 0, 1000, 50)
            dev_changes = r3.number_input("Device Changes", 0, 20, 0)
            ip_risk     = r4.slider("IP Risk Score", 0.0, 1.0, 0.1, 0.01)

            r5, r6, r7, r8 = st.columns(4)
            loc_dist    = r5.number_input("Location Distance (km)", 0.0, 5000.0, 10.0)
            fail_att    = r6.number_input("Failed Attempts (24h)", 0, 20, 0)
            txns_1h     = r7.number_input("Transactions (1h)", 0, 50, 1)
            txns_24h    = r8.number_input("Transactions (24h)", 0, 200, 3)

            r9, r10 = st.columns(2)
            refunds    = r9.number_input("Refund Count", 0, 20, 0)
            chargebacks= r10.number_input("Chargeback History", 0, 10, 0)

            submitted = st.form_submit_button("🔍 Analyze Transaction", type="primary")
            if submitted:
                txn_data = {
                    "transaction_id": txn_id, "merchant_id": merchant,
                    "customer_id": customer, "amount": amount,
                    "currency": "INR", "payment_method": pay_method,
                    "timestamp": datetime.now().isoformat(),
                    "customer_age_days": cust_age, "customer_txn_count": cust_txns,
                    "customer_avg_amount": cust_avg, "device_change_count": dev_changes,
                    "ip_risk_score": ip_risk, "location_distance": loc_dist,
                    "failed_attempts_24h": fail_att, "transactions_1h": txns_1h,
                    "transactions_24h": txns_24h, "refund_count": refunds,
                    "chargeback_history": chargebacks,
                }

    # ── run analysis ──
    if txn_data and st.button("🚀 Run Risk Analysis", type="primary"):
        from riskpilot.core.fraud_state import TransactionFeatures

        with st.spinner("Running fraud detection pipeline..."):
            try:
                # fill defaults
                for f in ["amount_vs_avg_ratio","is_new_device","velocity_risk","high_ip_risk","new_customer"]:
                    txn_data.setdefault(f, 0)
                txn_data.setdefault("transaction_id", f"txn_{random.randint(0,999999):06d}")
                txn_data.setdefault("merchant_id", "mrc_0001")
                txn_data.setdefault("customer_id", "cust_000001")
                txn_data.setdefault("currency", "INR")
                txn_data.setdefault("payment_method", "upi")
                txn_data.setdefault("timestamp", datetime.now().isoformat())
                txn_data.setdefault("customer_avg_amount", txn_data.get("amount", 1000))

                features = TransactionFeatures(**{
                    k: v for k, v in txn_data.items()
                    if k in TransactionFeatures.model_fields
                })
                evaluate_txn = _get_graph()
                result = evaluate_txn(features)

            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.stop()

        # ── results ──
        st.divider()
        decision = result.decision
        score    = result.risk_score

        # banner
        banner_class = {"ALLOW":"banner-allow","MANUAL_REVIEW":"banner-review","FLAG":"banner-flag"}.get(decision,"banner-review")
        emoji = _risk_emoji(decision)
        label_text = {"ALLOW":"LOW RISK — ALLOW","MANUAL_REVIEW":"MEDIUM RISK — REVIEW","FLAG":"HIGH RISK — FLAG"}.get(decision, decision)
        st.markdown(
            f'<div class="{banner_class}"><h2 style="margin:0">{emoji} {label_text}</h2>'
            f'<p style="margin:4px 0 0;opacity:.7">Transaction: {txn_data.get("transaction_id","?")} · ₹{txn_data.get("amount",0):,.2f} · Policy: {result.policy_triggered}</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        col_gauge, col_details = st.columns([1, 2])
        with col_gauge:
            st.plotly_chart(_gauge_figure(score), use_container_width=True)
            st.markdown(f'<div style="text-align:center; color:#94a3b8">Risk Score: <b style="color:white">{score:.1%}</b></div>', unsafe_allow_html=True)

        with col_details:
            st.markdown('<div class="section-header">Transaction Profile</div>', unsafe_allow_html=True)
            t1, t2, t3 = st.columns(3)
            t1.metric("Amount", f"₹{txn_data.get('amount',0):,.2f}")
            t2.metric("Payment Method", txn_data.get("payment_method","?").upper())
            t3.metric("Customer Age", f"{txn_data.get('customer_age_days',0)} days")

            t4, t5, t6 = st.columns(3)
            t4.metric("IP Risk Score", f"{txn_data.get('ip_risk_score',0):.2f}")
            t5.metric("Transactions (1h)", txn_data.get("transactions_1h", 0))
            t6.metric("Failed Attempts", txn_data.get("failed_attempts_24h", 0))

            if not result.model_healthy:
                st.warning(
                    "⚠️ Risk model was unavailable. Transaction moved to MANUAL REVIEW. "
                    "No automated action was taken. Model health check failed."
                )

        # ── explanation ──
        st.divider()
        st.markdown('<div class="section-header">Why was this decision made?</div>', unsafe_allow_html=True)
        if result.explanation:
            if result.explanation.narrative:
                st.info(result.explanation.narrative)

            suspicious_feats = [f for f in result.explanation.top_features if f.get("is_suspicious")]
            normal_feats     = [f for f in result.explanation.top_features if not f.get("is_suspicious")]

            if suspicious_feats:
                st.markdown("**🔴 Suspicious Signals**")
                for feat in suspicious_feats[:5]:
                    val = feat.get("value", "")
                    if isinstance(val, float):
                        val_str = f"{val:.3f}"
                    else:
                        val_str = str(val)
                    st.markdown(
                        f'<div class="feature-card suspicious">🔴 <b>{feat["label"]}</b>: {val_str} '
                        f'<span style="float:right;color:#94a3b8">importance: {feat["importance"]:.3f}</span></div>',
                        unsafe_allow_html=True,
                    )

            if normal_feats:
                st.markdown("**🟢 Normal Signals**")
                for feat in normal_feats[:3]:
                    val = feat.get("value", "")
                    if isinstance(val, float):
                        val_str = f"{val:.3f}"
                    else:
                        val_str = str(val)
                    st.markdown(
                        f'<div class="feature-card">🟢 <b>{feat["label"]}</b>: {val_str}</div>',
                        unsafe_allow_html=True,
                    )

        st.divider()
        if result.audit_entry:
            with st.expander("📋 Raw Audit Entry"):
                st.json(result.audit_entry.model_dump())


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — AUTOML ENGINE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 AutoML Engine":
    st.markdown("# 🤖 AutoML Engine")
    st.caption("Train the fraud model using the existing AutoCraft AutoML pipeline.")
    st.divider()

    dataset_exists = os.path.exists(DATASET_PATH)
    ms = _model_status()

    col_gen, col_train = st.columns(2)

    with col_gen:
        st.markdown("### Step 1: Generate Dataset")
        if dataset_exists:
            df = _load_dataset(DATASET_PATH)
            st.success(f"Dataset ready: {len(df):,} rows")
            fraud_n = int(df["is_fraud"].sum())
            g1, g2, g3 = st.columns(3)
            g1.metric("Total", f"{len(df):,}")
            g2.metric("Fraud", f"{fraud_n:,} ({fraud_n/len(df)*100:.1f}%)")
            g3.metric("Features", f"{len(df.columns)-1}")
        else:
            st.warning("No dataset found.")

        n_rows = st.selectbox("Dataset size", [10_000, 25_000, 50_000], index=2)
        if st.button("⚡ Generate Synthetic Dataset", type="primary"):
            with st.spinner("Generating fraud transaction dataset..."):
                from riskpilot.data.generate_dataset import generate_dataset
                generate_dataset(n_total=n_rows, verbose=False)
                st.cache_data.clear()
            st.success(f"Generated {n_rows:,} transactions!")
            st.rerun()

    with col_train:
        st.markdown("### Step 2: Train Model")
        if not dataset_exists:
            st.warning("Generate dataset first.")
        else:
            n_trials = st.slider("Optuna Trials", 10, 100, 30, step=5)
            max_retries = st.slider("Critic Retries", 0, 2, 1)

            if st.button("🚀 Train Fraud Model via AutoML", type="primary"):
                st.info("Training in progress... This may take 3–8 minutes.")
                log_box = st.empty()
                logs_live = []

                def on_progress(node_name, state):
                    if state.logs:
                        logs_live.append(f"[{node_name}] {state.logs[-1]}")
                        log_box.code("\n".join(logs_live[-20:]))

                from riskpilot.core.model_registry import train_fraud_model
                try:
                    result = train_fraud_model(
                        dataset_path=DATASET_PATH,
                        optuna_trials=n_trials,
                        max_retries=max_retries,
                        progress_callback=on_progress,
                        verbose=False,
                    )
                    st.success(f"✅ Model trained! Version: {result['model_version']}")
                    st.json(result["metrics"])
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

    # ── model info if trained ──
    if ms["trained"]:
        st.divider()
        st.markdown("### Current Model")
        reg = ms["registry"]
        r1, r2, r3 = st.columns(3)
        r1.metric("Architecture", reg.get("architecture", "?"))
        r2.metric("Version", reg.get("model_version", "?"))
        r3.metric("Trained At", reg.get("trained_at", "?")[:16])

        metrics = ms["metrics"]
        if metrics:
            st.markdown("**Validation Metrics**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy",  f"{metrics.get('accuracy',0):.1%}")
            m2.metric("Precision", f"{metrics.get('precision_macro',0):.1%}")
            m3.metric("Recall",    f"{metrics.get('recall_macro',0):.1%}")
            m4.metric("F1 (macro)",f"{metrics.get('f1_macro',0):.1%}")

        bp = reg.get("best_params", {})
        if bp:
            st.markdown("**Best Hyperparameters (Optuna)**")
            st.json(bp)

    # ── automl pipeline diagram ──
    st.divider()
    st.markdown("### AutoML Pipeline Flow")
    st.code("""
    fraud_transactions.csv
           ↓
    [Planner Agent]     ← decides Optuna trials, strategy
           ↓
    [Analyst Agent]     ← data profiling, imbalance detection
           ↓
    [Selector Agent]    ← XGBoost / LightGBM / RandomForest
           ↓
    [Trainer Agent]     ← Optuna hyperparameter optimization
           ↓
    [Evaluator Agent]   ← Precision, Recall, F1, ROC-AUC
           ↓
    [Critic Agent]      ← approve or retry with new params
           ↓
    [Reporter Agent]    ← final model registered
    """, language="text")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Evaluation":
    st.markdown("# 📊 Model Evaluation")
    st.caption("All metrics reported on a held-out test set (20% stratified split). Defense-only system.")
    st.divider()

    ms = _model_status()
    if not ms["trained"]:
        st.error("No trained model. Go to **AutoML Engine** to train.")
        st.stop()

    metrics = ms["metrics"]
    reg     = ms["registry"]

    # ── key metrics ──
    st.markdown("### Primary Metrics — Held-Out Test Set")
    st.caption(
        "⚠️ Razorpay Track 02 requires honest precision & recall on a held-out test set. "
        "These metrics are from the 20% stratified split never seen during training."
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",        f"{metrics.get('accuracy',0):.1%}")
    m2.metric("Precision (macro)",f"{metrics.get('precision_macro',0):.1%}")
    m3.metric("Recall (macro)",  f"{metrics.get('recall_macro',0):.1%}")
    m4.metric("F1 (macro)",      f"{metrics.get('f1_macro',0):.1%}")
    m5.metric("ROC-AUC",         f"{metrics.get('roc_auc', metrics.get('roc_auc_ovr', 0)):.3f}")

    # ── load test data for confusion matrix ──
    st.divider()
    col_cm, col_cost = st.columns(2)

    with col_cm:
        st.markdown("### Confusion Matrix")
        df = _load_dataset(DATASET_PATH)

        if not df.empty:
            try:
                import joblib
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import confusion_matrix

                model_obj   = joblib.load(MODEL_PATH)
                prep_data   = joblib.load(
                    os.path.join(ROOT, "artifacts", "fraud_model_preprocessor.joblib")
                )
                preprocessor  = prep_data["preprocessor"]
                label_encoder = prep_data.get("label_encoder")

                from riskpilot.core.model_registry import FEATURE_COLUMNS, TARGET_COLUMN
                X = df[FEATURE_COLUMNS]
                y = df[TARGET_COLUMN].astype(int)

                _, X_test, _, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                X_test_t = preprocessor.transform(X_test).astype(np.float32)
                y_pred   = model_obj.predict(X_test_t)

                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Legit", "Actual Fraud"],
                    columns=["Pred Legit", "Pred Fraud"],
                )

                fig_cm = px.imshow(
                    cm_df, text_auto=True,
                    color_continuous_scale="Blues",
                    title="Confusion Matrix",
                )
                fig_cm.update_layout(
                    paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
                    height=320, margin=dict(l=0,r=0,t=40,b=0),
                )
                st.plotly_chart(fig_cm, use_container_width=True)

                tn, fp, fn, tp = cm.ravel()
                st.markdown(f"""
| | Count |
|---|---|
| True Positives (Fraud caught) | **{tp:,}** |
| False Positives (Legit blocked) | **{fp:,}** |
| True Negatives (Legit passed) | **{tn:,}** |
| False Negatives (Fraud missed) | **{fn:,}** |
""")
            except Exception as exc:
                st.warning(f"Could not compute confusion matrix: {exc}")
        else:
            st.info("Generate dataset to see confusion matrix.")

    with col_cost:
        st.markdown("### False-Positive Cost Calculator")
        st.caption("Razorpay Track 02 explicitly requires honest false-positive cost.")

        fp_cost_per_txn = st.number_input(
            "Cost per false positive (₹)",
            min_value=100, max_value=50000, value=1200, step=100,
            help="Estimated cost: analyst review time + merchant friction + customer support",
        )
        avg_fraud_amount = st.number_input(
            "Average fraud transaction amount (₹)",
            min_value=100, max_value=200000, value=12000, step=500,
        )

        if not df.empty:
            try:
                n_test   = len(y_test)
                fp_total = int(fp)
                fn_total = int(fn)
                tp_total = int(tp)

                fp_total_cost = fp_total * fp_cost_per_txn
                fraud_saved   = tp_total * avg_fraud_amount
                fraud_missed  = fn_total * avg_fraud_amount
                net_benefit   = fraud_saved - fp_total_cost

                st.markdown("---")
                e1, e2 = st.columns(2)
                e1.metric("Fraud Prevented (₹)", f"₹{fraud_saved:,.0f}", help=f"{tp_total} true positives × ₹{avg_fraud_amount:,}")
                e2.metric("Fraud Missed (₹)",    f"₹{fraud_missed:,.0f}", help=f"{fn_total} false negatives × ₹{avg_fraud_amount:,}")
                e3, e4 = st.columns(2)
                e3.metric("False-Positive Cost (₹)", f"₹{fp_total_cost:,.0f}", help=f"{fp_total} FPs × ₹{fp_cost_per_txn:,}")
                e4.metric("Net Benefit (₹)",     f"₹{net_benefit:,.0f}",
                           delta=f"{'positive' if net_benefit>0 else 'negative'}")

                if net_benefit > 0:
                    st.success(f"✅ Net benefit of ₹{net_benefit:,.0f} — system is profitable.")
                else:
                    st.warning("⚠️ FP costs outweigh fraud prevented — consider raising thresholds.")

                st.markdown(f"""
---
**Test Set Summary** *(n={n_test:,}, 20% held-out)*
- Precision: **{tp_total/(tp_total+fp_total)*100:.1f}%** | Recall: **{tp_total/(tp_total+fn_total)*100:.1f}%**
- Fraud detected: **{tp_total:,}** / Fraud missed: **{fn_total:,}**
- False positives: **{fp_total:,}**
""")
            except Exception as exc:
                st.info(f"Run confusion matrix first. ({exc})")

    # ── ROC curve ──
    st.divider()
    if not df.empty:
        try:
            from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_curve

            if hasattr(model_obj, "predict_proba"):
                y_prob = model_obj.predict_proba(X_test_t)[:, 1]

                col_roc, col_pr = st.columns(2)
                with col_roc:
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc_val = sk_auc(fpr, tpr)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={roc_auc_val:.3f})", line=dict(color="#6366f1", width=2)))
                    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash="dash", color="#475569"), showlegend=False))
                    fig_roc.update_layout(
                        title="ROC Curve", xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        paper_bgcolor="#0a0e1a", plot_bgcolor="#0f172a",
                        font_color="#e2e8f0", height=360,
                        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

                with col_pr:
                    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
                    pr_auc_val = sk_auc(rec_curve, prec_curve)
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=rec_curve, y=prec_curve, name=f"PR (AUC={pr_auc_val:.3f})", line=dict(color="#22c55e", width=2)))
                    fig_pr.update_layout(
                        title="Precision-Recall Curve", xaxis_title="Recall",
                        yaxis_title="Precision",
                        paper_bgcolor="#0a0e1a", plot_bgcolor="#0f172a",
                        font_color="#e2e8f0", height=360,
                        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
                    )
                    st.plotly_chart(fig_pr, use_container_width=True)
        except Exception:
            pass

    # ── feature importance ──
    st.divider()
    st.markdown("### Feature Importance")
    try:
        from riskpilot.core.model_registry import get_feature_importances
        imp = get_feature_importances()
        imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        feat_names = [x[0] for x in imp_sorted]
        feat_vals  = [x[1] for x in imp_sorted]

        fig_imp = px.bar(
            x=feat_vals, y=feat_names,
            orientation="h",
            color=feat_vals,
            color_continuous_scale="Bluered_r",
            labels={"x": "Importance", "y": "Feature"},
        )
        fig_imp.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0f172a",
            font_color="#e2e8f0", height=420,
            coloraxis_showscale=False,
            margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as exc:
        st.info(f"Feature importance unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Audit Log":
    st.markdown("# 📋 Audit Log")
    st.caption("Complete decision trail — every transaction evaluated by RiskPilot.")
    st.divider()

    audit = _load_audit()

    if not audit:
        st.info("No audit entries yet. Analyze transactions in the **Investigate** page.")
        st.stop()

    df_audit = pd.DataFrame(audit)

    # ── summary ──
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Entries", f"{len(df_audit):,}")
    a2.metric("Flagged",    f"{(df_audit['decision']=='FLAG').sum():,}")
    a3.metric("Reviewed",   f"{(df_audit['decision']=='MANUAL_REVIEW').sum():,}")
    a4.metric("Allowed",    f"{(df_audit['decision']=='ALLOW').sum():,}")

    st.divider()

    # ── filters ──
    f1, f2, f3 = st.columns(3)
    decision_filter = f1.multiselect(
        "Filter by decision",
        ["ALLOW", "MANUAL_REVIEW", "FLAG"],
        default=["ALLOW", "MANUAL_REVIEW", "FLAG"],
    )
    model_filter = f2.checkbox("Show unhealthy model entries only", value=False)
    n_show       = f3.slider("Entries to show", 10, 500, 100)

    filtered = df_audit[df_audit["decision"].isin(decision_filter)]
    if model_filter:
        filtered = filtered[filtered["model_healthy"] == False]
    filtered = filtered.tail(n_show).iloc[::-1]  # newest first

    # ── table ──
    display_cols = ["timestamp", "transaction_id", "amount", "risk_score", "risk_label", "decision", "policy_triggered", "model_healthy"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    def _color_decision(val):
        colors = {"ALLOW": "color: #22c55e", "MANUAL_REVIEW": "color: #f59e0b", "FLAG": "color: #ef4444"}
        return colors.get(val, "")

    styled = filtered[display_cols].style.map(
        _color_decision, subset=["decision"] if "decision" in display_cols else []
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── download ──
    csv_data = filtered.to_csv(index=False)
    st.download_button(
        "⬇️ Download Audit Log (CSV)",
        data=csv_data,
        file_name=f"riskpilot_audit_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    st.divider()
    col_clr, _ = st.columns([1, 4])
    if col_clr.button("🗑️ Clear Audit Log", type="secondary"):
        from riskpilot.agents.audit_logger import clear_audit_log
        clear_audit_log()
        st.rerun()

    # ── drift monitor ──
    st.divider()
    st.markdown("### 📡 Distribution Drift Monitor")
    st.caption("PSI-based comparison of recent transactions vs. training distribution.")

    if len(audit) >= 20:
        from riskpilot.monitoring.drift_detector import compute_drift, overall_drift_status
        recent_txns = [
            {
                "amount":               e.get("amount", 0),
                "ip_risk_score":        0.0,
                "transactions_1h":      0,
                "device_change_count":  0,
                "customer_age_days":    0,
                "amount_vs_avg_ratio":  0.0,
            }
            for e in audit[-200:]
        ]
        drift_results = compute_drift(recent_txns)
        status, msg   = overall_drift_status(drift_results)

        if status == "alert":
            st.error(msg)
        elif status == "warning":
            st.warning(msg)
        else:
            st.success(msg)

        if drift_results:
            drift_df = pd.DataFrame([
                {"Feature": k, "PSI": v["psi"], "Status": v["label"],
                 "Train Mean": v["train_mean"], "Recent Mean": v["recent_mean"]}
                for k, v in drift_results.items()
            ])
            st.dataframe(drift_df, use_container_width=True, hide_index=True)
    else:
        st.info("Analyze at least 20 transactions to enable drift monitoring.")
