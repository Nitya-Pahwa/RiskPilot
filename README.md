# RiskPilot — AI Merchant Risk Agent

> **Razorpay Buildathon 2026 · Track 02**

RiskPilot is an AI-powered real-time fraud detection system for the Razorpay payments ecosystem. It scores transactions using a trained ML model, applies a deterministic policy engine, generates plain-English explanations for every decision, and logs a full audit trail — all through a LangGraph multi-agent pipeline.

---

## What It Solves

Traditional rule-based fraud systems are either too rigid (fraudsters learn to bypass fixed rules) or too opaque (black-box models that can't explain their decisions). RiskPilot combines:

- **Adaptive ML scoring** — a model trained on 17 behavioral features (velocity, IP risk, device changes, location anomaly, chargeback history, etc.)
- **Deterministic policy enforcement** — transparent, auditable rules that convert scores into actions
- **LLM-powered explainability** — every flag includes a plain-English reason grounded in feature signals, never invented
- **Full audit trail** — every transaction decision is logged with model version, score, policy triggered, and top risk factors
- **Drift monitoring** — Population Stability Index tracks when the live transaction distribution shifts away from training data

---

## Architecture

```
Transaction Input
       │
       ▼
[1] Transaction Analyzer     — validates & enriches features (velocity, device flags, IP risk)
       │
       ▼
[2] Risk Model Agent         — ML fraud classifier → risk score 0.0 – 1.0
       │
       ▼
[3] Policy Engine            — score < 0.30 → ALLOW | 0.30–0.70 → MANUAL REVIEW | ≥ 0.70 → FLAG
       │
       ▼
[4] Explanation Agent        — Groq LLaMA 3.3 → plain-English reason (feature-grounded only)
       │
       ▼
[5] Audit Logger             — structured AuditEntry → compliance log
       │
       ▼
     Output (risk_score, decision, explanation, audit_entry)
```

The pipeline is built with **LangGraph**. If the model is unavailable, the system automatically routes to `MANUAL_REVIEW` — no silent failures, no false approvals.

---

## Project Structure

```
.
├── run_riskpilot.py              # Main entry point (generate data, train, demo)
├── run_pipeline.py               # AutoML pipeline CLI
├── requirements.txt
├── README.md
├── riskpilot/
│   ├── agents/
│   │   ├── risk_agent.py         # Loads model, scores transaction
│   │   ├── policy_engine.py      # Deterministic risk policy rules
│   │   ├── explanation_agent.py  # LLM narrative from feature signals
│   │   └── audit_logger.py       # Writes structured audit entry
│   ├── core/
│   │   ├── fraud_graph.py        # LangGraph pipeline definition
│   │   ├── fraud_state.py        # Pydantic state models
│   │   └── model_registry.py     # Train, save, load, version the model
│   ├── data/
│   │   └── generate_dataset.py   # Synthetic fraud dataset generator
│   ├── monitoring/
│   │   └── drift_detector.py     # PSI-based distribution drift monitor
│   └── ui/
│       └── dashboard.py          # Streamlit dashboard
└── automl_agent/                 # AutoML backbone (model training pipeline)
    ├── agents/                   # analyst, planner, trainer, evaluator, critic, reporter
    ├── core/                     # LangGraph graph, LLM config, state models
    └── utils/                    # preprocessing, data loader, MLflow tracker
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph |
| ML Model Training | scikit-learn, XGBoost, LightGBM, Optuna |
| Hyperparameter Tuning | Optuna |
| LLM Explanations | Groq (LLaMA 3.3 70B) via LangChain |
| Dashboard | Streamlit, Plotly |
| Data | pandas, NumPy |
| Model Serialization | Joblib |
| Experiment Tracking | MLflow |
| State Validation | Pydantic |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Nitya-Pahwa/RiskPilot.git
cd RiskPilot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows:
```bash
venv\Scripts\activate
```

On macOS / Linux:
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the LLM (optional)

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

The system works without an API key using rule-based fallbacks for explanations.

---

## Usage

### Full Setup (generate data + train model)

```bash
python run_riskpilot.py
```

### Generate dataset only

```bash
python run_riskpilot.py --generate-only
```

### Train only (dataset must exist)

```bash
python run_riskpilot.py --train-only
```

### Run demo evaluation on sample transactions

```bash
python run_riskpilot.py --demo
```

### Launch the Streamlit dashboard

```bash
streamlit run riskpilot/ui/dashboard.py
```

---

## Dashboard Features

- **Transaction Evaluator** — submit any transaction and get an instant risk score, decision, and LLM explanation
- **Audit Log** — full searchable log of all evaluated transactions with model version and risk reasons
- **Model Registry** — view trained model version, architecture, training date, and performance metrics
- **Drift Monitor** — real-time PSI chart across 6 key features with Stable / Warning / Alert status

---

## Policy Rules

| Condition | Decision |
|-----------|---------|
| Risk score < 0.30 | ✅ ALLOW |
| Risk score 0.30 – 0.70 | ⚠️ MANUAL REVIEW |
| Risk score ≥ 0.70 | 🚨 FLAG |
| FLAG + amount > ₹50,000 | Requires explicit human approval |
| Model unavailable | MANUAL REVIEW always |

All decisions are **defense-only** — no funds are ever automatically blocked without a human in the loop.

---

## Drift Monitoring

The drift monitor uses **Population Stability Index (PSI)** on 6 key features:

| PSI Range | Status |
|-----------|--------|
| < 0.10 | ✅ Stable |
| 0.10 – 0.20 | ⚠️ Moderate drift — monitor closely |
| > 0.20 | 🚨 Significant drift — retraining recommended |

---

## Notes

- The `data/` directory (synthetic dataset) and `artifacts/` directory (trained models) are excluded from this repo — run `python run_riskpilot.py` to regenerate both locally.
- Never commit your `.env` file — it contains your API key.
- The full pipeline works without a Groq API key; explanations fall back to rule-based narratives.

---

## ScreenShots

<img width="1920" height="885" alt="Screenshot (2835)" src="https://github.com/user-attachments/assets/834cff06-82f2-48fa-ab09-f97af749bd4d" />


<img width="1920" height="897" alt="Screenshot (2846)" src="https://github.com/user-attachments/assets/413ade7d-e0b9-4916-95b6-2ba8032642a1" />
<img width="1920" height="878" alt="Screenshot (2836)" src="https://github.com/user-attachments/assets/fdc34a36-c573-4cd8-af64-88c62202f629" />





<img width="1920" height="899" alt="Screenshot (2837)" src="https://github.com/user-attachments/assets/3d587d16-c56a-436c-8aa3-7ce3c364536b" />

<img width="1920" height="883" alt="Screenshot (2838)" src="https://github.com/user-attachments/assets/a335aa69-e771-4537-b2ef-5a7a2f593a8e" />
<img width="1920" height="880" alt="Screenshot (2839)" src="https://github.com/user-attachments/assets/197aee17-ed01-4d72-8a28-ef1856ba43eb" />


<img width="1920" height="906" alt="Screenshot (2840)" src="https://github.com/user-attachments/assets/f6f0d660-40da-498e-8eac-f310f8bd4f2f" />


<img width="1920" height="856" alt="Screenshot (2841)" src="https://github.com/user-attachments/assets/1f6376cb-d8c8-4c7b-94da-428ddce0ae3f" />

<img width="1920" height="831" alt="Screenshot (2842)" src="https://github.com/user-attachments/assets/d17e913a-acf1-44ef-a103-ce17cd0eec9a" />



<img width="1920" height="904" alt="Screenshot (2843)" src="https://github.com/user-attachments/assets/ad6b52f7-3470-4afb-95a5-e16bada4c343" />


<img width="1920" height="902" alt="Screenshot (2844)" src="https://github.com/user-attachments/assets/f832c530-7b50-4fc0-9608-1506577c9e62" />



<img width="1920" height="867" alt="Screenshot (2845)" src="https://github.com/user-attachments/assets/ecb9d59f-0696-4809-b22f-9abb80c57867" />


