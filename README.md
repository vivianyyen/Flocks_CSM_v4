# 🛡️ Flocks CSM v5 — Cyber Threat Intelligence Dashboard

> Real-time cybersecurity incident monitoring, LSTM forecasting, and AHP-based decision analysis.
> Built with **Streamlit** + **Supabase** + **Groq AI**.

---

## Overview

**Flocks CSM v5** extends the threat intelligence dashboard with three integrated AI/analytical layers aligned to Malaysia's NAIO Action Plan 2026–2030:

| Layer | Method | Purpose |
|---|---|---|
| 1 | **LSTM Neural Network** | Predict future incident volumes (next 7 days) |
| 2 | **Quantitative Risk Analysis** | Likelihood × Impact matrix + Monte Carlo simulation |
| 3 | **AHP Decision Analysis** | Prioritise incidents for analyst response |

---

## Project Structure

```
Flocks_CSM_v5/
├── application.py                  # Main entry point + navigation
├── requirements.txt
├── .streamlit/
│   └── secrets.toml                # Credentials (never commit!)
└── utils/
    ├── supabase_client.py          # Supabase connection
    ├── charts.py                   # Plotly & WordCloud charts
    ├── chatbot.py                  # Groq AI analyst chatbot
    ├── risk_scorer.py              # Rule-based risk scoring (baseline)
    ├── lstm_forecaster.py          # ⭐ LSTM incident volume forecasting
    ├── risk_analysis.py            # ⭐ Risk matrix + Monte Carlo
    ├── decision_analysis.py        # ⭐ AHP prioritisation (no TOPSIS)
    └── page_ai_risk_decision.py    # ⭐ AI Risk Decision Centre page
```

---

## Pages

| Page | Description |
|---|---|
| 📰 Cyber News | Global + Malaysia incident feed, charts, filters |
| 🔴 Ransomware Live | Ransomware victim tracker |
| 🤖 AI Analyst | Groq-powered natural language chatbot |
| 🧠 AI Risk Decision | LSTM forecast + Risk Analysis + AHP ← NEW |

---

## AI Risk Decision Centre — Feature Details

### Layer 1 — LSTM Forecasting
- Predicts **daily incident count for the next 7 days**
- Architecture: `LSTM(64) → Dense(32, ReLU) → Dense(7)`
- Trained on historical Supabase incident data
- Uncertainty bands grow with forecast horizon (±10% per day)
- Adjustable lookback window (7–30 days) and forecast horizon (3–14 days)

### Layer 2 — Quantitative Risk Analysis
- **Likelihood × Impact** risk matrix (ISO 31000)
- **Monte Carlo simulation** (N=1,000): P5/P50/P95 confidence intervals
- **Sector Composite Risk Index**: mean risk × log(1 + count)
- **7-day rolling risk trend** chart

### Layer 3 — AHP Decision Analysis
- **5×5 pairwise comparison matrix** (Saaty 1–9 scale)
- Eigenvector-derived priority weights
- **Consistency Ratio check** (CR < 0.10 required)
- Colour-coded **response queue** (Critical / High / Medium / Monitor)

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure credentials
```toml
# .streamlit/secrets.toml
[supabase]
url = "https://xxxxxxxxxxxx.supabase.co"
key = "your-anon-public-key"

[groq]
api_key = "your-groq-api-key"
```

### 3. Run
```bash
streamlit run application.py
```

---

## Deployment (Streamlit Community Cloud — Free)

1. Push to a **public** GitHub repo (ensure `secrets.toml` is in `.gitignore`)
2. Go to https://share.streamlit.io → **New app**
3. Set `application.py` as the main file
4. Add secrets under **Advanced settings → Secrets**
5. Click **Deploy**

---

## NAIO Alignment

| NAIO Area | How This Project Addresses It |
|---|---|
| Area 3 — AI Adaptation | LSTM + AHP demonstrate practical AI in cybersecurity ops |
| Area 4 — AI Ethics | Transparent, explainable AHP weights + CR validation |
| Area 5 — AI Impact Study | Forecast + prioritisation quantify AI's operational value |
| Area 7 — Datasets | Live Supabase pipeline drives all analytical layers |
