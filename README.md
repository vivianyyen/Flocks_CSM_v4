# 🛡️ Incident Intel — Cybersecurity & Threat Intelligence Dashboard

> Real-time cybersecurity incident monitoring and ransomware tracking dashboard built with **Streamlit** + **Supabase** + **Groq AI (Flocks AI)**.

---

## Overview

**Incident Intel** is a full-stack threat intelligence dashboard that aggregates cybersecurity news and ransomware victim data into a single, interactive interface. It enables security analysts and researchers to monitor global cyber threats in real time, explore trends across categories and countries, and ask natural language questions via an AI-powered analyst chatbot backed by Groq's fast LLM inference.

---

## Project Structure

```
incident_dashboard/
├── application.py                # Main dashboard entry point
├── requirements.txt
├── .streamlit/
│   └── secrets.toml              # Your credentials (never commit this!)
└── utils/
    ├── supabase_client.py        # Supabase connection & data fetching
    ├── charts.py                 # All Plotly & WordCloud chart functions
    ├── chatbot.py                # Groq-powered AI analyst chatbot
    ├── risk_scorer.py            # Rule-based risk scoring engine (baseline)
    ├── dl_classifier.py          # ⭐ Deep Learning: TF-IDF → MLP classifier
    ├── risk_analysis.py          # ⭐ Quantitative Risk: Likelihood×Impact + Monte Carlo
    ├── decision_analysis.py      # ⭐ MCDA: AHP + TOPSIS prioritisation
    └── page_ai_risk_decision.py  # ⭐ AI Risk Decision Centre page
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure credentials
```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.template .streamlit/secrets.toml
# Then edit .streamlit/secrets.toml with your real keys
```

Your `.streamlit/secrets.toml` should look like:
```toml
[supabase]
url = "https://xxxxxxxxxxxx.supabase.co"
key = "your-anon-public-key"        # Found in Supabase → Settings → API

[groq]
api_key = "your-groq-api-key"       # From console.groq.com
```

### 3. Supabase table schema

**Tab 1 — Incidents table** (default name: `incidents`):

| Column | Type |
|---|---|
| id | int8 / uuid |
| title | text |
| publication_date | timestamptz |
| source | text |
| url | text |
| summary | text |
| relevant_keywords | text |
| category | text |
| country | text |
| impact | text |
| incident_type | text |
| entity_affected | text |
| incident_date | timestamptz |

**Tab 2 — Ransomware victims table** (default name: `ransomware_victims`):

| Column | Type |
|---|---|
| id | int8 / uuid |
| post_title | text |
| group_name | text |
| country | text |
| activity | text |
| discovered | timestamptz |
| published | timestamptz |
| website | text |
| post_url | text (unique) |
| description | text |
| created_at | timestamptz |

> If your tables have different names, update `get_data("incidents")` and `get_data("ransomware_victims")` in `application.py`.

### 4. Run the dashboard
```bash
streamlit run application.py
```

Open http://localhost:8501 in your browser.

---

## Features

### 📰 Tab 1 — Cyber News
| Feature | Description |
|---|---|
| **KPI Cards** | Total incidents, sources, critical count, countries, new this week |
| **Category Bar Chart** | Horizontal bar — incidents per category |
| **Incident Type Donut** | Pie/donut breakdown by incident type |
| **Timeline Area Chart** | Weekly trend, stacked by category |
| **Impact Distribution** | Critical → High → Medium → Low |
| **Choropleth Map** | World map coloured by incident count per country |
| **Source Breakdown** | Top crawled domains |
| **Word Clouds** | Generated from summary text & relevant keywords |
| **Raw Data Table** | Filterable, sortable incident table |
| **AI Chatbot** | Groq-powered analyst; answers questions about the loaded data |
| **Sidebar Filters** | Date range, category, country, impact |
| **Auto-refresh** | Toggle 60-second live refresh |

### 🧠 NEW: AI Risk Decision Centre (Page 4)
| Feature | Description |
|---|---|
| **DL Classifier** | TF-IDF → MLP neural network (128→64→4), trained on live data |
| **Confidence Histogram** | Per-severity confidence distribution from the DL model |
| **ISO 31000 Risk Matrix** | Likelihood × Impact scatter with quadrant zones |
| **Monte Carlo Simulation** | N=1,000 runs → P5/P50/P95 uncertainty bands per incident |
| **Sector Risk Index** | Composite risk aggregated by industry sector |
| **AHP Weights Chart** | Eigenvector-derived criteria weights with Consistency Ratio |
| **TOPSIS Ranking** | Incidents ranked by distance from ideal best/worst |
| **Priority Queue** | Top 20 incidents with colour-coded response recommendations |
| **AHP vs TOPSIS Plot** | Cross-method validation scatter |


| Feature | Description |
|---|---|
| **KPI Cards** | Total victims, active groups, most active group, countries hit, new this week |
| **Weekly Victims Timeline** | Bar chart of victim count over time |
| **Top 10 Groups** | Ranked bar chart of most active ransomware groups |
| **World Map** | Choropleth map of victims by country |
| **Sector Breakdown** | Pie chart of industries targeted |
| **Victim Feed** | Scrollable cards with group, country, sector, date, and source link |
| **Sidebar Filters** | Date range, threat group, country, sector |

---

## Free Deployment on Streamlit Community Cloud

1. Push your project to a **public** GitHub repo
   (make sure `.streamlit/secrets.toml` is in `.gitignore`)
2. Go to https://share.streamlit.io → **New app**
3. Select your repo and set `application.py` as the main file
4. Add your secrets under **Advanced settings → Secrets**
5. Click **Deploy** — free, no credit card needed

---

## Cost Summary (All Free Tiers)

| Service | Free Tier |
|---|---|
| Supabase | 500 MB DB, 2 GB bandwidth / month |
| Streamlit Community Cloud | Unlimited public apps |
| Groq API | Free tier available at console.groq.com |
