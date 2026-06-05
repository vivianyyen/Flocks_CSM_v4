"""
pages/page_ai_risk_decision.py
────────────────────────────────────────────────────────────────────────────────
AI Risk Decision Centre
────────────────────────
Integrates three analytical layers:

  Layer 1 — Deep Learning (DL) Classifier
    TF-IDF → MLP neural network trained on live Supabase incident data.
    Predicts severity (Critical/High/Medium/Low) + confidence score.

  Layer 2 — Quantitative Risk Analysis
    Likelihood × Impact matrix, Monte Carlo uncertainty simulation (N=1000),
    sector-level composite risk index, and risk trend over time.

  Layer 3 — Multi-Criteria Decision Analysis (MCDA)
    AHP (Analytic Hierarchy Process) + TOPSIS to rank incidents and generate
    prioritised response recommendations for security analysts.

Aligned to Malaysia NAIO Action Plan 2026–2030:
  • Area 3: Acceleration of AI Technology Adaptation
  • Area 5: AI Impact Study for Government
  • Area 4: AI Code of Ethics (explainability, transparency)
────────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Internal modules ──────────────────────────────────────────────────────────
from utils.dl_classifier  import DLClassifier
from utils.risk_analysis  import (
    enrich_with_risk,
    compute_sector_risk_index,
    compute_risk_trend,
    risk_matrix_quadrant,
    monte_carlo_one,
)
from utils.decision_analysis import (
    prioritise,
    ahp_weights,
    consistency_ratio,
    weight_table,
    run_ahp,
    run_topsis,
)

# ── Colour palette ────────────────────────────────────────────────────────────
COLOURS = {
    "Critical": "#FF4B4B",
    "High":     "#FF8C42",
    "Medium":   "#FFD166",
    "Low":      "#06D6A0",
    "Extreme":  "#FF4B4B",
    "Moderate": "#FFD166",
    "bg":       "#0E1117",
    "card":     "#1A1D27",
    "accent":   "#7C3AED",
}


# ═════════════════════════════════════════════════════════════════════════════
#  HELPER WIDGETS
# ═════════════════════════════════════════════════════════════════════════════

def _metric_card(title: str, value: str, delta: str = "", colour: str = "#7C3AED"):
    st.markdown(
        f"""
        <div style="background:{COLOURS['card']};border-left:4px solid {colour};
                    padding:16px 20px;border-radius:8px;margin-bottom:4px;">
            <div style="font-size:11px;color:#9CA3AF;text-transform:uppercase;
                        letter-spacing:.08em;">{title}</div>
            <div style="font-size:26px;font-weight:700;color:#F9FAFB;
                        line-height:1.2;">{value}</div>
            <div style="font-size:12px;color:#6B7280;">{delta}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _section(title: str, icon: str = ""):
    st.markdown(
        f"<h3 style='color:#E5E7EB;margin-top:2rem;'>{icon} {title}</h3>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — DEEP LEARNING PANEL
# ═════════════════════════════════════════════════════════════════════════════

def _render_dl_panel(clf: DLClassifier, df_enriched: pd.DataFrame):
    _section("Deep Learning Severity Classifier", "🧠")

    card = clf.model_card()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("Architecture", "MLP Neural Net", "TF-IDF → 128 → 64 → 4",
                     COLOURS["accent"])
    with c2:
        _metric_card("Training Rows", str(card["training_rows"]),
                     "Silver labels from rule-based scorer", "#06D6A0")
    with c3:
        acc = card["val_accuracy"]
        _metric_card("Validation Accuracy",
                     f"{acc*100:.1f}%" if isinstance(acc, float) else str(acc),
                     "20% held-out validation set", "#FF8C42")
    with c4:
        _metric_card("Vocab Size", f"{card['vocab_size']:,}",
                     "Unigrams + bigrams", "#FFD166")

    # Confidence distribution
    if "dl_confidence" in df_enriched.columns and "dl_label" in df_enriched.columns:
        col_a, col_b = st.columns(2)

        with col_a:
            fig = px.histogram(
                df_enriched, x="dl_confidence", color="dl_label",
                nbins=25, barmode="overlay",
                color_discrete_map=COLOURS,
                title="Prediction Confidence Distribution",
                labels={"dl_confidence": "Confidence", "dl_label": "Severity"},
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="", margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            label_counts = df_enriched["dl_label"].value_counts().reset_index()
            label_counts.columns = ["Severity", "Count"]
            fig2 = px.pie(
                label_counts, names="Severity", values="Count",
                color="Severity", color_discrete_map=COLOURS,
                title="DL Severity Distribution",
                hole=0.45, template="plotly_dark",
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Model card expander
    with st.expander("🔍 Model Architecture Details"):
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Architecture | `TF-IDF  →  MLP (128, 64)` |
        | Activation | `ReLU` |
        | Optimiser | `Adam` |
        | Max Iterations | `{card['max_iter']}` |
        | Early Stopping | `Yes (patience=20)` |
        | Vocabulary Size | `{card['vocab_size']:,} features` |
        | Training Samples | `{card['training_rows']}` |
        | Validation Accuracy | `{card['val_accuracy']}` |
        | Status | `{card['status']}` |

        **Note:** This is a genuine Multi-Layer Perceptron (neural network) using
        backpropagation and stochastic gradient descent. It learns to generalise
        the rule-based risk labels from raw incident text, making predictions on
        unseen threats without manual keyword tuning.
        """)


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — RISK ANALYSIS PANEL
# ═════════════════════════════════════════════════════════════════════════════

def _render_risk_panel(df_enriched: pd.DataFrame):
    _section("Quantitative Risk Analysis", "📊")

    # KPIs
    mean_risk  = df_enriched["risk_index"].mean()
    max_risk   = df_enriched["risk_index"].max()
    extreme_n  = (df_enriched["risk_quadrant"] == "Extreme").sum()
    mc_spread  = (df_enriched["risk_mc_p95"] - df_enriched["risk_mc_p5"]).mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("Mean Risk Index", f"{mean_risk:.3f}",
                     "Likelihood × Impact", COLOURS["accent"])
    with c2:
        _metric_card("Peak Risk", f"{max_risk:.3f}",
                     "Highest single incident", COLOURS["Critical"])
    with c3:
        _metric_card("Extreme Quadrant", str(int(extreme_n)),
                     "ISO 31000 — Extreme zone", COLOURS["High"])
    with c4:
        _metric_card("MC Uncertainty Band", f"±{mc_spread:.3f}",
                     "Avg P5–P95 spread (N=1000)", COLOURS["Medium"])

    col_left, col_right = st.columns(2)

    # ── Risk Matrix scatter ───────────────────────────────────────────────────
    with col_left:
        _section("Risk Matrix (Likelihood × Impact)", "🎯")
        fig = px.scatter(
            df_enriched.head(300),
            x="likelihood", y="impact_score",
            color="risk_quadrant",
            size="risk_index",
            size_max=18,
            hover_data=["dl_label", "dl_confidence", "risk_index"],
            color_discrete_map={
                "Extreme": COLOURS["Critical"],
                "High":    COLOURS["High"],
                "Moderate": COLOURS["Medium"],
                "Low":     COLOURS["Low"],
            },
            title="ISO 31000 Risk Matrix",
            labels={
                "likelihood":   "Likelihood →",
                "impact_score": "Impact →",
                "risk_quadrant": "Quadrant",
            },
            template="plotly_dark",
        )
        # Quadrant lines
        fig.add_hline(y=0.6, line_dash="dot", line_color="#555", line_width=1)
        fig.add_hline(y=0.4, line_dash="dot", line_color="#555", line_width=1)
        fig.add_vline(x=0.6, line_dash="dot", line_color="#555", line_width=1)
        fig.add_vline(x=0.4, line_dash="dot", line_color="#555", line_width=1)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Monte Carlo violin ────────────────────────────────────────────────────
    with col_right:
        _section("Monte Carlo Uncertainty (N=1,000)", "🎲")
        mc_df = df_enriched[["dl_label", "risk_mc_p5", "risk_mc_p50", "risk_mc_p95"]].copy()
        mc_melt = mc_df.melt(
            id_vars="dl_label",
            value_vars=["risk_mc_p5", "risk_mc_p50", "risk_mc_p95"],
            var_name="Percentile", value_name="Risk",
        )
        mc_melt["Percentile"] = mc_melt["Percentile"].map(
            {"risk_mc_p5": "P5 (Optimistic)",
             "risk_mc_p50": "P50 (Median)",
             "risk_mc_p95": "P95 (Pessimistic)"}
        )
        fig2 = px.box(
            mc_melt, x="Percentile", y="Risk", color="dl_label",
            color_discrete_map=COLOURS,
            title="Monte Carlo Risk Distribution by Severity",
            template="plotly_dark",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20), legend_title_text="",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Sector Risk Index ─────────────────────────────────────────────────────
    _section("Sector Composite Risk Index", "🏭")
    sector_df = compute_sector_risk_index(df_enriched)
    if not sector_df.empty:
        fig3 = px.bar(
            sector_df.head(12),
            x="composite_risk", y="sector",
            orientation="h",
            color="composite_risk",
            color_continuous_scale=["#06D6A0", "#FFD166", "#FF8C42", "#FF4B4B"],
            title="Sector Risk Index  (mean risk × log(incident count))",
            labels={"composite_risk": "Composite Risk", "sector": "Sector"},
            template="plotly_dark",
            text="composite_risk",
        )
        fig3.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(t=50, b=20, r=80),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Risk Trend ────────────────────────────────────────────────────────────
    trend_df = compute_risk_trend(df_enriched)
    if not trend_df.empty:
        _section("Risk Index Trend (7-day Rolling Avg)", "📈")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=trend_df["date"], y=trend_df["mean_risk"],
            name="Daily Mean Risk", mode="lines",
            line=dict(color="#7C3AED", width=1), opacity=0.5,
        ))
        fig4.add_trace(go.Scatter(
            x=trend_df["date"], y=trend_df["rolling_avg"],
            name="7-day Rolling Avg", mode="lines",
            line=dict(color="#FF8C42", width=2.5),
        ))
        fig4.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30, b=20),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig4, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — DECISION ANALYSIS PANEL
# ═════════════════════════════════════════════════════════════════════════════

def _render_decision_panel(df_enriched: pd.DataFrame):
    _section("Multi-Criteria Decision Analysis (AHP + TOPSIS)", "⚖️")

    # AHP weights
    col_w, col_cr = st.columns([3, 1])
    with col_w:
        wt = weight_table()
        fig = px.bar(
            wt, x="Criterion", y="Weight",
            color="Weight",
            color_continuous_scale=["#06D6A0", "#FF8C42", "#FF4B4B"],
            title="AHP Criteria Weights (Saaty Eigenvector Method)",
            template="plotly_dark",
            text="Weight (%)",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_cr:
        cr = consistency_ratio()
        cr_ok = cr < 0.10
        st.markdown(
            f"""
            <div style="background:{COLOURS['card']};border-left:4px solid
            {'#06D6A0' if cr_ok else '#FF4B4B'};padding:20px;border-radius:8px;
            margin-top:20px;">
                <div style="font-size:11px;color:#9CA3AF;text-transform:uppercase;">
                    AHP Consistency Ratio</div>
                <div style="font-size:36px;font-weight:700;color:#F9FAFB;">
                    {cr:.4f}</div>
                <div style="font-size:13px;color:{'#06D6A0' if cr_ok else '#FF4B4B'};">
                    {'✅ Acceptable (CR < 0.10)' if cr_ok else '⚠️ Review matrix'}</div>
                <div style="font-size:11px;color:#6B7280;margin-top:8px;">
                    Saaty's threshold: CR &lt; 0.10 indicates<br>
                    sufficiently consistent judgements.</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Prioritised incidents table
    _section("Prioritised Incident Response Queue", "🚨")
    top_df = prioritise(df_enriched, top_n=20)

    display_cols_raw = [
        "title", "post_title", "category", "country", "dl_label",
        "dl_confidence", "risk_index", "topsis_score", "ahp_score",
        "priority_score", "priority_rank", "recommendation",
    ]
    display_cols = [c for c in display_cols_raw if c in top_df.columns]

    # Title column
    title_col = next((c for c in ("title", "post_title") if c in top_df.columns), None)

    def _row_style(row):
        s = float(row.get("priority_score", 0))
        if s >= 0.75: colour = "#FF4B4B22"
        elif s >= 0.55: colour = "#FF8C4222"
        elif s >= 0.35: colour = "#FFD16622"
        else: colour = "#06D6A022"
        return [f"background-color: {colour}"] * len(row)

    styled = (
        top_df[display_cols]
        .rename(columns={
            "dl_label":       "DL Severity",
            "dl_confidence":  "Confidence",
            "risk_index":     "Risk Index",
            "topsis_score":   "TOPSIS",
            "ahp_score":      "AHP",
            "priority_score": "Priority",
            "priority_rank":  "Rank",
            "recommendation": "Action",
        })
        .style
        .apply(_row_style, axis=1)
        .format({
            "Confidence": "{:.0%}",
            "Risk Index": "{:.3f}",
            "TOPSIS":     "{:.3f}",
            "AHP":        "{:.3f}",
            "Priority":   "{:.3f}",
        }, na_rep="-")
    )
    st.dataframe(styled, use_container_width=True, height=480)

    # AHP vs TOPSIS comparison
    _section("AHP vs TOPSIS Score Comparison", "🔬")
    if "ahp_score" in top_df.columns and "topsis_score" in top_df.columns:
        fig5 = px.scatter(
            top_df,
            x="ahp_score", y="topsis_score",
            color="dl_label",
            size="priority_score",
            size_max=20,
            hover_data=[title_col] if title_col else None,
            color_discrete_map=COLOURS,
            title="AHP Score vs TOPSIS Score (size = priority)",
            labels={"ahp_score": "AHP Score →", "topsis_score": "TOPSIS Score →"},
            template="plotly_dark",
        )
        fig5.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line=dict(color="#555", dash="dot"),
        )
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Methodology expander
    with st.expander("📖 Methodology — AHP & TOPSIS Explained"):
        st.markdown(f"""
        ### Analytic Hierarchy Process (AHP)
        - Pairwise comparison matrix (6×6) built using **Saaty's 1–9 scale**
        - Priority weights derived via the **eigenvector method**
        - **Consistency Ratio = {consistency_ratio():.4f}** (threshold < 0.10)
        - Criteria: Risk Index, Impact Score, Likelihood, MC P95, Sector Weight, DL Confidence

        ### TOPSIS
        - Normalises the decision matrix using vector normalisation
        - Applies AHP-derived weights to the normalised matrix
        - Calculates Euclidean distance from **Ideal Best** and **Ideal Worst** solutions
        - Final score = distance to worst / (distance to best + distance to worst)
        - Score → 1.0 means closest to worst (most critical threat)

        ### Combined Priority Score
        ```
        priority_score = (ahp_score + topsis_score) / 2
        ```
        This equal blend ensures neither method dominates and provides
        robust, cross-validated threat prioritisation.

        ### Alignment to NAIO Malaysia 2026–2030
        | NAIO Area | How This Addresses It |
        |---|---|
        | Area 3 — AI Adaptation | Demonstrates practical AI integration in cybersecurity ops |
        | Area 4 — AI Ethics | Transparent, explainable scoring with traceable weights |
        | Area 5 — AI Impact Study | Quantified impact on decision-making efficiency |
        | Area 7 — Datasets | Live Supabase data drives all three analytical layers |
        """)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def page_ai_risk_decision(get_data_fn):
    """
    Main entry point.  Pass the get_data() function from application.py.
    """
    st.markdown(
        """
        <div style="padding:24px 0 8px;">
            <h1 style="color:#F9FAFB;font-size:2rem;font-weight:800;margin:0;">
                🧠 AI Risk Decision Centre
            </h1>
            <p style="color:#9CA3AF;margin:6px 0 0;">
                Deep Learning  ·  Quantitative Risk Analysis  ·  AHP + TOPSIS Decision Analysis
                &nbsp;|&nbsp; Aligned to Malaysia NAIO Action Plan 2026–2030
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Data loading ──────────────────────────────────────────────────────────
    with st.spinner("Loading incident data…"):
        df_raw = get_data_fn("global_news")
        if df_raw is None or df_raw.empty:
            df_raw = get_data_fn("incidents")

    if df_raw is None or df_raw.empty:
        st.error("⚠️ No data available. Check your Supabase connection.")
        return

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ AI Risk Centre Settings")
        retrain = st.button("🔄 Retrain DL Model", use_container_width=True,
                            help="Re-trains the neural network on current data")
        top_n   = st.slider("Top incidents to analyse", 50, 500, 200, 50)
        st.markdown("---")
        st.markdown(
            "<small style='color:#6B7280;'>Model trains automatically on first load.<br>"
            "Click Retrain after new data arrives.</small>",
            unsafe_allow_html=True,
        )

    df = df_raw.head(top_n).copy()

    # ── Layer 1: Train / load DL classifier ──────────────────────────────────
    cache_key = f"dl_clf_{len(df_raw)}"
    if retrain or cache_key not in st.session_state:
        with st.spinner("🧠 Training neural network…"):
            clf = DLClassifier()
            clf.train(df_raw)          # train on full dataset
            st.session_state[cache_key] = clf
            st.session_state["dl_clf_current"] = clf
    else:
        clf = st.session_state.get("dl_clf_current") or st.session_state[cache_key]

    # ── Build text blobs for prediction ──────────────────────────────────────
    title_col   = next((c for c in ("title", "post_title") if c in df.columns), None)
    summary_col = next((c for c in ("summary", "description") if c in df.columns), None)

    texts = []
    for _, row in df.iterrows():
        parts = [
            str(row.get(title_col,   "") or ""),
            str(row.get(summary_col, "") or ""),
        ]
        texts.append(" ".join(p for p in parts if p).strip() or "unknown")

    with st.spinner("🔮 Running DL predictions…"):
        dl_preds = clf.predict(texts)

    # ── Layer 2: Risk enrichment ──────────────────────────────────────────────
    with st.spinner("📊 Running risk analysis + Monte Carlo…"):
        df_enriched = enrich_with_risk(df, dl_preds)

    # ── Render panels ─────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🧠 Layer 1 — Deep Learning",
        "📊 Layer 2 — Risk Analysis",
        "⚖️ Layer 3 — Decision Analysis",
    ])

    with tab1:
        _render_dl_panel(clf, df_enriched)

    with tab2:
        _render_risk_panel(df_enriched)

    with tab3:
        _render_decision_panel(df_enriched)
