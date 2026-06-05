"""
utils/page_ai_risk_decision.py
────────────────────────────────────────────────────────────────────────────────
AI Risk Decision Centre — v5
─────────────────────────────
Three integrated layers:

  Layer 1 — Deep Learning (LSTM) Incident Forecasting
    Predicts future daily incident volumes for the next 7 days using an
    LSTM recurrent neural network trained on historical Supabase data.

  Layer 2 — Quantitative Risk Analysis
    Likelihood × Impact risk matrix, Monte Carlo simulation (N=1,000),
    sector composite risk index, and rolling risk trend.

  Layer 3 — Decision Analysis (AHP)
    Analytic Hierarchy Process prioritises current incidents for response
    using expert-defined criteria weights and Saaty's eigenvector method.

Aligned to Malaysia NAIO Action Plan 2026–2030:
  Area 3 — Acceleration of AI Technology Adaptation
  Area 5 — AI Impact Study for Government
  Area 4 — AI Code of Ethics (transparent, explainable scoring)
────────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.lstm_forecaster  import LSTMForecaster
from utils.risk_analysis    import (
    enrich_with_risk,
    compute_sector_risk_index,
    compute_risk_trend,
)
from utils.decision_analysis import (
    run_ahp,
    ahp_weights,
    consistency_ratio,
    weight_table,
)

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "Critical": "#FF4B4B",
    "High":     "#FF8C42",
    "Medium":   "#FFD166",
    "Low":      "#06D6A0",
    "Monitor":  "#06D6A0",
    "Extreme":  "#FF4B4B",
    "Moderate": "#FFD166",
    "card":     "#1A1D27",
    "accent":   "#7C3AED",
}


# ── Tiny helpers ──────────────────────────────────────────────────────────────

def _kpi(title, value, sub="", colour="#7C3AED"):
    st.markdown(
        f"""<div style="background:{C['card']};border-left:4px solid {colour};
            padding:16px 20px;border-radius:8px;margin-bottom:4px;">
            <div style="font-size:11px;color:#9CA3AF;text-transform:uppercase;
                        letter-spacing:.08em;">{title}</div>
            <div style="font-size:26px;font-weight:700;color:#F9FAFB;">{value}</div>
            <div style="font-size:12px;color:#6B7280;">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _section(title, icon=""):
    st.markdown(
        f"<h3 style='color:#E5E7EB;margin-top:2rem;'>{icon} {title}</h3>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — LSTM FORECASTING
# ═════════════════════════════════════════════════════════════════════════════

def _render_lstm_panel(forecaster: LSTMForecaster, df_raw: pd.DataFrame):
    _section("LSTM Incident Volume Forecasting", "📈")

    card = forecaster.model_card()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi("Model", "LSTM", f"Hidden: {card['hidden_units']} units", C["accent"])
    with c2:
        _kpi("Input Window", card["lookback_window"], "Days of history used", "#06D6A0")
    with c3:
        _kpi("Forecast Horizon", card["forecast_horizon"], "Days ahead predicted", "#FF8C42")
    with c4:
        _kpi("Training MSE", str(card["final_mse"]), "Final training loss", "#FFD166")

    fdf = forecaster.forecast_df()
    if fdf.empty:
        st.warning("Not enough historical data to forecast (need ≥ 20 days).")
        return

    # Trim to last 60 days + forecast for readability
    hist   = fdf[~fdf["is_forecast"]].tail(60)
    future = fdf[fdf["is_forecast"]]

    fig = go.Figure()

    # Historical actual line
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["predicted_count"],
        name="Historical Actuals",
        mode="lines",
        line=dict(color="#7C3AED", width=2),
    ))

    # Forecast uncertainty band
    fig.add_trace(go.Scatter(
        x=pd.concat([future["date"], future["date"][::-1]]),
        y=pd.concat([future["upper_bound"], future["lower_bound"][::-1]]),
        fill="toself",
        fillcolor="rgba(255,140,66,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Uncertainty Band",
        showlegend=True,
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future["date"], y=future["predicted_count"],
        name="LSTM Forecast",
        mode="lines+markers",
        line=dict(color="#FF8C42", width=2.5, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
    ))

    # Divider line
    if not hist.empty and not future.empty:
        fig.add_vline(
            x=str(hist["date"].iloc[-1]),
            line_dash="dot", line_color="#555",
            annotation_text="Forecast starts →",
            annotation_font_color="#9CA3AF",
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Daily Incident Count — Historical + 7-Day LSTM Forecast",
        xaxis_title="Date",
        yaxis_title="Incident Count",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=60, b=30),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    _section("Forecast Details", "📋")
    ft = future[["date", "predicted_count", "lower_bound", "upper_bound"]].copy()
    ft["date"]            = ft["date"].dt.strftime("%a %d %b %Y")
    ft["predicted_count"] = ft["predicted_count"].astype(int)
    ft["lower_bound"]     = ft["lower_bound"].astype(int)
    ft["upper_bound"]     = ft["upper_bound"].astype(int)
    ft.columns            = ["Date", "Predicted Incidents", "Lower Bound", "Upper Bound"]
    st.dataframe(ft, use_container_width=True, hide_index=True)

    # Model card expander
    with st.expander("🔍 LSTM Architecture Details"):
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Architecture | `LSTM({card['hidden_units']}) → Dense(32, ReLU) → Dense({forecaster.forecast_days})` |
        | Input | Last **{forecaster.lookback}** daily incident counts |
        | Output | Next **{forecaster.forecast_days}** daily incident counts |
        | Optimiser | `{card['optimiser']}` |
        | Loss Function | `{card['loss_function']}` |
        | Training Epochs | `{card['training_epochs']}` |
        | Training Days | `{card['training_days']} days of data` |
        | Final MSE | `{card['final_mse']}` |
        | Status | `{card['status']}` |

        **How it works:**
        The LSTM (Long Short-Term Memory) network learns temporal patterns in
        daily incident counts. At each time step, the LSTM cell maintains a
        **cell state** (long-term memory) and **hidden state** (short-term memory),
        controlled by three gates — Forget, Input, and Output — each with
        learnable weight matrices trained via backpropagation through time (BPTT).
        The final hidden state is passed through two Dense layers to produce the
        {forecaster.forecast_days}-day forecast.
        """)


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — RISK ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def _render_risk_panel(df_enriched: pd.DataFrame):
    _section("Quantitative Risk Analysis", "📊")

    mean_risk = df_enriched["risk_index"].mean()
    max_risk  = df_enriched["risk_index"].max()
    extreme_n = (df_enriched["risk_quadrant"] == "Extreme").sum()
    mc_spread = (df_enriched["risk_mc_p95"] - df_enriched["risk_mc_p5"]).mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi("Mean Risk Index", f"{mean_risk:.3f}", "Likelihood × Impact", C["accent"])
    with c2:
        _kpi("Peak Risk", f"{max_risk:.3f}", "Highest single incident", C["Critical"])
    with c3:
        _kpi("Extreme Zone", str(int(extreme_n)), "ISO 31000 Extreme quadrant", C["High"])
    with c4:
        _kpi("MC Uncertainty", f"±{mc_spread:.3f}", "Avg P5–P95 spread", C["Medium"])

    col_l, col_r = st.columns(2)

    with col_l:
        _section("Risk Matrix", "🎯")
        fig = px.scatter(
            df_enriched.head(300),
            x="likelihood", y="impact_score",
            color="risk_quadrant",
            size="risk_index", size_max=18,
            color_discrete_map={
                "Extreme": C["Critical"], "High": C["High"],
                "Moderate": C["Medium"], "Low": C["Low"],
            },
            title="ISO 31000 Risk Matrix (Likelihood × Impact)",
            labels={"likelihood": "Likelihood →", "impact_score": "Impact →",
                    "risk_quadrant": "Zone"},
            template="plotly_dark",
        )
        fig.add_hline(y=0.6, line_dash="dot", line_color="#444", line_width=1)
        fig.add_hline(y=0.4, line_dash="dot", line_color="#444", line_width=1)
        fig.add_vline(x=0.6, line_dash="dot", line_color="#444", line_width=1)
        fig.add_vline(x=0.4, line_dash="dot", line_color="#444", line_width=1)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        _section("Monte Carlo Uncertainty (N=1,000)", "🎲")
        mc_df = df_enriched[["severity", "risk_mc_p5", "risk_mc_p50", "risk_mc_p95"]].copy()
        if "severity" not in mc_df.columns:
            mc_df["severity"] = df_enriched.get("dl_label", "Unknown")
        mc_melt = mc_df.melt(
            id_vars="severity",
            value_vars=["risk_mc_p5", "risk_mc_p50", "risk_mc_p95"],
            var_name="Percentile", value_name="Risk",
        )
        mc_melt["Percentile"] = mc_melt["Percentile"].map({
            "risk_mc_p5":  "P5 — Optimistic",
            "risk_mc_p50": "P50 — Median",
            "risk_mc_p95": "P95 — Pessimistic",
        })
        fig2 = px.box(
            mc_melt, x="Percentile", y="Risk", color="severity",
            color_discrete_map=C,
            title="Monte Carlo Risk Distribution by Severity",
            template="plotly_dark",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20), legend_title_text="",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Sector risk
    _section("Sector Composite Risk Index", "🏭")
    sector_df = compute_sector_risk_index(df_enriched)
    if not sector_df.empty:
        fig3 = px.bar(
            sector_df.head(12),
            x="composite_risk", y="sector", orientation="h",
            color="composite_risk",
            color_continuous_scale=["#06D6A0", "#FFD166", "#FF8C42", "#FF4B4B"],
            title="Sector Risk Index = Mean Risk × log(1 + Incident Count)",
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

    # Risk trend
    trend_df = compute_risk_trend(df_enriched)
    if not trend_df.empty:
        _section("Risk Index Trend (7-day Rolling Average)", "📉")
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
#  LAYER 3 — AHP DECISION ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def _render_ahp_panel(df_enriched: pd.DataFrame):
    _section("AHP Decision Analysis — Incident Prioritisation", "⚖️")

    col_w, col_cr = st.columns([3, 1])

    with col_w:
        wt = weight_table()
        fig = px.bar(
            wt, x="Criterion", y="Weight",
            color="Weight",
            color_continuous_scale=["#06D6A0", "#FF8C42", "#FF4B4B"],
            title="AHP Criteria Weights  (Saaty Eigenvector Method)",
            template="plotly_dark",
            text="Weight (%)",
            hover_data=["Meaning"],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_cr:
        cr    = consistency_ratio()
        cr_ok = cr < 0.10
        colour = "#06D6A0" if cr_ok else "#FF4B4B"
        st.markdown(
            f"""<div style="background:{C['card']};border-left:4px solid {colour};
                padding:20px;border-radius:8px;margin-top:24px;">
                <div style="font-size:11px;color:#9CA3AF;text-transform:uppercase;">
                    Consistency Ratio</div>
                <div style="font-size:36px;font-weight:700;color:#F9FAFB;">{cr:.4f}</div>
                <div style="font-size:13px;color:{colour};">
                    {'✅ Acceptable (CR < 0.10)' if cr_ok else '⚠️ Review matrix'}</div>
                <div style="font-size:11px;color:#6B7280;margin-top:8px;">
                    Saaty: CR &lt; 0.10 confirms<br>consistent judgements.</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Run AHP
    df_ahp = run_ahp(df_enriched)

    # KPIs
    critical_n = (df_ahp["priority_tier"] == "Critical").sum()
    high_n     = (df_ahp["priority_tier"] == "High").sum()
    mean_ahp   = df_ahp["ahp_score"].mean()

    c1, c2, c3 = st.columns(3)
    with c1:
        _kpi("Critical Incidents", str(int(critical_n)),
             "AHP score ≥ 0.75", C["Critical"])
    with c2:
        _kpi("High Priority", str(int(high_n)),
             "AHP score 0.50–0.75", C["High"])
    with c3:
        _kpi("Mean AHP Score", f"{mean_ahp:.3f}",
             "Across all incidents", C["accent"])

    # Priority distribution
    col_a, col_b = st.columns(2)

    with col_a:
        tier_counts = df_ahp["priority_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig2 = px.pie(
            tier_counts, names="Tier", values="Count",
            color="Tier",
            color_discrete_map=C,
            title="AHP Priority Tier Distribution",
            hole=0.45, template="plotly_dark",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        fig3 = px.histogram(
            df_ahp, x="ahp_score", nbins=20,
            color="priority_tier",
            color_discrete_map=C,
            title="AHP Score Distribution",
            labels={"ahp_score": "AHP Score", "priority_tier": "Priority"},
            template="plotly_dark",
            barmode="stack",
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20), legend_title_text="",
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Response queue table
    _section("Top 20 Priority Incidents — Response Queue", "🚨")
    title_col = next((c for c in ("title", "post_title") if c in df_ahp.columns), None)
    show_cols_raw = [
        title_col, "category", "country", "severity",
        "risk_index", "likelihood", "impact_score",
        "ahp_score", "priority_tier", "recommendation",
    ]
    show_cols = [c for c in show_cols_raw if c and c in df_ahp.columns]

    top20 = df_ahp.head(20)[show_cols].rename(columns={
        title_col:      "Incident"   if title_col else None,
        "risk_index":   "Risk Index",
        "likelihood":   "Likelihood",
        "impact_score": "Impact",
        "ahp_score":    "AHP Score",
        "priority_tier": "Priority",
        "recommendation": "Action",
        "severity":     "Severity",
    })

    def _row_colour(row):
        t = row.get("Priority", "")
        bg = {"Critical": "#FF4B4B22", "High": "#FF8C4222",
              "Medium": "#FFD16622"}.get(t, "#06D6A022")
        return [f"background-color:{bg}"] * len(row)

    styled = (
        top20.style
        .apply(_row_colour, axis=1)
        .format({"Risk Index": "{:.3f}", "Likelihood": "{:.3f}",
                 "Impact": "{:.3f}", "AHP Score": "{:.3f}"}, na_rep="-")
    )
    st.dataframe(styled, use_container_width=True, height=480)

    # Methodology expander
    with st.expander("📖 AHP Methodology Explained"):
        weights = ahp_weights()
        st.markdown(f"""
        ### Analytic Hierarchy Process (Saaty, 1980)

        **Step 1 — Pairwise Comparison Matrix**
        A 5×5 matrix captures the relative importance of each criterion
        using Saaty's 1–9 scale (1 = equal, 9 = extreme importance).

        **Step 2 — Normalisation**
        Each column is divided by its sum, then row averages give the
        **priority weights** (principal eigenvector approximation).

        **Step 3 — Consistency Check**
        Consistency Ratio = **{consistency_ratio():.4f}** (threshold < 0.10 ✅)
        This confirms the pairwise judgements are logically consistent.

        **Step 4 — Scoring**
        Each incident is scored as the weighted sum of its normalised
        criteria values:

        ```
        AHP Score = {' + '.join([f'{w:.3f}×{c.replace("_"," ")}' for c, w in weights.items()])}
        ```

        **Priority Tiers**
        | Score | Tier | Action |
        |---|---|---|
        | ≥ 0.75 | Critical | Escalate within 1 hour |
        | 0.50–0.75 | High | Investigate within 24 hours |
        | 0.30–0.50 | Medium | Review within 72 hours |
        | < 0.30 | Monitor | Weekly report |

        **Alignment to NAIO Malaysia 2026–2030**
        | Area | Contribution |
        |---|---|
        | Area 3 — AI Adaptation | Demonstrates practical AI decision support |
        | Area 4 — AI Ethics | Transparent, traceable, explainable scoring |
        | Area 5 — AI Impact Study | Quantified improvement in response prioritisation |
        """)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def page_ai_risk_decision(get_data_fn):
    st.markdown(
        """
        <div style="padding:24px 0 8px;">
            <h1 style="color:#F9FAFB;font-size:2rem;font-weight:800;margin:0;">
                🧠 AI Risk Decision Centre
            </h1>
            <p style="color:#9CA3AF;margin:6px 0 0;">
                LSTM Forecasting &nbsp;·&nbsp; Quantitative Risk Analysis
                &nbsp;·&nbsp; AHP Decision Analysis
                &nbsp;|&nbsp; Aligned to Malaysia NAIO 2026–2030
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ AI Risk Centre")
        retrain      = st.button("🔄 Retrain LSTM", use_container_width=True,
                                 help="Re-trains the LSTM on current data")
        forecast_days = st.slider("Forecast horizon (days)", 3, 14, 7)
        lookback      = st.slider("LSTM lookback window (days)", 7, 30, 14)
        top_n         = st.slider("Incidents to analyse", 50, 500, 200, 50)
        st.markdown("---")
        st.markdown(
            "<small style='color:#6B7280;'>LSTM trains automatically on first load.</small>",
            unsafe_allow_html=True,
        )

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading data…"):
        df_raw = get_data_fn("global_news")
        if df_raw is None or df_raw.empty:
            df_raw = get_data_fn("incidents")

    if df_raw is None or df_raw.empty:
        st.error("⚠️ No data available. Check your Supabase connection.")
        return

    df = df_raw.head(top_n).copy()

    # ── Train LSTM ────────────────────────────────────────────────────────────
    cache_key = f"lstm_{len(df_raw)}_{forecast_days}_{lookback}"
    if retrain or cache_key not in st.session_state:
        with st.spinner("🧠 Training LSTM forecaster…"):
            fc = LSTMForecaster(
                lookback      = lookback,
                forecast_days = forecast_days,
            )
            fc.fit(df_raw)
            st.session_state[cache_key]     = fc
            st.session_state["lstm_current"] = fc
    else:
        fc = st.session_state.get("lstm_current") or st.session_state[cache_key]

    # ── Risk enrichment (uses existing rule-based severity as fallback) ────────
    with st.spinner("📊 Running risk analysis…"):
        df_enriched = enrich_with_risk(df)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📈 Layer 1 — LSTM Forecasting",
        "📊 Layer 2 — Risk Analysis",
        "⚖️ Layer 3 — AHP Decision",
    ])

    with tab1:
        _render_lstm_panel(fc, df_raw)

    with tab2:
        _render_risk_panel(df_enriched)

    with tab3:
        _render_ahp_panel(df_enriched)
