"""
utils/page_ai_risk_decision.py
────────────────────────────────────────────────────────────────────────────────
LSTM Incident Forecasting Page
────────────────────────────────
Deep Learning page that predicts future daily incident volumes using an
LSTM recurrent neural network trained on historical Supabase data.

Aligned to Malaysia NAIO Action Plan 2026–2030:
  Area 3 — Acceleration of AI Technology Adaptation
  Area 5 — AI Impact Study for Government
────────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.lstm_forecaster import LSTMForecaster

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "card":   "#1A1D27",
    "accent": "#7C3AED",
}


# ── KPI card helper ───────────────────────────────────────────────────────────

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


# ── Main render ───────────────────────────────────────────────────────────────

def _render_lstm(forecaster: LSTMForecaster):
    card = forecaster.model_card()

    # KPI row — architecture info only, no R2/MAPE
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi("Model", "LSTM", f"Hidden: {card['hidden_units']} units", C["accent"])
    with c2:
        _kpi("Forecast Horizon", card["forecast_horizon"], "Days ahead predicted", "#FF8C42")
    with c3:
        _kpi("Training Days", str(card["training_days"]), "Days used to train", "#FFD166")
    with c4:
        r2_val = card.get("r2", "N/A")
        r2_str = f"{r2_val:.4f}" if isinstance(r2_val, float) else str(r2_val)
        r2_colour = "#06D6A0" if isinstance(r2_val, float) and r2_val >= 0.7 else "#FF8C42" if isinstance(r2_val, float) and r2_val >= 0.4 else "#FF4B4B"
        _kpi("R² Score", r2_str, "Validation set (30%)", r2_colour)

    # Build forecast dataframe
    fdf = forecaster.forecast_df()
    if fdf.empty:
        st.warning("Not enough historical data to forecast. Need at least 10 days.")
        return

    hist   = fdf[~fdf["is_forecast"]].tail(60)
    future = fdf[fdf["is_forecast"]]

    # ── Forecast chart ────────────────────────────────────────────────────────
    fig = go.Figure()

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["predicted_count"],
        name="Historical Actuals",
        mode="lines+markers",
        line=dict(color="#7C3AED", width=2),
        marker=dict(size=4),
    ))

    # Connector dot — bridge between last actual and first forecast
    if not hist.empty and not future.empty:
        bridge_x = [hist["date"].iloc[-1], future["date"].iloc[0]]
        bridge_y = [hist["predicted_count"].iloc[-1], future["predicted_count"].iloc[0]]
        fig.add_trace(go.Scatter(
            x=bridge_x, y=bridge_y,
            mode="lines",
            line=dict(color="#FF8C42", width=1.5, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Uncertainty band — future only
    fig.add_trace(go.Scatter(
        x=pd.concat([future["date"], future["date"][::-1]]),
        y=pd.concat([future["upper_bound"], future["lower_bound"][::-1]]),
        fill="toself",
        fillcolor="rgba(255,140,66,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Uncertainty Band",
        hoverinfo="skip",
    ))

    # Forecast line — clearly in the future
    fig.add_trace(go.Scatter(
        x=future["date"], y=future["predicted_count"],
        name="LSTM Forecast (Future)",
        mode="lines+markers",
        line=dict(color="#FF8C42", width=2.5, dash="dash"),
        marker=dict(size=10, symbol="diamond", color="#FF8C42",
                    line=dict(color="#fff", width=1.5)),
    ))

    # Vertical divider at forecast boundary
    if not hist.empty and not future.empty:
        fig.add_vline(
            x=str(future["date"].iloc[0]),
            line_dash="dash", line_color="#888", line_width=1.5,
            annotation_text=" Forecast →",
            annotation_font_color="#FF8C42",
            annotation_font_size=13,
            annotation_position="top left",
        )

    # Shaded background for forecast region
    if not future.empty:
        fig.add_vrect(
            x0=str(future["date"].iloc[0]),
            x1=str(future["date"].iloc[-1]),
            fillcolor="rgba(255,140,66,0.05)",
            layer="below", line_width=0,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Daily Incident Count — Historical Actuals + LSTM Future Forecast",
        xaxis_title="Date",
        yaxis_title="Incident Count",
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=70, b=30),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#2A2D3A"),
        yaxis=dict(showgrid=True, gridcolor="#2A2D3A", rangemode="tozero"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast table ────────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#E5E7EB;margin-top:2rem;'>📋 Forecast Details</h3>",
        unsafe_allow_html=True,
    )
    ft = future[["date", "predicted_count", "lower_bound", "upper_bound"]].copy()
    ft["date"]            = ft["date"].dt.strftime("%a %d %b %Y")
    ft["predicted_count"] = ft["predicted_count"].astype(int)
    ft["lower_bound"]     = ft["lower_bound"].astype(int)
    ft["upper_bound"]     = ft["upper_bound"].astype(int)
    ft.columns            = ["Date", "Predicted Incidents", "Lower Bound", "Upper Bound"]
    st.dataframe(ft, use_container_width=True, hide_index=True)

    # ── Architecture expander ─────────────────────────────────────────────────
    with st.expander("🔍 LSTM Architecture Details"):
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Architecture | `LSTM({card['hidden_units']}) → Dense(32, ReLU) → Dense({forecaster.forecast_days})` |
        | Input Window | Last **{forecaster.lookback}** daily incident counts |
        | Output | Next **{forecaster.forecast_days}** daily incident counts |
        | Optimiser | `{card['optimiser']}` |
        | Loss Function | `{card['loss_function']}` |
        | Training Epochs | `{card['training_epochs']}` |
        | Total Days | `{card['training_days']} days of data` |
        | Train Split | `70% of sliding windows` |
        | Validation Split | `30% of sliding windows` |
        | R² (Validation) | `{card.get('r2', 'N/A')}` |
        | Status | `{card['status']}` |

        **Train / Validation Split:**
        The available sliding windows are split 70% for training and 30% for
        validation. The R² score is computed on the **validation set only**
        (the 30% the model never saw during training), giving an honest measure
        of how well the LSTM generalises to unseen data.

        **How it works:**
        The LSTM (Long Short-Term Memory) network learns temporal patterns in
        daily incident counts. At each time step, the LSTM cell maintains a
        **cell state** (long-term memory) and **hidden state** (short-term memory),
        controlled by three gates — Forget, Input, and Output — each with
        learnable weight matrices trained via backpropagation through time (BPTT).
        The final hidden state is passed through two Dense layers to produce the
        {forecaster.forecast_days}-day future forecast.
        """)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def page_ai_risk_decision(get_data_fn):
    st.markdown(
        """
        <div style="padding:24px 0 8px;">
            <h1 style="color:#F9FAFB;font-size:2rem;font-weight:800;margin:0;">
                🧠 LSTM Incident Forecasting
            </h1>
            <p style="color:#9CA3AF;margin:6px 0 0;">
                Deep Learning — predicts future daily incident volumes
                &nbsp;|&nbsp; Aligned to Malaysia NAIO 2026–2030
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ LSTM Settings")
        retrain       = st.button("🔄 Retrain LSTM", use_container_width=True)
        forecast_days = st.slider("Forecast horizon (days)", 3, 14, 7)
        st.markdown("---")
        st.markdown(
            "<small style='color:#6B7280;'>Model trains automatically on first load.</small>",
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

    # ── Train LSTM ────────────────────────────────────────────────────────────
    cache_key = f"lstm_{len(df_raw)}_{forecast_days}"
    if retrain or cache_key not in st.session_state:
        with st.spinner("🧠 Training LSTM…"):
            fc = LSTMForecaster(forecast_days=forecast_days)
            fc.fit(df_raw)
            st.session_state[cache_key]      = fc
            st.session_state["lstm_current"] = fc
    else:
        fc = st.session_state.get("lstm_current") or st.session_state[cache_key]

    # ── Render ────────────────────────────────────────────────────────────────
    _render_lstm(fc)
