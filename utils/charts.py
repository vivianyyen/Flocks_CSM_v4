"""
utils/charts.py
All Plotly + WordCloud chart functions for the dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

# ── Shared dark theme for all Plotly charts ──────────────────────────────────
DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans", color="#8b949e", size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    colorway=["#388bfd","#3fb950","#f78166","#d2a8ff","#ffa657","#79c0ff","#56d364"],
)

AXIS = dict(
    gridcolor="#21262d",
    zerolinecolor="#21262d",
    tickcolor="#484f58",
    linecolor="#21262d",
)


def _apply_dark(fig):
    fig.update_layout(**DARK)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    return fig


# ── Helpers ──────────────────────────────────────────────────────────────────
def _safe_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and not df[col].dropna().empty


def _placeholder(msg: str):
    st.markdown(f"<div style='color:#484f58;font-size:13px;padding:24px 0;text-align:center'>{msg}</div>", unsafe_allow_html=True)


# ── Chart functions ──────────────────────────────────────────────────────────

def render_incidents_by_category(df: pd.DataFrame):
    st.markdown("**Incidents by Category**")
    if not _safe_col(df, "category"):
        _placeholder("No category data"); return
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    fig = px.bar(counts, x="Count", y="Category", orientation="h",
                 color="Count", color_continuous_scale=["#21262d","#388bfd"])
    fig.update_coloraxes(showscale=False)
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), **DARK)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    st.plotly_chart(fig, width='stretch')


def render_incidents_by_type(df: pd.DataFrame):
    st.markdown("**Incident Types**")
    col = "incident_type"
    if not _safe_col(df, col):
        _placeholder("No incident type data"); return
    counts = df[col].value_counts().reset_index()
    counts.columns = ["Type", "Count"]
    fig = px.pie(counts, names="Type", values="Count", hole=0.55)
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(**DARK)
    st.plotly_chart(fig, width='stretch')


def render_timeline(df: pd.DataFrame):
    st.markdown("**Incidents Over Time**")
    if not _safe_col(df, "incident_date"):
        _placeholder("No date data"); return
    df2 = df.copy()
    df2["week"] = df2["incident_date"].dt.to_period("W").dt.start_time
    agg = df2.groupby(["week", "category"]).size().reset_index(name="Count") if _safe_col(df, "category") \
        else df2.groupby("week").size().reset_index(name="Count")
    if "category" in agg.columns:
        fig = px.area(agg, x="week", y="Count", color="category",
                      line_group="category")
    else:
        fig = px.area(agg, x="week", y="Count")
    fig.update_traces(line_width=1.5)
    _apply_dark(fig)
    st.plotly_chart(fig, width='stretch')


def render_impact_distribution(df: pd.DataFrame):
    st.markdown("**Impact Level**")
    if not _safe_col(df, "impact"):
        _placeholder("No impact data"); return
    order = ["Critical", "High", "Medium", "Low"]
    color_map = {
        "Critical": "#f78166",
        "High":     "#ffa657",
        "Medium":   "#e3b341",
        "Low":      "#3fb950",
    }
    counts = df["impact"].value_counts().reindex(order).dropna().reset_index()
    counts.columns = ["Impact", "Count"]
    fig = px.funnel(counts, x="Count", y="Impact",
                    color="Impact", color_discrete_map=color_map)
    fig.update_layout(**DARK)
    st.plotly_chart(fig, width='stretch')


def render_incidents_by_country(df: pd.DataFrame):
    st.markdown("**Geographic Distribution**")
    if not _safe_col(df, "country"):
        _placeholder("No country data"); return
    counts = df["country"].value_counts().reset_index()
    counts.columns = ["country", "Count"]
    fig = px.choropleth(
        counts,
        locations="country",
        locationmode="country names",
        color="Count",
        color_continuous_scale=["#0d1117", "#388bfd"],
        title=""
    )
    fig.update_geos(
        bgcolor="rgba(0,0,0,0)",
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#21262d",
        showland=True, landcolor="#161b22",
        showocean=True, oceancolor="#0d1117",
    )
    fig.update_layout(**DARK, height=320)
    st.plotly_chart(fig, width='stretch')


def render_source_breakdown(df: pd.DataFrame):
    st.markdown("**Top Sources**")
    if not _safe_col(df, "source"):
        _placeholder("No source data"); return
    counts = df["source"].value_counts().head(8).reset_index()
    counts.columns = ["Source", "Count"]
    fig = px.bar(counts, x="Count", y="Source", orientation="h",
                 color_discrete_sequence=["#388bfd"])
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), **DARK)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    st.plotly_chart(fig, width='stretch')


def render_wordcloud(df: pd.DataFrame, column: str, title: str):
    """Render a word cloud using matplotlib + wordcloud library."""
    st.markdown(f"**{title}**")
    if not _safe_col(df, column):
        _placeholder(f"No `{column}` data"); return

    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        STOPWORDS = {
            "the","a","an","and","or","but","in","on","at","to","for",
            "of","is","are","was","were","been","with","this","that",
            "from","by","as","it","its","be","have","has","had","not",
            "also","more","than","which","who","will","can","one","new",
        }

        text = " ".join(df[column].dropna().astype(str).tolist()).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        words = [w for w in text.split() if len(w) > 3 and w not in STOPWORDS]
        text_clean = " ".join(words)

        if not text_clean.strip():
            _placeholder("Not enough text data"); return

        wc = WordCloud(
            width=700, height=320,
            background_color="#0d1117",
            colormap="Blues",
            max_words=100,
            collocations=False,
        ).generate(text_clean)

        fig, ax = plt.subplots(figsize=(7, 3.2))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    except ImportError:
        # Fallback: top-30 word frequency bar chart
        from wordcloud import WordCloud
        st.caption("`wordcloud` not installed — showing top keywords instead.")
        text = " ".join(df[column].dropna().astype(str).tolist()).lower()
        words = re.findall(r"\b[a-z]{4,}\b", text)
        freq = Counter(w for w in words).most_common(20)
        if not freq:
            _placeholder("Not enough data"); return
        wdf = pd.DataFrame(freq, columns=["Word", "Count"])
        fig = px.bar(wdf, x="Count", y="Word", orientation="h",
                     color_discrete_sequence=["#388bfd"])
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), **DARK)
        st.plotly_chart(fig, width='stretch')
