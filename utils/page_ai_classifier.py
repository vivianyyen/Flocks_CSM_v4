"""
utils/page_ai_classifier.py
────────────────────────────────────────────────────────────────────────────────
AI Incident Type Classifier Page
──────────────────────────────────
Classifies cybersecurity incidents into attack types using an MLP neural network
trained on TF-IDF features from incident titles and summaries.

Sections:
  1. Evaluation Card  — key metrics (Accuracy, Precision, Recall, F1)
  2. Training Curve   — loss over epochs
  3. Confusion Matrix — heatmap
  4. Per-Class Report — precision / recall / F1 per incident type
  5. Live Classifier  — analyst inputs a headline, model predicts instantly
────────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from utils.incident_classifier import IncidentClassifier

# ── Colours ───────────────────────────────────────────────────────────────────
C = {
    'card':    '#1A1D27',
    'accent':  '#7C3AED',
    'green':   '#06D6A0',
    'orange':  '#FF8C42',
    'yellow':  '#FFD166',
    'red':     '#FF4B4B',
    'text':    '#F9FAFB',
    'subtext': '#9CA3AF',
}

SEVERITY_COLOUR = {
    'Intrusion System':          '#FF4B4B',
    'Data Breach':               '#FFD166',
    'Compromise of Credentials': '#60A5FA',
    'Fraud':                     '#34D399',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kpi(col, title, value, sub='', colour='#7C3AED'):
    with col:
        st.markdown(
            f"""<div style="background:{C['card']};border-left:4px solid {colour};
                padding:16px 20px;border-radius:8px;margin-bottom:4px;">
                <div style="font-size:11px;color:{C['subtext']};text-transform:uppercase;
                            letter-spacing:.08em;">{title}</div>
                <div style="font-size:28px;font-weight:700;color:{C['text']};">{value}</div>
                <div style="font-size:12px;color:#6B7280;">{sub}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def _section(title, icon=''):
    st.markdown(
        f"<h3 style='color:#E5E7EB;margin-top:2rem;'>{icon} {title}</h3>",
        unsafe_allow_html=True,
    )


def _metric_colour(val: float, thresholds=(0.70, 0.50)) -> str:
    if val >= thresholds[0]: return C['green']
    if val >= thresholds[1]: return C['orange']
    return C['red']


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — EVALUATION CARD
# ═════════════════════════════════════════════════════════════════════════════

def _render_eval_card(clf: IncidentClassifier):
    m = clf.eval_metrics()
    card = clf.model_card()

    _section('Model Evaluation Card', '📊')

    # Row 1 — main metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    _kpi(c1, 'Accuracy',         f"{m['accuracy']*100:.1f}%",
         'Overall correctness', _metric_colour(m['accuracy']))
    _kpi(c2, 'Precision (wtd)',  f"{m['precision_weighted']:.4f}",
         'Weighted avg',         _metric_colour(m['precision_weighted']))
    _kpi(c3, 'Recall (wtd)',     f"{m['recall_weighted']:.4f}",
         'Weighted avg',         _metric_colour(m['recall_weighted']))
    _kpi(c4, 'F1 Score (wtd)',   f"{m['f1_weighted']:.4f}",
         'Weighted avg',         _metric_colour(m['f1_weighted']))
    _kpi(c5, 'F1 Score (macro)', f"{m['f1_macro']:.4f}",
         'Unweighted avg',       _metric_colour(m['f1_macro']))

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 — model info
    c1, c2, c3, c4, c5 = st.columns(5)
    _kpi(c1, 'Architecture',  'MLP',            '256 → 128 → 64',  C['accent'])
    _kpi(c2, 'Classes',       str(card['n_classes']),
         'Incident types',   C['accent'])
    _kpi(c3, 'Train Samples', str(card['train_samples']),
         '80% of dataset',   C['yellow'])
    _kpi(c4, 'Test Samples',  str(card['test_samples']),
         '20% of dataset',   C['yellow'])
    _kpi(c5, 'Epochs',        str(card['epochs_run']),
         'Early stopping',   C['orange'])

    # Expandable detail table
    with st.expander('📋 Full Model Card'):
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Architecture | `TF-IDF → MLP (256, 128, 64) → {card['n_classes']} classes` |
        | Activation | `ReLU` |
        | Optimiser | `Adam` |
        | Regularisation | `L2 (α=0.001) + Early Stopping (patience=15)` |
        | Vocabulary Size | `{card['vocab_size']:,} TF-IDF features` |
        | N-gram Range | `Unigrams + Bigrams` |
        | Train / Test Split | `80% / 20% stratified` |
        | Training Samples | `{card['train_samples']}` |
        | Test Samples | `{card['test_samples']}` |
        | Epochs Run | `{card['epochs_run']}` |
        | Accuracy | `{m['accuracy']*100:.2f}%` |
        | Precision (wtd) | `{m['precision_weighted']:.4f}` |
        | Recall (wtd) | `{m['recall_weighted']:.4f}` |
        | F1 (weighted) | `{m['f1_weighted']:.4f}` |
        | F1 (macro) | `{m['f1_macro']:.4f}` |
        | Status | `{card['status']}` |

        **What each metric means:**
        | Metric | Meaning |
        |---|---|
        | Accuracy | % of all predictions that are correct |
        | Precision | Of all incidents predicted as type X, how many actually are X |
        | Recall | Of all actual type X incidents, how many did the model catch |
        | F1 (weighted) | Harmonic mean of precision & recall, weighted by class size |
        | F1 (macro) | Same but treats all classes equally regardless of size |
        """)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — TRAINING CURVE
# ═════════════════════════════════════════════════════════════════════════════

def _render_training_curve(clf: IncidentClassifier):
    _section('Training Loss Curve', '📉')
    loss = clf.loss_curve
    if not loss:
        st.info('No training curve available.')
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(loss) + 1)),
        y=loss,
        mode='lines',
        name='Training Loss',
        line=dict(color=C['accent'], width=2),
        fill='tozeroy',
        fillcolor='rgba(124,58,237,0.1)',
    ))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title='MLP Training Loss (Cross-Entropy) over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        margin=dict(t=50, b=30),
        xaxis=dict(showgrid=True, gridcolor='#2A2D3A'),
        yaxis=dict(showgrid=True, gridcolor='#2A2D3A'),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — CONFUSION MATRIX
# ═════════════════════════════════════════════════════════════════════════════

def _render_confusion_matrix(clf: IncidentClassifier):
    _section('Confusion Matrix', '🔢')
    cm      = clf.confusion_matrix()
    classes = clf.classes()

    if cm is None:
        st.info('No confusion matrix available.')
        return

    fig = px.imshow(
        cm,
        labels=dict(x='Predicted', y='Actual', color='Count'),
        x=classes,
        y=classes,
        color_continuous_scale='Blues',
        text_auto=True,
        title='Confusion Matrix — Actual vs Predicted Incident Type',
        aspect='auto',
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=80),
        xaxis=dict(tickangle=45),
        coloraxis_showscale=False,
        font=dict(size=11),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — PER-CLASS REPORT
# ═════════════════════════════════════════════════════════════════════════════

def _render_per_class(clf: IncidentClassifier):
    _section('Per-Class Performance Report', '📋')

    report_df = clf.per_class_report()
    if report_df.empty:
        st.info('No per-class report available.')
        return

    col_l, col_r = st.columns(2)

    # F1 bar chart
    with col_l:
        fig = px.bar(
            report_df.sort_values('F1 Score'),
            x='F1 Score', y='Class',
            orientation='h',
            color='F1 Score',
            color_continuous_scale=['#FF4B4B', '#FFD166', '#06D6A0'],
            title='F1 Score per Incident Type',
            text='F1 Score',
            template='plotly_dark',
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            margin=dict(t=50, b=20, r=60),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Precision vs Recall scatter
    with col_r:
        fig2 = px.scatter(
            report_df,
            x='Precision', y='Recall',
            size='Support',
            color='F1 Score',
            color_continuous_scale=['#FF4B4B', '#FFD166', '#06D6A0'],
            hover_data=['Class', 'F1 Score', 'Support'],
            text='Class',
            title='Precision vs Recall (size = support)',
            template='plotly_dark',
        )
        fig2.update_traces(textposition='top center', textfont_size=8)
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            margin=dict(t=50, b=20),
            xaxis=dict(range=[0, 1.1]),
            yaxis=dict(range=[0, 1.1]),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Styled table
    def _colour_f1(val):
        if val >= 0.70: return 'background-color:#06D6A022;color:#06D6A0'
        if val >= 0.50: return 'background-color:#FFD16622;color:#FFD166'
        return 'background-color:#FF4B4B22;color:#FF4B4B'

    styled = (
        report_df.style
        .map(_colour_f1, subset=['F1 Score'])
        .format({'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1 Score': '{:.3f}'})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — LIVE CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════

def _render_live_classifier(clf: IncidentClassifier):
    _section('Live Incident Classifier', '🔍')

    st.markdown(
        "<p style='color:#9CA3AF;'>Paste any incident headline or summary — "
        "the model will predict the attack type instantly.</p>",
        unsafe_allow_html=True,
    )

    text_input = st.text_area(
        'Incident headline / summary',
        placeholder='e.g. LockBit ransomware group claims attack on Malaysian government ministry...',
        height=100,
    )

    if st.button('🔮 Classify Incident', use_container_width=True):
        if not text_input.strip():
            st.warning('Please enter some text.')
            return

        result = clf.predict_one(text_input.strip())
        label  = result['label']
        conf   = result['confidence']
        top3   = result['top3']
        colour = SEVERITY_COLOUR.get(label, C['accent'])

        # Result banner
        st.markdown(
            f"""<div style="background:{C['card']};border-left:6px solid {colour};
                padding:20px 24px;border-radius:10px;margin:16px 0;">
                <div style="font-size:12px;color:{C['subtext']};text-transform:uppercase;
                            letter-spacing:.1em;">Predicted Incident Type</div>
                <div style="font-size:32px;font-weight:800;color:{colour};
                            margin:6px 0;">{label}</div>
                <div style="font-size:14px;color:{C['subtext']};">
                    Confidence: <span style="color:{colour};font-weight:600;">
                    {conf*100:.1f}%</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Top 3 probabilities bar chart
        top3_df = pd.DataFrame(top3, columns=['Incident Type', 'Probability'])
        top3_df['Probability %'] = (top3_df['Probability'] * 100).round(1)

        fig = px.bar(
            top3_df,
            x='Probability %', y='Incident Type',
            orientation='h',
            color='Probability %',
            color_continuous_scale=['#7C3AED', colour],
            text='Probability %',
            title='Top 3 Predictions',
            template='plotly_dark',
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            margin=dict(t=50, b=20, r=60),
            yaxis=dict(autorange='reversed'),
            height=200,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Quick test examples
    st.markdown(
        "<p style='color:#6B7280;font-size:12px;margin-top:1rem;'>💡 Try these examples:</p>",
        unsafe_allow_html=True,
    )
    examples = [
        # Intrusion System
        'LockBit ransomware group claims attack on Malaysian government ministry demands payment',
        'APT group linked to nation-state deploys backdoor malware in Southeast Asian telecoms',
        # Data Breach
        'CIMB Bank Malaysia suffers data breach exposing customer financial records on dark web',
        # Compromise of Credentials
        'Phishing campaign targets Maybank users via fake SMS login pages stealing credentials',
        # Fraud
        'Scammers impersonate LHDN officers in phone fraud targeting Malaysian taxpayers',
    ]
    for ex in examples:
        if st.button(f'📌 {ex[:75]}...', key=ex):
            result = clf.predict_one(ex)
            label  = result['label']
            conf   = result['confidence']
            colour = SEVERITY_COLOUR.get(label, C['accent'])
            st.markdown(
                f"""<div style="background:{C['card']};border-left:4px solid {colour};
                    padding:12px 16px;border-radius:8px;margin:4px 0;">
                    <span style="color:{colour};font-weight:700;">{label}</span>
                    <span style="color:{C['subtext']};font-size:12px;margin-left:12px;">
                    {conf*100:.1f}% confidence</span>
                </div>""",
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def page_ai_classifier(get_data_fn):
    st.markdown(
        """<div style="padding:24px 0 8px;">
            <h1 style="color:#F9FAFB;font-size:2rem;font-weight:800;margin:0;">
                🧠 AI Incident Type Classifier
            </h1>
            <p style="color:#9CA3AF;margin:6px 0 0;">
                MLP Neural Network · TF-IDF Text Features · 4 Incident Categories
                &nbsp;|&nbsp; Aligned to Malaysia NAIO 2026–2030
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('### ⚙️ Classifier Settings')
        retrain = st.button('🔄 Retrain Model', use_container_width=True,
                            help='Retrain the MLP on current Supabase data')
        st.markdown('---')
        st.markdown(
            "<small style='color:#6B7280;'>Model trains automatically on first load.<br>"
            'Click Retrain after new incidents are added.</small>',
            unsafe_allow_html=True,
        )

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner('Loading incident data…'):
        df_gn  = get_data_fn('global_news')
        df_inc = get_data_fn('incidents')

        frames = [f for f in [df_gn, df_inc] if f is not None and not f.empty]
        if not frames:
            st.error('⚠️ No data available. Check your Supabase connection.')
            return

        import pandas as _pd
        df_all = _pd.concat(frames, ignore_index=True)

    # ── Train classifier ──────────────────────────────────────────────────────
    cache_key = f'clf_{len(df_all)}'
    if retrain or cache_key not in st.session_state:
        with st.spinner('🧠 Training MLP classifier…'):
            clf = IncidentClassifier()
            clf.train(df_all)
            st.session_state[cache_key]     = clf
            st.session_state['clf_current'] = clf
    else:
        clf = st.session_state.get('clf_current') or st.session_state[cache_key]

    if not clf.is_ready():
        st.error('Model could not be trained. Check that incident_type labels exist in the data.')
        return

    # ── Live Classifier only ──────────────────────────────────────────────────
    _render_live_classifier(clf)
