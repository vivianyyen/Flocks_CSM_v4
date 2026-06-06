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
import pickle
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
warnings.filterwarnings('ignore')

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
    'Scam/Fraud':                '#34D399',
    'Other Cyber Incident':      '#9CA3AF',
    'Other':                     '#9CA3AF',
}


# ── Incident Classifier Class ─────────────────────────────────────────────────

class IncidentClassifier:
    """MLP Classifier for cybersecurity incident types using TF-IDF features."""
    
    def __init__(self):
        self.vectorizer = None
        self.label_encoder = None
        self.mlp = None
        self.is_trained = False
        self.classes_ = None
        self.eval_metrics_cache = None
        self.loss_curve_cache = None
        self.confusion_matrix_cache = None
        self.per_class_report_cache = None
        self.model_card_cache = None
    
    def train(self, df, text_columns=None, label_column='incident_type'):
        """
        Train the MLP classifier on incident data.
        
        Args:
            df: DataFrame with incident data
            text_columns: list of column names to combine for text features
            label_column: column name for target labels
        """
        if text_columns is None:
            text_columns = ['title', 'summary']
        
        # Combine text columns
        available_cols = [c for c in text_columns if c in df.columns]
        if not available_cols:
            st.error(f"None of {text_columns} columns found in data")
            return False
        
        df = df.copy()
        df['text'] = df[available_cols[0]].fillna('')
        for col in available_cols[1:]:
            df['text'] += ' ' + df[col].fillna('')
        
        # Drop rows with no label
        df = df.dropna(subset=[label_column])
        df = df[df[label_column].str.strip() != '']
        
        # Label mapping - consolidate into categories
        label_map = {
            # Intrusion System
            'Malware':                          'Intrusion System',
            'Ransomware':                       'Intrusion System',
            'Ransomware Attack':                'Intrusion System',
            'Advanced Persistent Threat (APT)': 'Intrusion System',
            'APT':                              'Intrusion System',
            'DDoS':                             'Intrusion System',
            'Vulnerability':                    'Intrusion System',
            'Zero-Day':                         'Intrusion System',
            'Supply Chain':                     'Intrusion System',
            'Supply Chain Attack':              'Intrusion System',
            'Unauthorised Access':              'Intrusion System',
            
            # Data Breach
            'Data Breach':                      'Data Breach',
            'Insider Threat':                   'Data Breach',
            'Data Leak':                        'Data Breach',
            'Sell Data':                        'Data Breach',
            
            # Compromise of Credentials
            'Phishing':                         'Compromise of Credentials',
            'Social Engineering':               'Compromise of Credentials',
            'Credential Stuffing':              'Compromise of Credentials',
            'Account Takeover':                 'Compromise of Credentials',
            'Brute Force':                      'Compromise of Credentials',
            
            # Fraud
            'Fraud':                            'Scam/Fraud',
            'Financial Fraud':                  'Scam/Fraud',
            'Scam/Fraud':                       'Scam/Fraud',
            
            # Other categories
            'Cybersecurity':                    'Other',
            'Other Cyber Incident':             'Other Cyber Incident',
            'Others':                           'Other',
        }
        
        df[label_column] = df[label_column].replace(label_map)
        
        # Remove ambiguous classes
        df = df[~df[label_column].isin(['Multiple'])]
        
        # Keep classes with enough samples (>= 10)
        counts = df[label_column].value_counts()
        valid = counts[counts >= 10].index
        df = df[df[label_column].isin(valid)].reset_index(drop=True)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df[label_column])
        self.classes_ = list(self.label_encoder.classes_)
        
        # Train/test split
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df['text'], y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
        
        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents='unicode',
            stop_words='english',
        )
        
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)
        
        # Train MLP classifier
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            verbose=False,
        )
        
        self.mlp.fit(X_train, y_train)
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.mlp.predict(X_test)
        
        self.is_trained = True
        
        # Clear cached metrics
        self.eval_metrics_cache = None
        self.loss_curve_cache = None
        self.confusion_matrix_cache = None
        self.per_class_report_cache = None
        self.model_card_cache = None
        
        return True
    
    def is_ready(self):
        """Check if model is trained and ready."""
        return self.is_trained and self.mlp is not None
    
    def eval_metrics(self):
        """Return evaluation metrics dict."""
        if self.eval_metrics_cache is not None:
            return self.eval_metrics_cache
        
        if not self.is_ready():
            return {}
        
        accuracy = accuracy_score(self.y_test, self.y_pred)
        f1_macro = f1_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        f1_weight = f1_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        precision = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        
        self.eval_metrics_cache = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weight,
            'precision_weighted': precision,
            'recall_weighted': recall,
        }
        return self.eval_metrics_cache
    
    @property
    def loss_curve(self):
        """Return training loss curve."""
        if self.loss_curve_cache is not None:
            return self.loss_curve_cache
        if self.is_ready():
            self.loss_curve_cache = self.mlp.loss_curve_
            return self.loss_curve_cache
        return []
    
    def confusion_matrix(self):
        """Return confusion matrix."""
        if self.confusion_matrix_cache is not None:
            return self.confusion_matrix_cache
        if self.is_ready():
            self.confusion_matrix_cache = confusion_matrix(self.y_test, self.y_pred)
            return self.confusion_matrix_cache
        return None
    
    def per_class_report(self):
        """Return DataFrame with per-class metrics."""
        if self.per_class_report_cache is not None:
            return self.per_class_report_cache
        
        if not self.is_ready():
            return pd.DataFrame()
        
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=self.classes_,
            output_dict=True,
            zero_division=0
        )
        
        rows = []
        for cls in self.classes_:
            if cls in report:
                rows.append({
                    'Class': cls,
                    'Precision': report[cls]['precision'],
                    'Recall': report[cls]['recall'],
                    'F1 Score': report[cls]['f1-score'],
                    'Support': report[cls]['support'],
                })
        
        self.per_class_report_cache = pd.DataFrame(rows)
        return self.per_class_report_cache
    
    def model_card(self):
        """Return model information card."""
        if self.model_card_cache is not None:
            return self.model_card_cache
        
        if not self.is_ready():
            return {}
        
        self.model_card_cache = {
            'n_classes': len(self.classes_),
            'classes': self.classes_,
            'vocab_size': len(self.vectorizer.get_feature_names_out()),
            'epochs_run': self.mlp.n_iter_,
            'status': 'Trained' if self.is_trained else 'Not trained',
            'train_samples': len(self.X_test) * 4,
            'test_samples': len(self.X_test),
        }
        return self.model_card_cache
    
    def predict_one(self, text):
        """Predict incident type for a single text input."""
        if not self.is_ready():
            return {'label': 'Error', 'confidence': 0, 'top3': []}
        
        vec = self.vectorizer.transform([text])
        pred = self.mlp.predict(vec)[0]
        proba = self.mlp.predict_proba(vec)[0]
        label = self.label_encoder.inverse_transform([pred])[0]
        
        # Get top 3 predictions
        top3_idx = np.argsort(proba)[-3:][::-1]
        top3 = [(self.label_encoder.inverse_transform([i])[0], proba[i]) for i in top3_idx]
        
        return {
            'label': label,
            'confidence': proba.max(),
            'top3': top3,
        }
    
    def save(self, path_prefix='./models/'):
        """Save model artefacts to disk."""
        import os
        os.makedirs(path_prefix, exist_ok=True)
        
        with open(f'{path_prefix}mlp_classifier.pkl', 'wb') as f:
            pickle.dump(self.mlp, f)
        with open(f'{path_prefix}tfidf_vectoriser.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(f'{path_prefix}label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def load(self, path_prefix='./models/'):
        """Load model artefacts from disk."""
        try:
            with open(f'{path_prefix}mlp_classifier.pkl', 'rb') as f:
                self.mlp = pickle.load(f)
            with open(f'{path_prefix}tfidf_vectoriser.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(f'{path_prefix}label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.classes_ = list(self.label_encoder.classes_)
            self.is_trained = True
            return True
        except FileNotFoundError:
            return False


# ── UI Components ────────────────────────────────────────────────────────────

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


def _render_eval_card(clf: IncidentClassifier):
    m = clf.eval_metrics()
    card = clf.model_card()

    _section('Model Evaluation Card', '📊')

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
        """)


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


def _render_confusion_matrix(clf: IncidentClassifier):
    _section('Confusion Matrix', '🔢')
    cm = clf.confusion_matrix()
    classes = clf.classes_

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


def _render_per_class(clf: IncidentClassifier):
    _section('Per-Class Performance Report', '📋')

    report_df = clf.per_class_report()
    if report_df.empty:
        st.info('No per-class report available.')
        return

    col_l, col_r = st.columns(2)

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
        label = result['label']
        conf = result['confidence']
        top3 = result['top3']
        colour = SEVERITY_COLOUR.get(label, C['accent'])

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

    st.markdown(
        "<p style='color:#6B7280;font-size:12px;margin-top:1rem;'>💡 Try these examples:</p>",
        unsafe_allow_html=True,
    )
    
    examples = [
        ('🔴 Intrusion System', 'LockBit ransomware group claims attack on Malaysian government ministry demands payment'),
        ('🔴 Intrusion System', 'APT group linked to nation-state deploys backdoor malware in Southeast Asian telecoms'),
        ('🟡 Data Breach', 'CIMB Bank Malaysia suffers data breach exposing customer financial records on dark web'),
        ('🔵 Compromise of Credentials', 'Phishing campaign targets Maybank users via fake SMS login pages stealing credentials'),
        ('🟢 Scam/Fraud', 'Scammers impersonate LHDN officers in phone fraud targeting Malaysian taxpayers'),
    ]
    
    cols = st.columns(2)
    for i, (emoji, ex) in enumerate(examples):
        with cols[i % 2]:
            if st.button(f'{emoji} {ex[:60]}...', key=ex):
                result = clf.predict_one(ex)
                label = result['label']
                conf = result['confidence']
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


def _render_top_features(clf: IncidentClassifier):
    """Display top TF-IDF features per class."""
    if not clf.is_ready():
        return
    
    _section('Top Influential Words per Class', '📝')
    
    feature_names = np.array(clf.vectorizer.get_feature_names_out())
    weights = np.abs(clf.mlp.coefs_[0]).mean(axis=1)
    
    n_classes = len(clf.classes_)
    n_cols = min(3, n_classes)
    cols = st.columns(n_cols)
    
    for i, cls in enumerate(clf.classes_):
        top_idx = np.argsort(weights)[-10:][::-1]
        top_words = feature_names[top_idx]
        top_vals = weights[top_idx]
        
        with cols[i % n_cols]:
            fig = go.Figure(data=[
                go.Bar(
                    x=top_vals[::-1],
                    y=top_words[::-1],
                    orientation='h',
                    marker_color=SEVERITY_COLOUR.get(cls, C['accent']),
                    text=[f'{v:.4f}' for v in top_vals[::-1]],
                    textposition='outside',
                )
            ])
            fig.update_layout(
                title=cls,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title='Avg Weight',
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


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
                MLP Neural Network · TF-IDF Text Features · Incident Categories
                &nbsp;|&nbsp; Aligned to Malaysia NAIO 2026–2030
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('### ⚙️ Classifier Settings')
        
        show_advanced = st.checkbox('Show Advanced Analytics', value=False,
                                    help='Display training curve, confusion matrix, and per-class metrics')
        
        retrain = st.button('🔄 Retrain Model', use_container_width=True,
                            help='Retrain the MLP on current Supabase data')
        st.markdown('---')
        st.markdown(
            "<small style='color:#6B7280;'>Model trains automatically on first load.<br>"
            'Click Retrain after new incidents are added.</small>',
            unsafe_allow_html=True,
        )

    with st.spinner('Loading incident data from Supabase...'):
        # Fetch from cyber_news table
        df = get_data_fn('cyber_news')
        
        if df is None or df.empty:
            st.error('⚠️ No data available in cyber_news table. Check your Supabase connection.')
            return

    cache_key = f'clf_{len(df)}_{hash(str(df.columns))}'
    
    if retrain or cache_key not in st.session_state:
        with st.spinner('🧠 Training MLP classifier on cyber_news data...'):
            clf = IncidentClassifier()
            success = clf.train(df)
            
            if success:
                st.session_state[cache_key] = clf
                st.session_state['clf_current'] = clf
                st.success('✅ Model trained successfully!')
            else:
                st.error('❌ Model training failed. Check your data has title, summary, and incident_type columns.')
                return
    else:
        clf = st.session_state.get('clf_current') or st.session_state[cache_key]

    if not clf.is_ready():
        st.error('Model could not be trained. Check that incident_type labels exist in the data.')
        return

    _render_live_classifier(clf)
    
    if show_advanced:
        _render_eval_card(clf)
        _render_training_curve(clf)
        _render_confusion_matrix(clf)
        _render_per_class(clf)
        _render_top_features(clf)
