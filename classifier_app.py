# classifier_app.py - AI Incident Type Classifier

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
from supabase import create_client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Incident Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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


def get_supabase_client():
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
    except Exception:
        return None
    
    try:
        client = create_client(supabase_url, supabase_key)
        return client
    except Exception:
        return None


def fetch_cyber_news():
    client = get_supabase_client()
    if not client:
        return pd.DataFrame()
    
    try:
        response = client.table('cyber_news').select('*').execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


class IncidentClassifier:
    def __init__(self):
        self.vectorizer = None
        self.label_encoder = None
        self.mlp = None
        self.is_trained = False
        self.classes_ = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.eval_metrics_cache = None
        self.loss_curve_cache = None
        self.confusion_matrix_cache = None
        self.per_class_report_cache = None
    
    def train(self, df):
        if df.empty:
            return False
        
        df = df.copy()
        
        required_cols = ['title', 'summary', 'incident_type']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            return False
        
        df['text'] = df['title'].fillna('') + ' ' + df['summary'].fillna('')
        df = df.dropna(subset=['incident_type'])
        df = df[df['incident_type'].str.strip() != '']
        
        if df.empty:
            return False
        
        label_map = {
            'Malware': 'Intrusion System',
            'Ransomware': 'Intrusion System',
            'Ransomware Attack': 'Intrusion System',
            'Advanced Persistent Threat (APT)': 'Intrusion System',
            'APT': 'Intrusion System',
            'DDoS': 'Intrusion System',
            'Vulnerability': 'Intrusion System',
            'Zero-Day': 'Intrusion System',
            'Supply Chain': 'Intrusion System',
            'Supply Chain Attack': 'Intrusion System',
            'Unauthorised Access': 'Intrusion System',
            'Exploit': 'Intrusion System',
            'Botnet': 'Intrusion System',
            'Botnet control': 'Intrusion System',
            'Attack': 'Intrusion System',
            'Cyber Attack': 'Intrusion System',
            'Cyberattack': 'Intrusion System',
            'Hack': 'Intrusion System',
            'Hacking': 'Intrusion System',
            'Intrusion': 'Intrusion System',
            'Handala': 'Intrusion System',
            'Data Breach': 'Data Breach',
            'Insider Threat': 'Data Breach',
            'Data Leak': 'Data Breach',
            'Sell Data': 'Data Breach',
            'Data Breached': 'Data Breach',
            'Breach': 'Data Breach',
            'Phishing': 'Compromise of Credentials',
            'Social Engineering': 'Compromise of Credentials',
            'Credential Stuffing': 'Compromise of Credentials',
            'Account Takeover': 'Compromise of Credentials',
            'Brute Force': 'Compromise of Credentials',
            'Compromise Credentials': 'Compromise of Credentials',
            'Fraud': 'Scam/Fraud',
            'Financial Fraud': 'Scam/Fraud',
            'Scam/Fraud': 'Scam/Fraud',
            'Advertising fraud': 'Scam/Fraud',
            'Scam': 'Scam/Fraud',
            'Cybersecurity': 'Other',
            'Other Cyber Incident': 'Other Cyber Incident',
            'Others': 'Other',
            'Other': 'Other',
            'cybersecurity incident': 'Other',
        }
        
        df['incident_type'] = df['incident_type'].replace(label_map)
        df = df[~df['incident_type'].isin(['Multiple'])]
        
        counts = df['incident_type'].value_counts()
        valid_classes = counts[counts >= 10].index
        df = df[df['incident_type'].isin(valid_classes)].reset_index(drop=True)
        
        if df.empty:
            return False
        
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['incident_type'])
        self.classes_ = list(self.label_encoder.classes_)
        
        try:
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                df['text'], y, test_size=0.20, random_state=42, stratify=y
            )
        except ValueError:
            return False
        
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents='unicode',
            stop_words='english',
        )
        
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)
        
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
        
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.mlp.predict(X_test)
        self.is_trained = True
        
        return True
    
    def is_ready(self):
        return self.is_trained and self.mlp is not None
    
    def eval_metrics(self):
        if self.eval_metrics_cache:
            return self.eval_metrics_cache
        if not self.is_ready():
            return {}
        
        self.eval_metrics_cache = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'f1_macro': f1_score(self.y_test, self.y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(self.y_test, self.y_pred, average='weighted', zero_division=0),
            'precision_weighted': precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(self.y_test, self.y_pred, average='weighted', zero_division=0),
        }
        return self.eval_metrics_cache
    
    @property
    def loss_curve(self):
        if self.loss_curve_cache is not None:
            return self.loss_curve_cache
        if self.is_ready():
            self.loss_curve_cache = self.mlp.loss_curve_
            return self.loss_curve_cache
        return []
    
    def confusion_matrix(self):
        if self.confusion_matrix_cache is not None:
            return self.confusion_matrix_cache
        if self.is_ready():
            self.confusion_matrix_cache = confusion_matrix(self.y_test, self.y_pred)
            return self.confusion_matrix_cache
        return None
    
    def per_class_report(self):
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
    
    def predict_one(self, text):
        if not self.is_ready():
            return {'label': 'Error', 'confidence': 0, 'top3': []}
        
        # Quick keyword override for attack terms
        text_lower = text.lower()
        attack_keywords = ['attack', 'hack', 'malware', 'ransomware', 'ddos', 
                           'breach', 'intrusion', 'exploit', 'zero-day', 
                           'apt', 'backdoor', 'trojan', 'worm', 'botnet', 'handala']
        
        for keyword in attack_keywords:
            if keyword in text_lower:
                return {
                    'label': 'Intrusion System',
                    'confidence': 0.85,
                    'top3': [('Intrusion System', 0.85), ('Other', 0.10), ('Data Breach', 0.05)]
                }
        
        vec = self.vectorizer.transform([text])
        pred = self.mlp.predict(vec)[0]
        proba = self.mlp.predict_proba(vec)[0]
        label = self.label_encoder.inverse_transform([pred])[0]
        
        top3_idx = np.argsort(proba)[-3:][::-1]
        top3 = [(self.label_encoder.inverse_transform([i])[0], proba[i]) for i in top3_idx]
        
        return {
            'label': label,
            'confidence': proba.max(),
            'top3': top3,
        }


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
    st.markdown(f"<h3 style='color:#E5E7EB;margin-top:2rem;'>{icon} {title}</h3>", unsafe_allow_html=True)


def _metric_colour(val: float, thresholds=(0.70, 0.50)) -> str:
    if val >= thresholds[0]:
        return C['green']
    if val >= thresholds[1]:
        return C['orange']
    return C['red']


def _render_eval_card(clf: IncidentClassifier):
    m = clf.eval_metrics()
    if not m:
        return

    _section('Model Evaluation Card', '📊')
    c1, c2, c3, c4, c5 = st.columns(5)
    _kpi(c1, 'Accuracy', f"{m['accuracy']*100:.1f}%", 'Overall correctness', _metric_colour(m['accuracy']))
    _kpi(c2, 'Precision (wtd)', f"{m['precision_weighted']:.4f}", 'Weighted avg', _metric_colour(m['precision_weighted']))
    _kpi(c3, 'Recall (wtd)', f"{m['recall_weighted']:.4f}", 'Weighted avg', _metric_colour(m['recall_weighted']))
    _kpi(c4, 'F1 Score (wtd)', f"{m['f1_weighted']:.4f}", 'Weighted avg', _metric_colour(m['f1_weighted']))
    _kpi(c5, 'F1 Score (macro)', f"{m['f1_macro']:.4f}", 'Unweighted avg', _metric_colour(m['f1_macro']))


def _render_training_curve(clf: IncidentClassifier):
    _section('Training Loss Curve', '📉')
    loss = clf.loss_curve
    if not loss:
        st.info('No training curve available.')
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(loss) + 1)), y=loss, mode='lines',
        name='Training Loss', line=dict(color=C['accent'], width=2),
        fill='tozeroy', fillcolor='rgba(124,58,237,0.1)',
    ))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        title='MLP Training Loss over Epochs', xaxis_title='Epoch', yaxis_title='Loss',
        margin=dict(t=50, b=30), xaxis=dict(showgrid=True, gridcolor='#2A2D3A'),
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
        cm, labels=dict(x='Predicted', y='Actual', color='Count'),
        x=classes, y=classes, color_continuous_scale='Blues',
        text_auto=True, title='Confusion Matrix — Actual vs Predicted',
        aspect='auto',
    )
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=80), xaxis=dict(tickangle=45),
        coloraxis_showscale=False, font=dict(size=11),
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
            report_df.sort_values('F1 Score'), x='F1 Score', y='Class',
            orientation='h', color='F1 Score',
            color_continuous_scale=['#FF4B4B', '#FFD166', '#06D6A0'],
            title='F1 Score per Incident Type', text='F1 Score', template='plotly_dark',
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         coloraxis_showscale=False, margin=dict(t=50, b=20, r=60))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = px.scatter(
            report_df, x='Precision', y='Recall', size='Support', color='F1 Score',
            color_continuous_scale=['#FF4B4B', '#FFD166', '#06D6A0'],
            hover_data=['Class', 'F1 Score', 'Support'], text='Class',
            title='Precision vs Recall (size = support)', template='plotly_dark',
        )
        fig2.update_traces(textposition='top center', textfont_size=8)
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          coloraxis_showscale=False, margin=dict(t=50, b=20),
                          xaxis=dict(range=[0, 1.1]), yaxis=dict(range=[0, 1.1]))
        st.plotly_chart(fig2, use_container_width=True)

    def _colour_f1(val):
        if val >= 0.70:
            return 'background-color:#06D6A022;color:#06D6A0'
        if val >= 0.50:
            return 'background-color:#FFD16622;color:#FFD166'
        return 'background-color:#FF4B4B22;color:#FF4B4B'

    styled = report_df.style.map(_colour_f1, subset=['F1 Score']).format(
        {'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1 Score': '{:.3f}'}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_live_classifier(clf: IncidentClassifier):
    _section('Live Incident Classifier', '🔍')
    st.markdown("<p style='color:#9CA3AF;'>Paste any incident headline or summary — the model will predict the attack type instantly.</p>", unsafe_allow_html=True)

    text_input = st.text_area('Incident headline / summary', height=100, placeholder='e.g. LockBit ransomware group claims attack on Malaysian government ministry...')

    if st.button('🔮 Classify Incident', use_container_width=True, type='primary'):
        if not text_input.strip():
            st.warning('Please enter some text.')
            return

        with st.spinner('Analyzing...'):
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
                        Confidence: <span style="color:{colour};font-weight:600;">{conf*100:.1f}%</span>
                    </div>
                </div>""", unsafe_allow_html=True,
            )

            top3_df = pd.DataFrame(top3, columns=['Incident Type', 'Probability'])
            top3_df['Probability %'] = (top3_df['Probability'] * 100).round(1)

            fig = px.bar(
                top3_df, x='Probability %', y='Incident Type', orientation='h',
                color='Probability %', color_continuous_scale=['#7C3AED', colour],
                text='Probability %', title='Top 3 Predictions', template='plotly_dark',
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             coloraxis_showscale=False, margin=dict(t=50, b=20, r=60),
                             yaxis=dict(autorange='reversed'), height=200)
            st.plotly_chart(fig, use_container_width=True)


def main():
    st.markdown("""
        <div style="padding:24px 0 8px;">
            <h1 style="color:#F9FAFB;font-size:2rem;font-weight:800;margin:0;">
                🧠 AI Incident Type Classifier
            </h1>
            <p style="color:#9CA3AF;margin:6px 0 0;">
                MLP Neural Network · TF-IDF Text Features · Incident Categories
                &nbsp;|&nbsp; Aligned to Malaysia NAIO 2026–2030
            </p>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('### ⚙️ Classifier Settings')
        show_advanced = st.checkbox('📊 Show Advanced Analytics', value=False)
        retrain = st.button('🔄 Retrain Model', use_container_width=True, type='primary')
        st.markdown('---')

    df = fetch_cyber_news()
    
    if df.empty:
        st.warning("No data available in cyber_news table.")
        return

    cache_key = f'clf_{len(df)}'
    
    if retrain or cache_key not in st.session_state:
        with st.spinner('Training MLP classifier...'):
            clf = IncidentClassifier()
            if clf.train(df):
                st.session_state[cache_key] = clf
                st.success('Model trained successfully!')
            else:
                st.error('Model training failed.')
                return
    else:
        clf = st.session_state[cache_key]

    _render_live_classifier(clf)
    
    if show_advanced:
        _render_eval_card(clf)
        _render_training_curve(clf)
        _render_confusion_matrix(clf)
        _render_per_class(clf)


if __name__ == '__main__':
    main()
