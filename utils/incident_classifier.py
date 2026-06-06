"""
utils/incident_classifier.py
────────────────────────────────────────────────────────────────────────────────
MLP Incident Type Classifier
──────────────────────────────
Architecture: TF-IDF Vectoriser → Multi-Layer Perceptron (256 → 128 → 64 → N classes)

Trained on combined global_news + incidents data.
Predicts incident_type (Malware, Ransomware, Phishing, Data Breach, etc.)
from incident title + summary text.

Public API
──────────
  IncidentClassifier.train(df)         → trained instance
  IncidentClassifier.predict(texts)    → list of {label, confidence, top3}
  IncidentClassifier.predict_one(text) → {label, confidence, top3}
  IncidentClassifier.is_ready()        → bool
  IncidentClassifier.eval_metrics()    → dict of evaluation metrics
  IncidentClassifier.model_card()      → dict
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
)

# ── Constants ─────────────────────────────────────────────────────────────────

# Map all original incident types into 4 top-level categories
LABEL_MAP = {
    # ── Intrusion System ──────────────────────────────────────────────────────
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

    # ── Data Breach ───────────────────────────────────────────────────────────
    'Data Breach':                      'Data Breach',
    'Insider Threat':                   'Data Breach',

    # ── Compromise of Credentials ─────────────────────────────────────────────
    'Phishing':                         'Compromise of Credentials',
    'Social Engineering':               'Compromise of Credentials',
    'Credential Stuffing':              'Compromise of Credentials',

    # ── Fraud ─────────────────────────────────────────────────────────────────
    'Fraud':                            'Fraud',
    'Financial Fraud':                  'Fraud',
    'Cybersecurity':                    'Fraud',
    'Multiple':                         'Fraud',
    'Others':                           'Fraud',
}
MIN_CLASS_SAMPLES = 4
TEST_SIZE         = 0.20
RANDOM_STATE      = 42


class IncidentClassifier:

    def __init__(self):
        self.vectoriser = TfidfVectorizer(
            max_features  = 3000,
            ngram_range   = (1, 2),
            sublinear_tf  = True,
            strip_accents = 'unicode',
            stop_words    = 'english',
        )
        self.model = MLPClassifier(
            hidden_layer_sizes  = (256, 128, 64),
            activation          = 'relu',
            solver              = 'adam',
            alpha               = 0.001,
            max_iter            = 500,
            random_state        = RANDOM_STATE,
            early_stopping      = True,
            validation_fraction = 0.15,
            n_iter_no_change    = 15,
            verbose             = False,
        )
        self.label_enc    = LabelEncoder()
        self._trained     = False
        self._metrics: Dict  = {}
        self._classes: List  = []
        self._conf_matrix    = None
        self._report_dict    = {}
        self.n_train         = 0
        self.n_test          = 0
        self.n_classes       = 0
        self.n_epochs        = 0
        self.loss_curve: List[float] = []

    # ── Data prep ─────────────────────────────────────────────────────────────

    @staticmethod
    def _prepare(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        d = df.copy()
        d['text'] = (
            d.get('title',   pd.Series(dtype=str)).fillna('') + ' ' +
            d.get('summary', pd.Series(dtype=str)).fillna('')
        ).str.strip()

        if 'incident_type' not in d.columns:
            return None

        d = d.dropna(subset=['incident_type'])
        d = d[d['incident_type'].str.strip() != '']
        d['incident_type'] = d['incident_type'].replace(LABEL_MAP)

        counts = d['incident_type'].value_counts()
        valid  = counts[counts >= MIN_CLASS_SAMPLES].index
        d = d[d['incident_type'].isin(valid)].reset_index(drop=True)

        return d if len(d) >= 20 else None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> "IncidentClassifier":
        d = self._prepare(df)
        if d is None:
            self._trained = False
            return self

        self.label_enc.fit(d['incident_type'].unique())
        self._classes = list(self.label_enc.classes_)
        y = self.label_enc.transform(d['incident_type'])

        X_tr_txt, X_te_txt, y_tr, y_te = train_test_split(
            d['text'], y,
            test_size    = TEST_SIZE,
            random_state = RANDOM_STATE,
            stratify     = y,
        )

        self.n_train = len(X_tr_txt)
        self.n_test  = len(X_te_txt)

        X_tr = self.vectoriser.fit_transform(X_tr_txt)
        X_te = self.vectoriser.transform(X_te_txt)

        self.model.fit(X_tr, y_tr)

        self._trained  = True
        self.n_classes = len(self._classes)
        self.n_epochs  = self.model.n_iter_
        self.loss_curve = list(self.model.loss_curve_)

        # ── Evaluation on test set ────────────────────────────────────────────
        y_pred = self.model.predict(X_te)

        self._metrics = {
            'accuracy':          round(float(accuracy_score(y_te, y_pred)), 4),
            'precision_weighted': round(float(precision_score(y_te, y_pred, average='weighted', zero_division=0)), 4),
            'recall_weighted':    round(float(recall_score(y_te, y_pred, average='weighted', zero_division=0)), 4),
            'f1_weighted':        round(float(f1_score(y_te, y_pred, average='weighted', zero_division=0)), 4),
            'f1_macro':           round(float(f1_score(y_te, y_pred, average='macro', zero_division=0)), 4),
        }
        self._conf_matrix  = confusion_matrix(y_te, y_pred)
        self._report_dict  = classification_report(
            y_te, y_pred,
            target_names = self.label_enc.classes_,
            output_dict  = True,
            zero_division = 0,
        )
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, texts: List[str]) -> List[Dict]:
        if not self._trained:
            return [{'label': 'Unknown', 'confidence': 0.0, 'top3': []} for _ in texts]

        X     = self.vectoriser.transform(texts)
        preds = self.model.predict(X)
        probas = self.model.predict_proba(X)

        results = []
        for pred, proba in zip(preds, probas):
            label = self.label_enc.inverse_transform([pred])[0]
            top3  = sorted(
                zip(self._classes, proba.tolist()),
                key=lambda x: -x[1]
            )[:3]
            results.append({
                'label':      label,
                'confidence': round(float(proba.max()), 4),
                'top3':       [(c, round(p, 4)) for c, p in top3],
            })
        return results

    def predict_one(self, text: str) -> Dict:
        return self.predict([text])[0]

    def is_ready(self) -> bool:
        return self._trained

    def eval_metrics(self) -> Dict:
        return self._metrics.copy()

    def model_card(self) -> Dict:
        return {
            'architecture':    'TF-IDF → MLP (256, 128, 64)',
            'activation':      'ReLU',
            'optimiser':       'Adam',
            'regularisation':  'L2 (α=0.001) + Early Stopping',
            'vocab_size':      len(self.vectoriser.vocabulary_) if self._trained else 0,
            'n_classes':       self.n_classes,
            'train_samples':   self.n_train,
            'test_samples':    self.n_test,
            'epochs_run':      self.n_epochs,
            'status':          'Trained ✅' if self._trained else 'Not trained ❌',
            **self._metrics,
        }

    def classes(self) -> List[str]:
        return self._classes

    def confusion_matrix(self):
        return self._conf_matrix

    def per_class_report(self) -> pd.DataFrame:
        if not self._report_dict:
            return pd.DataFrame()
        rows = []
        for cls in self._classes:
            if cls in self._report_dict:
                r = self._report_dict[cls]
                rows.append({
                    'Class':     cls,
                    'Precision': round(r['precision'], 3),
                    'Recall':    round(r['recall'], 3),
                    'F1 Score':  round(r['f1-score'], 3),
                    'Support':   int(r['support']),
                })
        return pd.DataFrame(rows).sort_values('F1 Score', ascending=False)
