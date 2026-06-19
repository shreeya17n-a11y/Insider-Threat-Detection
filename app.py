import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ThreatScope · Insider Threat Detection",
    layout="wide",
    page_icon="🛡",
    initial_sidebar_state="collapsed",
)

# Initialize navigation session state
if "page" not in st.session_state:
    st.session_state.page = "overview"

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE SYSTEM (PREMIUM BLACK & WHITE THEME)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,900;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html, body, .stApp {
    background: #ffffff !important;
    color: #18181b !important;
    font-family: 'Inter', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* Hide default Streamlit clutter */
section[data-testid="stSidebar"] {display: none !important}
header[data-testid="stHeader"] {display: none !important}
footer {display: none !important}
div[data-testid="stDecoration"] {display: none !important}

/* Center-align the core block layout */
.block-container {
    max-width: 1100px !important;
    padding: 40px 24px 100px !important;
    margin: 0 auto !important;
}

p, span, label, div, h1, h2, h3, h4, li, small, td, th {
    color: inherit !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #f4f4f5;
}
::-webkit-scrollbar-thumb {
    background: #d4d4d8;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #a1a1aa;
}

/* TOPBAR */
.topbar {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 12px;
    padding: 0 24px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
}
.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-icon {
    font-size: 20px;
}
.brand-name {
    font-size: 15px;
    font-weight: 700;
    color: #09090b !important;
    letter-spacing: -0.3px;
}
.brand-badge {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    background: #f4f4f5;
    color: #71717a !important;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid #e4e4e7;
}
.topbar-status {
    display: flex;
    align-items: center;
    gap: 8px;
}
.status-dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
}
.status-text {
    font-size: 12px;
    color: #71717a !important;
    font-weight: 500;
}

/* TAB NAVIGATION */
div[data-nav-active] {
    display: flex;
    justify-content: center;
    width: 100%;
}
div[data-nav-active] button {
    background: transparent !important;
    color: #71717a !important;
    border: none !important;
    border-radius: 0px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 12px 0 !important;
    width: 100% !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
div[data-nav-active] button:hover {
    color: #09090b !important;
    border-bottom: 2px solid #e4e4e7 !important;
}
div[data-nav-active="true"] button {
    color: #09090b !important;
    border-bottom: 2px solid #09090b !important;
    font-weight: 600 !important;
}

/* Primary buttons */
div[data-btn-primary="true"] > div.stButton > button {
    background: #09090b !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    padding: 12px 0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border: none !important;
    width: 100% !important;
    transition: background 0.15s ease !important;
}
div[data-btn-primary="true"] > div.stButton > button:hover {
    background: #27272a !important;
}
div[data-btn-primary="true"] > div.stButton > button p {
    color: #ffffff !important;
}

/* HERO SECTION */
.pg-hero {
    padding: 64px 0 52px;
    border-bottom: 1px solid #e4e4e7;
    margin-bottom: 56px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.pg-eyebrow {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #a1a1aa !important;
    margin-bottom: 16px;
}
.pg-title {
    font-size: clamp(36px, 4vw, 56px);
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1.1;
    color: #09090b !important;
    margin-bottom: 16px;
}
.pg-title em {
    color: #71717a !important;
    font-style: normal;
}
.pg-sub {
    font-size: 15px;
    color: #71717a !important;
    line-height: 1.7;
    max-width: 600px;
    margin: 0 auto;
}

/* STAT ROW */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #e4e4e7;
    border: 1px solid #e4e4e7;
    border-radius: 12px;
    overflow: hidden;
    margin-top: 40px;
    width: 100%;
}
.stat-cell {
    background: #ffffff;
    padding: 24px 28px;
    text-align: center;
}
.stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #09090b !important;
    line-height: 1;
    margin-bottom: 8px;
}
.stat-lbl {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #a1a1aa !important;
}

/* SECTION HEADINGS */
.sec-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #a1a1aa !important;
    margin-bottom: 8px;
}
.sec-title {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #09090b !important;
    margin-bottom: 10px;
}
.sec-desc {
    font-size: 13px;
    color: #71717a !important;
    line-height: 1.7;
    max-width: 580px;
    margin-bottom: 32px;
}

/* PANELS & CARDS */
.panel {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 12px;
    overflow: hidden;
}
.panel-head {
    background: #fafafa;
    border-bottom: 1px solid #e4e4e7;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.panel-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #09090b;
}
.panel-ttl {
    font-size: 12px;
    font-weight: 600;
    color: #09090b !important;
}
.panel-body {
    padding: 24px;
}

.card {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 12px;
    padding: 24px;
    transition: all 0.2s ease;
}
.card:hover {
    border-color: #09090b;
}
.card-lbl {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #a1a1aa !important;
    margin-bottom: 8px;
}
.card-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #09090b !important;
    line-height: 1;
    margin-bottom: 6px;
}
.card-sub {
    font-size: 11px;
    color: #71717a !important;
    line-height: 1.5;
}

/* METRIC GRID CARDS */
.m-card {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 12px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
}
.m-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: #e4e4e7;
}
.m-card.hi::before {
    background: #09090b;
}
.m-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 30px;
    font-weight: 700;
    color: #09090b !important;
    line-height: 1;
    margin-bottom: 8px;
}
.m-lbl {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #a1a1aa !important;
    margin-bottom: 5px;
}
.m-desc {
    font-size: 11px;
    color: #71717a !important;
    line-height: 1.4;
}

/* PARAM ROW */
.p-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 0;
    border-bottom: 1px solid #f4f4f5;
}
.p-row:last-child {
    border-bottom: none;
}
.p-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #71717a !important;
    min-width: 160px;
}
.p-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: #09090b !important;
    background: #f4f4f5;
    border: 1px solid #e4e4e7;
    padding: 4px 12px;
    border-radius: 4px;
}
.p-desc {
    font-size: 11px;
    color: #a1a1aa !important;
    text-align: right;
    max-width: 220px;
}

/* CONFUSION MATRIX */
.cm-outer {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e4e4e7;
}
.cm-table {
    width: 100%;
    border-collapse: collapse;
}
.cm-head {
    background: #fafafa;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #71717a !important;
    padding: 12px 18px;
    text-align: center;
}
.cm-cell {
    padding: 24px 16px;
    text-align: center;
    border: 1px solid #f4f4f5;
}
.cm-n {
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 5px;
}
.cm-t {
    font-size: 9px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #a1a1aa !important;
}
.tp-c {background: #09090b}
.tp-c .cm-n {color: #ffffff !important}
.tp-c .cm-t {color: #a1a1aa !important}
.tn-c {background: #ffffff}
.tn-c .cm-n {color: #09090b !important}
.fp-c {background: #fafafa}
.fp-c .cm-n {color: #a1a1aa !important}
.fn-c {background: #fafafa}
.fn-c .cm-n {color: #a1a1aa !important}

/* FEATURE IMPORTANCE BARS */
.fi-row {
    margin-bottom: 16px;
}
.fi-hd {
    display: flex;
    justify-content: space-between;
    margin-bottom: 7px;
}
.fi-name {
    font-size: 12px;
    color: #71717a !important;
    font-weight: 500;
}
.fi-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #09090b !important;
    font-weight: 600;
}
.fi-track {
    background: #e4e4e7;
    border-radius: 2px;
    height: 4px;
}
.fi-fill {
    height: 4px;
    background: #09090b;
    border-radius: 2px;
}

/* COMPARISON TABLE */
.ct {
    width: 100%;
    border-collapse: collapse;
}
.ct th {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #71717a !important;
    padding: 12px 18px;
    border-bottom: 1px solid #e4e4e7;
    text-align: left;
    background: #fafafa;
}
.ct td {
    padding: 14px 18px;
    font-size: 12px;
    color: #71717a !important;
    border-bottom: 1px solid #f4f4f5;
}
.ct tr.rf-row td {
    color: #ffffff !important;
    background: #09090b;
}
.ct tr.rf-row td:first-child {
    font-weight: 700;
}
.ct td.best {
    color: #09090b !important;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
}
.ct tr.rf-row td.best {
    color: #ffffff !important;
}
.ct td.norm {
    font-family: 'JetBrains Mono', monospace;
}

/* WHY CARDS */
.why-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin-bottom: 20px;
}
.why-card {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 12px;
    padding: 22px;
}
.why-ico {
    font-size: 18px;
    margin-bottom: 12px;
}
.why-ttl {
    font-size: 13px;
    font-weight: 700;
    color: #09090b !important;
    margin-bottom: 8px;
}
.why-dsc {
    font-size: 11px;
    color: #71717a !important;
    line-height: 1.7;
}

/* BADGES & TUNING */
.gs-badge {
    display: inline-block;
    background: #f4f4f5;
    border: 1px solid #e4e4e7;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #09090b !important;
    padding: 4px 12px;
    border-radius: 4px;
    margin: 3px 3px 0 0;
}
.improvement-up {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    color: #09090b !important;
}
.improvement-same {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: #a1a1aa !important;
}

/* VERDICT CARD */
.v-threat {
    background: #09090b;
    border-radius: 12px;
    padding: 26px 28px;
}
.v-safe {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 12px;
    padding: 26px 28px;
}
.v-threat .v-ico, .v-threat .v-ttl, .v-threat .v-via {
    color: #ffffff !important;
}
.v-safe .v-ico, .v-safe .v-ttl {
    color: #09090b !important;
}
.v-safe .v-via {
    color: #71717a !important;
}
.v-ico {
    font-size: 24px;
    margin-bottom: 10px;
}
.v-ttl {
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 4px;
}
.v-via {
    font-size: 10px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.risk-wrap {
    margin-top: 22px;
}
.risk-hd {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 8px;
}
.risk-lbl {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #a1a1aa !important;
}
.risk-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #09090b !important;
}
.risk-track {
    background: #e4e4e7;
    border-radius: 2px;
    height: 4px;
}
.risk-fill {
    height: 4px;
    background: #09090b;
    border-radius: 2px;
    transition: width 0.5s;
}

.sig-tag {
    display: inline-block;
    background: #f4f4f5;
    border: 1px solid #e4e4e7;
    color: #71717a !important;
    font-size: 10px;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    margin: 3px 3px 0 0;
}

/* WIDGET OVERRIDES */
.stNumberInput input {
    background: #ffffff !important;
    color: #09090b !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
}
.stNumberInput input:focus {
    border-color: #09090b !important;
    box-shadow: none !important;
    outline: none !important;
}
.stNumberInput label, .stSlider label, .stSelectbox label, .stMultiSelect label {
    font-size: 10px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    color: #71717a !important;
}
.stSelectbox [data-baseweb="select"] {
    background: #ffffff !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 8px !important;
}
.stSelectbox [data-baseweb="select"] * {
    color: #09090b !important;
    background: #ffffff !important;
    font-size: 13px !important;
}
.stMultiSelect [data-baseweb="select"] {
    background: #ffffff !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 8px !important;
}
.stMultiSelect [data-baseweb="select"] * {
    color: #09090b !important;
    background: #ffffff !important;
}
[data-baseweb="tag"] {
    background: #09090b !important;
    color: #ffffff !important;
    border-radius: 4px !important;
}
[data-baseweb="slider"] [role="slider"] {
    background: #09090b !important;
    border-color: #09090b !important;
}
[data-baseweb="slider"]>div>div {
    background: #09090b !important;
}
.stSpinner>div {
    border-top-color: #09090b !important;
}
.stProgress>div>div {
    background-color: #09090b !important;
}

.stCheckbox label {
    font-size: 12px !important;
    color: #71717a !important;
}
[data-baseweb="checkbox"] [data-checked="true"] {
    background: #09090b !important;
    border-color: #09090b !important;
}

.div-line {
    border: none;
    border-top: 1px solid #e4e4e7;
    margin: 48px 0;
}

.info-box {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-left: 3px solid #09090b;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    font-size: 12px;
    color: #71717a !important;
    line-height: 1.8;
    margin-bottom: 20px;
}
.info-box strong {
    color: #09090b !important;
}

[data-testid="stArrowVegaLiteChart"] canvas {
    filter: none !important;
}

/* FOOTER */
.pg-footer {
    border-top: 1px solid #e4e4e7;
    padding-top: 24px;
    margin-top: 80px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.pg-footer-l {
    font-size: 11px;
    color: #a1a1aa !important;
    line-height: 1.6;
}
.pg-footer-r {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #a1a1aa !important;
}
</style>
""", unsafe_allow_html=True)


# ── DATA PROCESSING FUNCTIONS ─────────────────────────────────────────────────

@st.cache_data
def load_data():
    data = pd.read_csv("reduced_data2.csv")
    data = data.rename(columns={'from': 'sender'})
    data['date']        = pd.to_datetime(data['date'])
    data['day_of_week'] = data['date'].dt.dayofweek
    data['email_count'] = data.groupby('sender')['sender'].transform('count')
    data['attachments'] = data['attachments'].fillna(0)
    data['size']        = data['size'].fillna(0)
    data['insider_threat'] = (
        (data['email_count'] > 30) |
        (data['attachments'] > 5)  |
        (data['day_of_week'] >= 4) |
        (data['size'] > 1_000_000)
    ).astype(int)
    return data


@st.cache_data
def get_splits():
    data     = load_data()
    FEATURES = ['email_count', 'attachments', 'day_of_week', 'size']
    X, y     = data[FEATURES], data['insider_threat']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sc       = StandardScaler()
    X_tr_sc  = sc.fit_transform(X_tr)
    X_te_sc  = sc.transform(X_te)
    return X_tr_sc, X_te_sc, y_tr, y_te, sc, FEATURES


@st.cache_data
def get_comparison(tuned_params=None):
    X_tr, X_te, y_tr, y_te, sc, _ = get_splits()
    
    if tuned_params is not None:
        depth = tuned_params.get("max_depth")
        rf_model = RandomForestClassifier(
            n_estimators=tuned_params.get("n_estimators", 100),
            max_depth=depth,
            min_samples_split=tuned_params.get("min_samples_split", 2),
            min_samples_leaf=tuned_params.get("min_samples_leaf", 1),
            random_state=42, n_jobs=-1
        )
    else:
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest":       rf_model,
        "SVM":                 SVC(kernel='rbf', probability=True, random_state=42),
    }
    rows = []
    for name, clf in models.items():
        clf.fit(X_tr, y_tr)
        p  = clf.predict(X_te)
        pr = clf.predict_proba(X_te)[:, 1]
        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_te, p) * 100, 1),
            "Precision": round(precision_score(y_te, p, zero_division=0) * 100, 1),
            "Recall":    round(recall_score(y_te, p, zero_division=0) * 100, 1),
            "F1":        round(f1_score(y_te, p, zero_division=0) * 100, 1),
            "AUC-ROC":   round(roc_auc_score(y_te, pr) * 100, 1),
        })
    return pd.DataFrame(rows)


def train_rf(n_est, max_dep, min_split, min_leaf):
    X_tr, X_te, y_tr, y_te, sc, FEATURES = get_splits()
    clf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=max_dep if max_dep != 0 else None,
        min_samples_split=min_split,
        min_samples_leaf=min_leaf,
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    proba = clf.predict_proba(X_te)[:, 1]
    m = {
        "accuracy":  accuracy_score(y_te, preds),
        "precision": precision_score(y_te, preds, zero_division=0),
        "recall":    recall_score(y_te, preds, zero_division=0),
        "f1":        f1_score(y_te, preds, zero_division=0),
        "roc_auc":   roc_auc_score(y_te, proba),
    }
    cm_v = confusion_matrix(y_te, preds).ravel()
    fi   = dict(zip(FEATURES, clf.feature_importances_))
    return clf, sc, m, cm_v, fi


# ── INITIAL STATISTICS & BASE MODEL PREPARATION ──────────────────────────────

df          = load_data()
N_EMAILS    = len(df)
N_USERS     = df['sender'].nunique()
THREAT_RATE = round(df['insider_threat'].mean() * 100, 1)
N_FLAGGED   = df[df['insider_threat'] == 1]['sender'].nunique()

if "base_clf" not in st.session_state:
    _clf, _sc, _m, _cm, _fi = train_rf(100, 6, 2, 1)
    st.session_state.base_clf      = _clf
    st.session_state.base_scaler   = _sc
    st.session_state.base_metrics  = _m
    st.session_state.base_cm       = _cm
    st.session_state.base_fi       = _fi
    st.session_state.tuned_params  = None
    st.session_state.tuned_metrics = None
    st.session_state.tuned_cm      = None
    st.session_state.tuned_fi      = None
    st.session_state.tuned_clf     = None


# ── TOPBAR RENDER ─────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">🛡</div>
    <span class="brand-name">ThreatScope</span>
    <span class="brand-badge">Random Forest</span>
  </div>
  <div class="topbar-status">
    <div class="status-dot"></div>
    <span class="status-text">{N_EMAILS:,} records loaded</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── NAVIGATION TAB BAR (ELEGANT 4-COLUMN BUTTONS) ─────────────────────────────

nav_items = [
    ("overview",   "Overview"),
    ("tuning",     "Hyperparameter Tuning"),
    ("evaluation", "Model Evaluation"),
    ("predict",    "Live Prediction"),
]
nc1, nc2, nc3, nc4 = st.columns(4)

for col, (key, label) in zip([nc1, nc2, nc3, nc4], nav_items):
    with col:
        is_active = st.session_state.page == key
        st.markdown(
            f'<div data-nav-active="{"true" if is_active else "false"}">',
            unsafe_allow_html=True,
        )
        if st.button(label, key=f"nav_{key}"):
            st.session_state.page = key
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div style="border-top:1px solid #e4e4e7;margin-top:-2px;margin-bottom:32px"></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "overview":

    st.markdown(f"""
    <div class="pg-hero">
      <div class="pg-eyebrow">Insider Threat Detection · ML Dashboard</div>
      <div class="pg-title">Detecting Insider Threats<br><em>with Precision.</em></div>
      <div class="pg-sub">
        A Random Forest classifier trained on organizational email behaviour
        to surface anomalous user activity before damage is done.
      </div>
      <div class="stat-grid">
        <div class="stat-cell">
          <div class="stat-num">{N_EMAILS:,}</div>
          <div class="stat-lbl">Emails Analyzed</div>
        </div>
        <div class="stat-cell">
          <div class="stat-num">{N_USERS:,}</div>
          <div class="stat-lbl">Unique Users</div>
        </div>
        <div class="stat-cell">
          <div class="stat-num">{THREAT_RATE}%</div>
          <div class="stat-lbl">Flagged Rate</div>
        </div>
        <div class="stat-cell">
          <div class="stat-num">{N_FLAGGED:,}</div>
          <div class="stat-lbl">Flagged Users</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Features used
    st.markdown("""
    <div class="sec-label">Detection Signals</div>
    <div class="sec-title">Behavioural Features</div>
    <div class="sec-desc">Four key signals extracted from raw email metadata power the classifier.</div>
    """, unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns(4)
    features_info = [
        ("📧", "Email Volume", "email_count", "Total emails sent per user. High volume relative to peers is a primary indicator."),
        ("📎", "Attachments", "attachments", "Emails containing attachments. Data exfiltration often involves attachment-heavy behaviour."),
        ("📅", "Day of Week", "day_of_week", "Activity on weekends (Sat–Sun) is anomalous for most organizations."),
        ("📦", "Email Size", "size", "Large emails may indicate bulk data transfer outside normal business communication."),
    ]
    for col, (ico, name, feat, desc) in zip([fc1, fc2, fc3, fc4], features_info):
        with col:
            st.markdown(f"""
            <div class="card">
              <div style="font-size:20px;margin-bottom:14px">{ico}</div>
              <div class="card-lbl">{feat}</div>
              <div style="font-size:14px;font-weight:700;color:#09090b;margin-bottom:10px">{name}</div>
              <div class="card-sub">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="div-line">', unsafe_allow_html=True)

    # Threat labelling logic
    st.markdown("""
    <div class="sec-label">Ground Truth</div>
    <div class="sec-title">Threat Labelling Logic</div>
    <div class="sec-desc">A user is labelled a potential insider threat if any of the following conditions are met:</div>
    """, unsafe_allow_html=True)

    rc1, rc2 = st.columns(2)
    rules = [
        ("Email count > 30", "Volume anomaly — sends significantly more emails than average"),
        ("Attachments > 5",  "Data exfiltration signal — attaches files to many emails"),
        ("Day of week ≥ 4",  "Weekend activity — operates outside normal business hours"),
        ("Email size > 1MB", "Bulk transfer signal — sends unusually large emails"),
    ]
    for col, (r, d) in zip([rc1, rc1, rc2, rc2], rules):
        with col:
            st.markdown(f"""
            <div class="info-box"><strong>{r}</strong><br>{d}</div>
            """, unsafe_allow_html=True)

    # Dataset distribution
    st.markdown('<hr class="div-line">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-label">Data Distribution</div>
    <div class="sec-title">Class Balance</div>
    """, unsafe_allow_html=True)

    dist_c1, dist_c2 = st.columns([1, 1.8])
    with dist_c1:
        vc = df['insider_threat'].value_counts()
        normal  = vc.get(0, 0)
        flagged = vc.get(1, 0)
        st.markdown(f"""
        <div class="card" style="margin-bottom:10px">
          <div class="card-lbl">Normal Records</div>
          <div class="card-val">{normal:,}</div>
          <div class="card-sub">{round(normal/len(df)*100,1)}% of dataset</div>
        </div>
        <div class="card">
          <div class="card-lbl">Threat Records</div>
          <div class="card-val">{flagged:,}</div>
          <div class="card-sub">{round(flagged/len(df)*100,1)}% of dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with dist_c2:
        chart_df = df['insider_threat'].value_counts().rename(index={0: 'Normal', 1: 'Flagged Threat'})
        st.bar_chart(chart_df, color="#09090b", height=220)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: HYPERPARAMETER TUNING
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "tuning":

    st.markdown("""
    <div class="pg-hero" style="padding-bottom:40px">
      <div class="pg-eyebrow">Random Forest · Grid Search CV</div>
      <div class="pg-title">Hyperparameter<br><em>Tuning</em></div>
      <div class="pg-sub">
        Define the search space for each parameter. The system will automatically
        run 5-fold cross-validated Grid Search to find the optimal combination
        that maximises your chosen metric.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <strong>What is Hyperparameter Tuning?</strong><br>
      Every ML model has internal settings (hyperparameters) that control how it learns.
      Rather than guessing, <strong>Grid Search</strong> systematically tries every combination
      of values you specify and selects whichever configuration produces the highest cross-validation score.
      Cross-validation ensures results generalize — the model is tested on data it hasn't seen during each fold.
    </div>
    """, unsafe_allow_html=True)

    # Search space config
    st.markdown("""
    <div class="sec-label">Step 1</div>
    <div class="sec-title">Define Search Space</div>
    <div class="sec-desc">
        Select which values to include in the grid for each parameter.
        Grid Search will try every combination.
    </div>
    """, unsafe_allow_html=True)

    sp_c1, sp_gap, sp_c2 = st.columns([1, 0.05, 1])

    with sp_c1:
        st.markdown('<div class="panel"><div class="panel-head"><div class="panel-dot"></div><span class="panel-ttl">Tree Structure Parameters</span></div><div class="panel-body">', unsafe_allow_html=True)

        n_est_opts = st.multiselect(
            "n_estimators  (number of trees)",
            options=[10, 25, 50, 75, 100, 150, 200],
            default=[50, 100, 200],
            help="More trees = more stable but slower."
        )
        max_dep_opts = st.multiselect(
            "max_depth  (0 = unlimited)",
            options=[0, 3, 4, 5, 6, 8, 10],
            default=[4, 6, 8],
            help="Controls how deep each tree grows."
        )
        st.markdown("</div></div>", unsafe_allow_html=True)

    with sp_c2:
        st.markdown('<div class="panel"><div class="panel-head"><div class="panel-dot"></div><span class="panel-ttl">Leaf & Split Parameters</span></div><div class="panel-body">', unsafe_allow_html=True)

        min_spl_opts = st.multiselect(
            "min_samples_split",
            options=[2, 5, 10, 20],
            default=[2, 5, 10],
            help="Min samples required to split an internal node."
        )
        min_leaf_opts = st.multiselect(
            "min_samples_leaf",
            options=[1, 2, 4, 8],
            default=[1, 2],
            help="Min samples required at a leaf node."
        )
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Scoring selection
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-label">Step 2</div>
    <div class="sec-title">Choose Optimisation Metric</div>
    """, unsafe_allow_html=True)

    sc_c1, sc_c2, sc_c3 = st.columns([1, 1, 2])
    with sc_c1:
        scoring_choice = st.selectbox(
            "Optimise Grid Search for",
            options=["f1", "recall", "precision", "accuracy", "roc_auc"],
            index=0,
            help="In security, Recall is critical — missing a real threat is worse than a false alarm."
        )
    with sc_c2:
        cv_folds = st.selectbox("Cross-validation folds", options=[3, 5, 10], index=1)

    # Run button & combo details
    st.markdown("<br>", unsafe_allow_html=True)
    n_combinations = (
        max(1, len(n_est_opts)) *
        max(1, len(max_dep_opts)) *
        max(1, len(min_spl_opts)) *
        max(1, len(min_leaf_opts))
    )
    st.markdown(f"""
    <div style="font-size:11px;color:#71717a;margin-bottom:16px;font-family:'JetBrains Mono',monospace">
      Grid combinations: {n_combinations} &nbsp;·&nbsp;
      CV folds: {cv_folds} &nbsp;·&nbsp;
      Total model fits: {n_combinations * cv_folds}
    </div>
    """, unsafe_allow_html=True)

    run_col, _ = st.columns([1, 3])
    with run_col:
        st.markdown('<div data-btn-primary="true">', unsafe_allow_html=True)
        run_tune = st.button("Run Grid Search", key="run_tune")
        st.markdown("</div>", unsafe_allow_html=True)

    # Execute tuning
    if run_tune:
        if not n_est_opts or not max_dep_opts or not min_spl_opts or not min_leaf_opts:
            st.error("Please select at least one value for every parameter.")
        else:
            max_dep_grid = [None if v == 0 else v for v in max_dep_opts]

            param_grid = {
                "n_estimators":      n_est_opts,
                "max_depth":         max_dep_grid,
                "min_samples_split": min_spl_opts,
                "min_samples_leaf":  min_leaf_opts,
            }

            X_tr, X_te, y_tr, y_te, sc_fit, FEATS = get_splits()

            with st.spinner(f"Running Grid Search — {n_combinations * cv_folds} fits in progress…"):
                gs = GridSearchCV(
                    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
                    param_grid=param_grid,
                    scoring=scoring_choice,
                    cv=cv_folds,
                    n_jobs=-1,
                    refit=True,
                    verbose=0,
                )
                gs.fit(X_tr, y_tr)

            best_clf  = gs.best_estimator_
            best_p    = gs.best_params_
            cv_score  = gs.best_score_

            # Evaluate tuned model
            t_preds = best_clf.predict(X_te)
            t_proba = best_clf.predict_proba(X_te)[:, 1]
            t_m = {
                "accuracy":  accuracy_score(y_te, t_preds),
                "precision": precision_score(y_te, t_preds, zero_division=0),
                "recall":    recall_score(y_te, t_preds, zero_division=0),
                "f1":        f1_score(y_te, t_preds, zero_division=0),
                "roc_auc":   roc_auc_score(y_te, t_proba),
            }
            t_cm = confusion_matrix(y_te, t_preds).ravel()
            t_fi = dict(zip(FEATS, best_clf.feature_importances_))

            # Save results to session state
            st.session_state.tuned_params  = best_p
            st.session_state.tuned_metrics = t_m
            st.session_state.tuned_cm      = t_cm
            st.session_state.tuned_fi      = t_fi
            st.session_state.tuned_clf     = best_clf
            st.session_state.tuned_cv      = cv_score
            st.session_state.tuned_scoring = scoring_choice

            # Also update base model used for evaluation and predictions
            st.session_state.base_clf     = best_clf
            st.session_state.base_scaler  = sc_fit
            st.session_state.base_metrics = t_m
            st.session_state.base_cm      = t_cm
            st.session_state.base_fi      = t_fi

    # Show tuning results if available
    if st.session_state.tuned_params is not None:
        st.markdown('<hr class="div-line" style="margin:40px 0">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sec-label">Grid Search Results</div>
        <div class="sec-title">Best Configuration Found</div>
        <div class="sec-desc">
          Grid Search completed. Best cross-validation
          <strong style="color:#09090b">{st.session_state.get("tuned_scoring","f1").upper()}</strong>
          score: <strong style="color:#09090b;font-family:'JetBrains Mono',monospace">
          {round(st.session_state.tuned_cv*100,1)}%</strong>
        </div>
        """, unsafe_allow_html=True)

        bp = st.session_state.tuned_params
        tm = st.session_state.tuned_metrics
        bm = st.session_state.base_metrics

        # Parameter explanation mappings
        PARAM_DESC = {
            "n_estimators":      "Number of trees in the forest",
            "max_depth":         "Maximum depth per tree (None = unlimited)",
            "min_samples_split": "Min samples required to split a node",
            "min_samples_leaf":  "Min samples at each leaf node",
        }

        res_c1, res_gap, res_c2 = st.columns([1, 0.05, 1])

        with res_c1:
            st.markdown('<div class="panel"><div class="panel-head"><div class="panel-dot"></div><span class="panel-ttl">Optimal Hyperparameters</span></div><div style="padding:0 24px">', unsafe_allow_html=True)
            for k, v in bp.items():
                disp_v = "None (unlimited)" if v is None else v
                desc   = PARAM_DESC.get(k, "")
                st.markdown(f"""
                <div class="p-row">
                  <span class="p-key">{k}</span>
                  <span class="p-val">{disp_v}</span>
                  <span class="p-desc">{desc}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="card" style="margin-top:10px">
              <div class="card-lbl">Best CV Score ({st.session_state.get("tuned_scoring","f1").upper()})</div>
              <div class="card-val">{round(st.session_state.tuned_cv*100,1)}%</div>
              <div class="card-sub">5-fold cross-validation on training data</div>
            </div>
            """, unsafe_allow_html=True)

        with res_c2:
            st.markdown("""
            <div class="sec-label" style="margin-bottom:16px">Before vs After Tuning (Test Set)</div>
            """, unsafe_allow_html=True)

            def delta_html(tuned_v, base_v):
                d = tuned_v - base_v
                if d > 0.001:
                    return f'<span class="improvement-up">▲ +{round(d*100,1)}%</span>'
                elif d < -0.001:
                    return f'<span style="color:#a1a1aa;font-family:JetBrains Mono,monospace;font-size:12px">▼ -{round(abs(d)*100,1)}%</span>'
                return '<span class="improvement-same">— no change</span>'

            metrics_compare = [
                ("Accuracy",  "accuracy"),
                ("Precision", "precision"),
                ("Recall",    "recall"),
                ("F1 Score",  "f1"),
                ("AUC-ROC",   "roc_auc"),
            ]
            bv_rows = ""
            for label, key in metrics_compare:
                base_v  = bm.get(key, 0)
                tuned_v = tm.get(key, 0)
                bv_rows += f"""
                <tr>
                  <td style="color:#71717a;font-size:12px;padding:12px 0;border-bottom:1px solid #e4e4e7">{label}</td>
                  <td style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#71717a;padding:12px 0;border-bottom:1px solid #e4e4e7">{round(base_v*100,1)}%</td>
                  <td style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#09090b;padding:12px 0;border-bottom:1px solid #e4e4e7;font-weight:600">{round(tuned_v*100,1)}%</td>
                  <td style="padding:12px 0;border-bottom:1px solid #e4e4e7">{delta_html(tuned_v, base_v)}</td>
                </tr>
                """

            st.markdown(f"""
            <table style="width:100%;border-collapse:collapse">
              <thead>
                <tr>
                  <th style="font-size:9px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:#71717a;padding:0 0 10px;text-align:left">Metric</th>
                  <th style="font-size:9px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:#71717a;padding:0 0 10px;text-align:left">Default RF</th>
                  <th style="font-size:9px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:#71717a;padding:0 0 10px;text-align:left">Tuned RF</th>
                  <th style="font-size:9px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:#71717a;padding:0 0 10px;text-align:left">Change</th>
                </tr>
              </thead>
              <tbody>{bv_rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="font-size:10px;color:#a1a1aa;margin-top:14px">
              Default RF: n_estimators=100, max_depth=6, min_samples_split=2, min_samples_leaf=1
            </div>
            """, unsafe_allow_html=True)

        # Feature importance chart
        st.markdown('<hr class="div-line" style="margin:40px 0">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label" style="margin-bottom:16px">Feature Importance (Tuned Model)</div>', unsafe_allow_html=True)

        fi = st.session_state.tuned_fi
        fi_c1, fi_c2 = st.columns(2)
        FEAT_DISPLAY = {
            'email_count': 'Email Volume',
            'attachments': 'Attachments',
            'day_of_week': 'Day of Week',
            'size':        'Email Size',
        }
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        max_fi    = max(v for _, v in sorted_fi)

        for i, (feat, imp) in enumerate(sorted_fi):
            col = fi_c1 if i % 2 == 0 else fi_c2
            with col:
                pct = round(imp * 100, 1)
                bar = round((imp / max_fi) * 100)
                st.markdown(f"""
                <div class="fi-row">
                  <div class="fi-hd">
                    <span class="fi-name">{FEAT_DISPLAY.get(feat, feat)}</span>
                    <span class="fi-pct">{pct}%</span>
                  </div>
                  <div class="fi-track"><div class="fi-fill" style="width:{bar}%"></div></div>
                </div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "evaluation":

    st.markdown("""
    <div class="pg-hero" style="padding-bottom:40px">
      <div class="pg-eyebrow">Random Forest · Performance Analysis</div>
      <div class="pg-title">Model<br><em>Evaluation</em></div>
      <div class="pg-sub">
        Comprehensive performance analysis of the Random Forest classifier
        on held-out test data — and the evidence-backed rationale
        for why it is the best choice for insider threat detection.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Determine evaluated model status
    using_tuned = st.session_state.tuned_metrics is not None
    m   = st.session_state.tuned_metrics if using_tuned else st.session_state.base_metrics
    cm_ = st.session_state.tuned_cm      if using_tuned else st.session_state.base_cm
    fi  = st.session_state.tuned_fi      if using_tuned else st.session_state.base_fi

    if using_tuned:
        st.markdown("""
        <div style="display:inline-flex;align-items:center;gap:8px;background:#f4f4f5;
        border:1px solid #e4e4e7;border-radius:6px;padding:8px 16px;margin-bottom:32px">
          <div style="width:6px;height:6px;background:#09090b;border-radius:50%"></div>
          <span style="font-size:11px;font-weight:600;color:#09090b;letter-spacing:.5px">
            Evaluating tuned model from Hyperparameter Tuning
          </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:inline-flex;align-items:center;gap:8px;background:#fafafa;
        border:1px solid #e4e4e7;border-radius:6px;padding:8px 16px;margin-bottom:32px">
          <div style="width:6px;height:6px;background:#a1a1aa;border-radius:50%"></div>
          <span style="font-size:11px;font-weight:500;color:#71717a;letter-spacing:.5px">
            Evaluating default model · Run Hyperparameter Tuning to update
          </span>
        </div>
        """, unsafe_allow_html=True)

    tn, fp, fn, tp = cm_
    total_test     = tn + fp + fn + tp

    # Metric cards grid
    st.markdown("""
    <div class="sec-label">Performance Metrics</div>
    <div class="sec-title">Test Set Results</div>
    <div class="sec-desc">Evaluated on 20% held-out data not seen during training.</div>
    """, unsafe_allow_html=True)

    e1, e2, e3, e4, e5 = st.columns(5)
    eval_metrics = [
        ("Accuracy",  m['accuracy'],  "Overall correct predictions", True),
        ("Precision", m['precision'], "Of flagged — how many are real", False),
        ("Recall",    m['recall'],    "Of real threats — how many caught", False),
        ("F1 Score",  m['f1'],        "Precision–Recall balance", False),
        ("AUC-ROC",   m['roc_auc'],   "Discrimination ability", False),
    ]
    for col, (lbl, val, desc, hi) in zip([e1, e2, e3, e4, e5], eval_metrics):
        with col:
            st.markdown(f"""
            <div class="m-card {"hi" if hi else ""}">
              <div class="m-lbl">{lbl}</div>
              <div class="m-val">{round(val*100,1)}%</div>
              <div class="m-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Confusion Matrix section
    st.markdown('<hr class="div-line" style="margin:44px 0">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-label">Confusion Matrix</div>
    <div class="sec-title">Prediction Breakdown</div>
    """, unsafe_allow_html=True)

    cm_col, interp_col = st.columns([1, 1.1], gap="large")

    with cm_col:
        st.markdown(f"""
        <div class="cm-outer">
          <table class="cm-table">
            <thead>
              <tr>
                <th class="cm-head" style="text-align:left"></th>
                <th class="cm-head">Predicted Normal</th>
                <th class="cm-head">Predicted Threat</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th class="cm-head" style="text-align:left;padding-left:18px">Actual Normal</th>
                <td class="cm-cell tn-c">
                  <div class="cm-n">{tn}</div>
                  <div class="cm-t">True Negative</div>
                </td>
                <td class="cm-cell fp-c">
                  <div class="cm-n">{fp}</div>
                  <div class="cm-t">False Positive</div>
                </td>
              </tr>
              <tr>
                <th class="cm-head" style="text-align:left;padding-left:18px">Actual Threat</th>
                <td class="cm-cell fn-c">
                  <div class="cm-n">{fn}</div>
                  <div class="cm-t">False Negative</div>
                </td>
                <td class="cm-cell tp-c">
                  <div class="cm-n">{tp}</div>
                  <div class="cm-t">True Positive</div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with interp_col:
        interp_items = [
            ("True Positive (TP)",  tp, f"Actual threats correctly caught. Each TP is a security incident prevented."),
            ("False Negative (FN)", fn, f"Real threats missed by the model. Critical to minimise — these are invisible attacks."),
            ("False Positive (FP)", fp, f"Normal users incorrectly flagged. Adds analyst workload but causes no direct harm."),
            ("True Negative (TN)",  tn, f"Normal users correctly cleared. {round(tn/total_test*100,1)}% of test records."),
        ]
        for label, count, desc in interp_items:
            st.markdown(f"""
            <div style="border-left:3px solid #09090b;padding:14px 18px;margin-bottom:8px;background:#ffffff;border:1px solid #e4e4e7;border-radius:0 6px 6px 0">
              <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px">
                <span style="font-size:12px;font-weight:600;color:#09090b">{label}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:18px;color:#09090b;font-weight:700">{count}</span>
              </div>
              <div style="font-size:11px;color:#71717a;line-height:1.6">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Why Random Forest (Preferred algorithm logic)
    st.markdown('<hr class="div-line" style="margin:44px 0">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-label">Algorithm Justification</div>
    <div class="sec-title">Why Random Forest?</div>
    <div class="sec-desc">
      Six evidence-backed reasons Random Forest is the optimal algorithm
      for insider threat detection on tabular email metadata.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="why-grid">
      <div class="why-card">
        <div class="why-ico">🌲</div>
        <div class="why-ttl">Ensemble Stability</div>
        <div class="why-dsc">
          Aggregates hundreds of independently trained decision trees via majority voting.
          This dramatically reduces variance — one noisy email record cannot skew the final prediction.
          Single trees overfit; forests generalize.
        </div>
      </div>
      <div class="why-card">
        <div class="why-ico">⚖️</div>
        <div class="why-ttl">Handles Class Imbalance</div>
        <div class="why-dsc">
          Insider threats are rare events (~{threat_rate}% of records). Bootstrap sampling ensures
          every tree sees a statistically representative mix of threats and normal activity,
          without requiring synthetic oversampling like SMOTE.
        </div>
      </div>
      <div class="why-card">
        <div class="why-ico">📊</div>
        <div class="why-ttl">Native Feature Importance</div>
        <div class="why-dsc">
          Provides built-in mean decrease in impurity scores for every feature — visible above.
          Analysts know <em>which</em> signal drove a flag (email volume vs. size vs. weekend),
          enabling explainable, auditable decisions.
        </div>
      </div>
      <div class="why-card">
        <div class="why-ico">🔗</div>
        <div class="why-ttl">Non-Linear Interactions</div>
        <div class="why-dsc">
          Threats emerge from <em>combinations</em>: moderate volume + large attachments + weekend
          activity is far more suspicious than any factor alone. RF tree splits capture these
          interactions natively without feature engineering.
        </div>
      </div>
      <div class="why-card">
        <div class="why-ico">📐</div>
        <div class="why-ttl">Scale Invariant</div>
        <div class="why-dsc">
          Tree-based splits are invariant to feature scaling and outliers. Unlike SVM or Logistic
          Regression, RF produces identical results whether email counts are in raw form or
          normalised — no StandardScaler artifacts can distort decisions.
        </div>
      </div>
      <div class="why-card">
        <div class="why-ico">🎯</div>
        <div class="why-ttl">Calibrated Risk Scores</div>
        <div class="why-dsc">
          Outputs a probability score (0–100%) rather than a binary flag. Security teams can
          prioritize investigation queues by risk level, allocating analyst time proportionally
          to severity — not treating all flags equally.
        </div>
      </div>
    </div>
    """.replace("{threat_rate}", str(THREAT_RATE)), unsafe_allow_html=True)

    # Why not alternatives
    st.markdown("""
    <div class="panel" style="margin-bottom:20px">
      <div class="panel-head">
        <div class="panel-dot" style="background:#09090b"></div>
        <span class="panel-ttl">Why Not the Alternatives?</span>
      </div>
      <div class="panel-body">
    """, unsafe_allow_html=True)

    alts = [
        ("Logistic Regression", "Assumes linear separability between classes. Insider threat patterns are fundamentally non-linear (a user who emails on Saturday AND sends large attachments is disproportionately suspicious), violating this core assumption and degrading recall on complex threat profiles."),
        ("Decision Tree (Single)", "Highly prone to overfitting — a single deep tree memorizes training noise rather than learning generalizable patterns. Minor variation in a user's email count can flip a prediction entirely. RF solves this via ensemble averaging."),
        ("Support Vector Machine", "Computationally expensive at scale, requires careful kernel selection, and doesn't naturally produce calibrated probabilities. Platt scaling adds complexity, and the model offers zero interpretability — unacceptable for a forensic security tool where every flag must be justifiable."),
    ]
    for name, reason in alts:
        st.markdown(f"""
        <div style="padding:14px 0;border-bottom:1px solid #f4f4f5">
          <div style="font-size:12px;font-weight:700;color:#09090b;margin-bottom:6px">{name}</div>
          <div style="font-size:11px;color:#71717a;line-height:1.7">{reason}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Comparative benchmark
    st.markdown('<hr class="div-line" style="margin:44px 0">', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-label">Benchmarking</div>
    <div class="sec-title">All Models Compared</div>
    <div class="sec-desc">Same dataset, same 80/20 split, same random seed. No cherry-picking.</div>
    """, unsafe_allow_html=True)

    with st.spinner("Benchmarking all algorithms…"):
        comp_df = get_comparison(st.session_state.tuned_params)

    rows_html = ""
    for _, row in comp_df.iterrows():
        is_rf = row["Model"] == "Random Forest"
        tr_cls = "rf-row" if is_rf else ""
        star   = " ★" if is_rf else ""

        def cell(val, col_name):
            is_best = val == comp_df[col_name].max()
            td_cls  = "best" if is_best else "norm"
            return f'<td class="{td_cls}">{val}%</td>'

        rows_html += f"""
        <tr class="{tr_cls}">
          <td><strong>{row['Model']}{star}</strong></td>
          {cell(row['Accuracy'],  'Accuracy')}
          {cell(row['Precision'], 'Precision')}
          {cell(row['Recall'],    'Recall')}
          {cell(row['F1'],        'F1')}
          {cell(row['AUC-ROC'],   'AUC-ROC')}
        </tr>
        """

    st.markdown(f"""
    <div class="cm-outer">
      <table class="ct">
        <thead>
          <tr>
            <th>Algorithm</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 Score</th>
            <th>AUC-ROC</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    <div style="font-size:10px;color:#71717a;margin-top:10px;font-family:'JetBrains Mono',monospace">
      ★ Selected model &nbsp;·&nbsp; Bold = best in column across all models
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: LIVE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "predict":

    st.markdown("""
    <div class="pg-hero" style="padding-bottom:40px">
      <div class="pg-eyebrow">Random Forest · Real-time Inference</div>
      <div class="pg-title">Live Risk<br><em>Assessment</em></div>
      <div class="pg-sub">
        Enter a user's behavioural data. The trained Random Forest model
        computes their insider threat probability in milliseconds.
      </div>
    </div>
    """, unsafe_allow_html=True)

    p_left, p_gap, p_right = st.columns([1, 0.06, 1.3])

    with p_left:
        st.markdown("""
        <div class="sec-label" style="margin-bottom:20px">User Behaviour Inputs</div>
        """, unsafe_allow_html=True)

        email_in  = st.number_input("Total Emails Sent",   min_value=0, value=12, step=1)
        attach_in = st.number_input("Attachments Sent",    min_value=0, value=0,  step=1)
        size_in   = st.number_input("Largest Email (bytes)", min_value=0, value=80000, step=10000)
        day_in    = st.slider("Day of Week  (0 = Mon · 6 = Sun)", 0, 6, 2)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div data-btn-primary="true">', unsafe_allow_html=True)
        predict_btn = st.button("Assess Risk", key="do_predict")
        st.markdown("</div>", unsafe_allow_html=True)

        # Reference Thresholds
        st.markdown("""
        <div style="margin-top:20px">
          <div class="card-lbl" style="margin-bottom:12px">Risk Thresholds</div>
          <div class="p-row" style="padding:10px 0">
            <span style="font-size:11px;color:#71717a">Email volume</span>
            <span class="gs-badge">&gt; 30</span>
          </div>
          <div class="p-row" style="padding:10px 0">
            <span style="font-size:11px;color:#71717a">Attachments</span>
            <span class="gs-badge">&gt; 5</span>
          </div>
          <div class="p-row" style="padding:10px 0">
            <span style="font-size:11px;color:#71717a">Day of week</span>
            <span class="gs-badge">≥ 4 (weekend)</span>
          </div>
          <div class="p-row" style="padding:10px 0;border-bottom:none">
            <span style="font-size:11px;color:#71717a">Email size</span>
            <span class="gs-badge">&gt; 1,000,000 bytes</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with p_right:
        if predict_btn:
            clf_live  = st.session_state.base_clf
            sc_live   = st.session_state.base_scaler

            inp_raw   = [[email_in, attach_in, day_in, size_in]]
            inp_sc    = sc_live.transform(inp_raw)
            label     = clf_live.predict(inp_sc)[0]
            prob      = clf_live.predict_proba(inp_sc)[0][1]
            pct       = round(prob * 100, 1)

            # Risk Verdict Display
            if label == 1:
                st.markdown(f"""
                <div class="v-threat">
                  <div style="font-size:24px;margin-bottom:10px;color:#ffffff">⚠</div>
                  <div class="v-ttl">Potential Insider Threat Detected</div>
                  <div class="v-via" style="color:#a1a1aa !important">RANDOM FOREST · CONFIDENCE {pct}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="v-safe">
                  <div style="font-size:24px;margin-bottom:10px;color:#10b981">✓</div>
                  <div class="v-ttl">No Threat Detected</div>
                  <div class="v-via" style="color:#71717a !important">RANDOM FOREST · THREAT PROBABILITY {pct}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Risk score visual indicator
            st.markdown(f"""
            <div class="risk-wrap">
              <div class="risk-hd">
                <span class="risk-lbl">Risk Score</span>
                <span class="risk-pct">{pct}%</span>
              </div>
              <div class="risk-track">
                <div class="risk-fill" style="width:{pct}%"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Input parameter anomaly breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card-lbl" style="margin-bottom:12px">Input Summary</div>', unsafe_allow_html=True)

            rows_breakdown = [
                ("Email Volume",   email_in,  email_in > 30,  "High volume anomaly"),
                ("Attachments",    attach_in, attach_in > 5,  "Exfiltration signal"),
                ("Day of Week",    f"{'Weekday' if day_in < 4 else 'Weekend'} ({day_in})", day_in >= 4, "Off-hours activity"),
                ("Email Size",     f"{size_in:,} B", size_in > 1_000_000, "Bulk transfer signal"),
            ]
            for feat, val, triggered, tip in rows_breakdown:
                indicator = "●" if triggered else "○"
                color     = "#ef4444" if triggered else "#a1a1aa"
                val_color = "#09090b"
                tip_color = "#ef4444" if triggered else "#71717a"
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                padding:12px 0;border-bottom:1px solid #f4f4f5">
                  <div>
                    <div style="font-size:12px;color:#71717a;margin-bottom:2px">{feat}</div>
                    <div style="font-size:11px;color:{tip_color}">{tip if triggered else "—"}</div>
                  </div>
                  <div style="display:flex;align-items:center;gap:10px">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:13px;color:{val_color};font-weight:600">{val}</span>
                    <span style="color:{color};font-size:14px">{indicator}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="height:340px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;
            background:#fafafa;border:1px solid #e4e4e7;border-radius:12px">
              <div style="font-size:36px;margin-bottom:16px;opacity:.2">◎</div>
              <div style="font-size:13px;color:#71717a;text-align:center;line-height:1.8">
                Enter user behaviour data<br>and press <strong style="color:#09090b">Assess Risk</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER (rendered across all pages)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="pg-footer">
  <div class="pg-footer-l">
    Detecting Insider Threats in Organizations using Machine Learning<br>
    <span style="color:#71717a">Random Forest Classifier · Scikit-learn · Streamlit</span>
  </div>
  <div class="pg-footer-r">
    {N_EMAILS:,} emails &nbsp;·&nbsp; {N_USERS:,} users &nbsp;·&nbsp; {THREAT_RATE}% flagged
  </div>
</div>
""", unsafe_allow_html=True)