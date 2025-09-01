import io
import os
import typing as t
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Page config
# -------------------------

st.set_page_config(
    page_title="Gwent Police — Analytics Dashboard",
    page_icon="",
    layout="wide"
)

st.title("EDA — Predictive Analytics Dashboard (Random Forest Only)")
st.caption("Random Forest • Predictive Modeling • AI and ML")

# -------------------------
# Helper functions
# -------------------------
@st.cache_data(show_spinner=False)
def load_csv(file: t.Union[str, io.BytesIO]) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    if "month" in df.columns:
        try:
            df["month"] = pd.to_datetime(df["month"], errors="coerce")
        except Exception:
            pass
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "crime_type" in df.columns:
        df = df[~df["crime_type"].isna()]
    if "month" in df.columns:
        df["year_month"] = df["month"].dt.to_period("M").astype(str)
        df["year"] = df["month"].dt.year
    for col in ["lsoa_name", "lsoa_code", "location"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

# -------------------------
# Sidebar — Data input
# -------------------------

st.sidebar.header("Gwent Police Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Police.UK street-level CSV")

# Optional sample path
default_path = "example.csv"
use_sample = st.sidebar.toggle("Use example file name in current folder", value=False)

df = None
if uploaded_file is not None:
    df = load_csv(uploaded_file)
elif use_sample and os.path.exists(default_path):
    df = load_csv(default_path)
else:
    st.info("Upload a CSV in the sidebar to begin.")
    st.stop()

st.success(f"Loaded {len(df):,} rows • {df.shape[1]} columns")

# -------------------------
# Predictive Modeling — Random Forest Only
# -------------------------

st.header("Predictive Model — Random Forest")

if {"year_month", "crime_type", "year"}.issubset(df.columns):

    # Select years for training
    years = sorted(df["year"].dropna().unique())
    default_years = years[-2:] if len(years) > 2 else years
    selected_years = st.multiselect("Select training years", options=years, default=default_years)
    if not selected_years:
        st.warning("Please select at least one year for training.")
        st.stop()

    model_df = df[df["year"].isin(selected_years)].dropna(subset=["crime_type"]).copy()

    # Features (fixed)
    features = []
    for col in ["location", "longitude", "latitude", "lsoa_name", "year_month"]:
        if col in model_df.columns:
            features.append(col)

    if not features:
        st.error("Required columns not found in dataset.")
        st.stop()

    X = model_df[features].copy()
    y = model_df["crime_type"].astype(str)

    # Preprocessing
    cat_cols = [c for c in features if X[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(with_mean=False), num_cols)
    ])

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

    pipe = Pipeline(steps=[("prep", preprocess), ("model", clf)])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    if st.button("Start Training"):
        with st.spinner("Training Random Forest..."):
            pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")

        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("Macro F1", f"{f1m:.3f}")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, zero_division=0))
else:
    st.warning("Dataset missing required columns: year_month, year, and crime_type")
