import os
import io
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Gwent Police Dashboard", layout="wide")
st.title("Exploratory Data Analysis & Predictive Modeling")

# -------------------------
# Load CSV
# -------------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df["year_month"] = df["month"].dt.to_period("M").astype(str)

    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def make_cm_df(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)

# -------------------------
# Sidebar input
# -------------------------
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
sample_path = "example.csv"

if uploaded_file:
    df = load_csv(uploaded_file)
elif os.path.exists(sample_path):
    df = load_csv(sample_path)
    st.info("Using example.csv")
else:
    st.warning("Please upload a CSV file.")
    st.stop()

st.success(f"Loaded {len(df)} rows")

# -------------------------
# Descriptive EDA
# -------------------------
st.header("Exploratory Data Analysis")

if "crime_type" in df.columns:
    st.subheader("Crimes by Type")
    counts = df["crime_type"].value_counts().reset_index()
    counts.columns = ["crime_type", "count"]
    chart = alt.Chart(counts).mark_bar().encode(
        x="count:Q", y=alt.Y("crime_type:N", sort="-x"), tooltip=["crime_type", "count"]
    )
    st.altair_chart(chart, use_container_width=True)

if "year_month" in df.columns:
    st.subheader("Crimes Over Time")
    ts = df.groupby("year_month").size().reset_index(name="count")
    ts["year_month"] = pd.to_datetime(ts["year_month"])
    line = alt.Chart(ts).mark_line(point=True).encode(
        x="year_month:T", y="count:Q"
    )
    st.altair_chart(line, use_container_width=True)

# -------------------------
# Predictive Modeling
# -------------------------
st.header("Predictive Modeling")

if "crime_type" not in df.columns or "year_month" not in df.columns:
    st.warning("Need 'crime_type' and 'year_month' for prediction.")
    st.stop()

# Select target and features
target = st.selectbox("Target column", ["crime_type"])
features = st.multiselect("Select features", 
                          [c for c in ["lsoa_name", "location", "year_month", "latitude", "longitude"] if c in df.columns],
                          default=["year_month"])

if not features:
    st.warning("Please select features")
    st.stop()

model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"])

# Train button
if st.button("Train Model"):
    data = df.dropna(subset=[target] + features)
    X, y = data[features], data[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preprocessing
    cat_cols = [c for c in features if X[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]

    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(with_mean=False), num_cols)
    ])

    # Model
    if model_choice == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)

    pipe = Pipeline([("prep", preproc), ("model", clf)])
    pipe.fit(X_train, y_train)

    # Predictions
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    st.metric("Accuracy", f"{acc:.2f}")
    st.metric("F1 Score", f"{f1:.2f}")

    st.subheader("Confusion Matrix")
    labels = sorted(y.unique())
    cm = make_cm_df(y_test, y_pred, labels)
    st.dataframe(cm)

    with st.expander("Classification Report"):
        st.text(classification_report(y_test, y_pred))
