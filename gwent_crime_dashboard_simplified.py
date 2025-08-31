import io
import os
import typing as t
import pandas as pd
import altair as alt
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Gwent Crime — Predictive Analytics Dashboard",
    page_icon="🚓",
    layout="wide"
)

st.title("🚓 Gwent Police Crime — Predictive Analytics Dashboard")
st.caption("EDA • Predictive Modeling • Interactive Insights")

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

    for col in ["lsoa_name", "location", "last_outcome_category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Drop unnecessary columns
    drop_cols = ["crime_id", "reported_by", "falls_within", "lsoa_code", "context", "year", "source"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df

def make_confusion_df(y_true, y_pred, labels) -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=pd.Index(labels, name="True"), columns=pd.Index(labels, name="Pred"))

# -------------------------
# Sidebar — Data input
# -------------------------
st.sidebar.header("📥 Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("👈 Upload a CSV in the sidebar to begin.")
    st.stop()

try:
    df = load_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read uploaded file: {e}")
    st.stop()

st.success(f"Loaded {len(df):,} rows • {df.shape[1]} columns")

# -------------------------
# EDA
# -------------------------
st.header("📊 Exploratory Data Analysis (EDA)")

colA, colB = st.columns(2)
with colA:
    if "crime_type" in df.columns:
        st.subheader("Crimes by Type")
        s = df["crime_type"].value_counts().reset_index()
        s.columns = ["crime_type", "count"]
        chart = alt.Chart(s).mark_bar().encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("crime_type:N", sort="-x", title="Crime Type"),
            tooltip=["crime_type", "count"]
        )
        st.altair_chart(chart, use_container_width=True)

with colB:
    if {"year_month", "crime_type"}.issubset(df.columns):
        st.subheader("Heatmap — Top 10 Crime Types by Month")

        # make sure year_month is datetime
        df["year_month"] = pd.to_datetime(df["year_month"], errors="coerce")

        # aggregate
        crime_month = df.groupby(["crime_type", "year_month"]).size().reset_index(name="count")

        # keep only top 10 crimes overall
        top10 = df["crime_type"].value_counts().head(10).index
        crime_month = crime_month[crime_month["crime_type"].isin(top10)]

        # build heatmap
        heatmap = alt.Chart(crime_month).mark_rect().encode(
            x=alt.X("year_month:T", title="Month", sort="x"),
            y=alt.Y("crime_type:N", title="Crime Type"),
            color=alt.Color("count:Q", title="Crimes", scale=alt.Scale(scheme="reds")),
            tooltip=["crime_type", "year_month:T", "count:Q"]
        )

        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.info("Columns 'year_month' and 'crime_type' are required for this chart.")
        
colC, colD = st.columns(2)
with colC:
    if {"year_month", "lsoa_name"}.issubset(df.columns):
        st.subheader("LSOA Month-on-Month Heatmap")
        lsoa_month = df.groupby(["lsoa_name", "year_month"]).size().reset_index(name="count")
        heatmap = alt.Chart(lsoa_month).mark_rect().encode(
            x=alt.X("year_month:N", title="Month"),
            y=alt.Y("lsoa_name:N", sort="-x", title="LSOA"),
            color=alt.Color("count:Q", title="Crimes"),
            tooltip=["lsoa_name", "year_month", "count"]
        )
        st.altair_chart(heatmap, use_container_width=True)

with colD:
    if {"latitude", "longitude"}.issubset(df.columns):
        st.subheader("Crime Map (sample up to 5,000 points)")
        map_df = df[["latitude", "longitude"]].dropna().sample(min(5000, len(df)), random_state=42)
        st.map(map_df.rename(columns={"latitude":"lat", "longitude":"lon"}))

# -------------------------
# Predictive Modeling (Last 6 months, Random Forest only)
# -------------------------
st.header("🤖 Predictive Model")

possible_targets = [c for c in ["crime_type", "last_outcome_category"] if c in df.columns]
if not possible_targets:
    st.warning("No suitable target column found.")
    st.stop()

target_col = possible_targets[0]
candidate_features = [c for c in df.columns if c not in [target_col, "month", "year_month"]]

# Only last 6 months for training
if "month" in df.columns and pd.api.types.is_datetime64_any_dtype(df["month"]):
    max_date = df["month"].max()
    six_months_ago = max_date - pd.DateOffset(months=6)
    df_model = df[df["month"] >= six_months_ago].copy()
else:
    df_model = df.copy()

st.info(f"Training model on last 6 months: {len(df_model):,} rows")

model_df = df_model.dropna(subset=[target_col]).copy()
X = model_df[candidate_features].copy()
y = model_df[target_col].astype(str)

@st.cache_resource(show_spinner=True)
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False))
            ]), num_cols)
        ]
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    pipe = Pipeline(steps=[("prep", preprocess), ("model", clf)])
    pipe.fit(X_train, y_train)

    return pipe, X_test, y_test

# Train only when user clicks
if st.button("🚀 Train Model"):
    pipe, X_test, y_test = train_model(X, y)
    y_pred = pipe.predict(X_test)
    labels = sorted(y.unique().tolist())

    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    st.metric("Macro F1", f"{f1_score(y_test, y_pred, average='macro'):.3f}")

    st.subheader("Confusion Matrix")
    cm_df = make_confusion_df(y_test, y_pred, labels)
    cm_chart = px.imshow(cm_df.values, x=labels, y=labels,
                         labels=dict(x="Predicted", y="True", color="Count"))
    st.plotly_chart(cm_chart, use_container_width=True)

    with st.expander("📄 Classification Report", expanded=False):
        st.text(classification_report(y_test, y_pred, zero_division=0))
