import io
import os
import typing as t
from datetime import datetime

import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------
# Page config
# -------------------------

st.set_page_config(
    page_title="Gwent Police — Analytics Dashboard",
    page_icon="",
    layout="wide"
)

st.title("EDA — Predictive Analytics Dashboard")
st.caption("EDA • Predictive Modeling • AI and ML")

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
    for col in ["lsoa_name", "lsoa_code", "location", "reported_by", "falls_within", "last_outcome_category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def pick_top_categories(series: pd.Series, top_n: int = 30) -> pd.Series:
    counts = series.value_counts(dropna=False)
    top = counts.head(top_n).index
    return series.where(series.isin(top), other="Other")

# -------------------------
# Sidebar — Data input
# -------------------------

st.sidebar.header("Gwent police Dashboard")
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
# Dataset Stats
# -------------------------

# Total crimes = total rows
total_crimes = len(df)

# Total months (if available)
total_months = df["year_month"].nunique() if "year_month" in df.columns else "N/A"

# Drop 'context' column from missing check if it exists
df_check = df.drop(columns=["context"], errors="ignore")

# Find rows with NaN or empty values (excluding 'context')
incomplete_rows = df_check[
    df_check.isna().any(axis=1) | (df_check.astype(str).apply(lambda x: x.str.strip() == "").any(axis=1))
]

incomplete_count = len(incomplete_rows)

# Show summary stats
col1, col2, col3 = st.columns(3)
col1.metric("Total Crimes", f"{total_crimes:,}")
col2.metric("Total Months", f"{total_months}")
col3.metric("Incomplete Rows", f"{incomplete_count:,}")

# -------------------------
# EDA
# -------------------------
st.header("Exploratory Data Analysis (EDA)")

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
    if "year_month" in df.columns:
        st.subheader("Trend by Month")
        ts = df.groupby("year_month").size().reset_index(name="count")
        ts["year_month"] = pd.to_datetime(ts["year_month"])
        line = alt.Chart(ts).mark_line(point=True).encode(
            x=alt.X("year_month:T", title="Month"),
            y=alt.Y("count:Q", title="Crimes"),
            tooltip=["year_month:T", "count:Q"]
        )
        st.altair_chart(line, use_container_width=True)

colC, colD = st.columns(2)
with colC:
    if "lsoa_name" in df.columns:
        st.subheader("Top Crime Locations")
        top_lsoa = df["lsoa_name"].value_counts().head(20).reset_index()
        top_lsoa.columns = ["lsoa_name", "count"]
        bar = alt.Chart(top_lsoa).mark_bar().encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("lsoa_name:N", sort="-x", title="LSOA"),
            tooltip=["lsoa_name", "count"]
        )
        st.altair_chart(bar, use_container_width=True)

with colD:
    if {"latitude", "longitude"}.issubset(df.columns):
        st.subheader("Crime Map (sample upto 5,000 pts)")
        map_df = df[["latitude", "longitude"]].dropna().sample(min(5000, len(df)), random_state=42)
        st.map(map_df.rename(columns={"latitude":"lat", "longitude":"lon"}))

st.subheader("Heatmap — Top 10 Crime Types by Month")

if {"month", "crime_type"}.issubset(df.columns):
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["year_month"] = df["month"].dt.to_period("M").astype(str)
    crime_month = (
        df.groupby(["crime_type", "year_month"])
          .size()
          .reset_index(name="count")
    )
    top10_types = df["crime_type"].value_counts().head(10).index
    crime_month = crime_month[crime_month["crime_type"].isin(top10_types)]
    heatmap = alt.Chart(crime_month).mark_rect().encode(
        x=alt.X("year_month:N", title="Month",
                sort=sorted(crime_month["year_month"].unique())),
        y=alt.Y("crime_type:N", title="Crime Type"),
        color=alt.Color("count:Q", title="Crimes", scale=alt.Scale(scheme="reds")),
        tooltip=["crime_type", "year_month", "count:Q"]
    )
    st.altair_chart(heatmap, use_container_width=True)
else:
    st.info("Columns 'month' and 'crime_type' are required for this chart.")

# -------------------------
# Predictive Modeling (6 history + 6 forecast)
# -------------------------

st.header("Predictive Model (6 Months History + 6 Months Forecast)")

if "year_month" in df.columns and "crime_type" in df.columns:
    if st.button("Run Predictor"):
        # Aggregate monthly counts
        ts = (
            df.groupby(["year_month", "crime_type"])
              .size()
              .reset_index(name="count")
        )
        ts["year_month"] = pd.to_datetime(ts["year_month"], errors="coerce")

        # Pick top 6 crime types
        top6_types = df["crime_type"].value_counts().head(6).index
        ts_top6 = ts[ts["crime_type"].isin(top6_types)]

        # Features
        ts_top6["year"] = ts_top6["year_month"].dt.year
        ts_top6["month"] = ts_top6["year_month"].dt.month
        ts_top6["time_index"] = (
            (ts_top6["year"] - ts_top6["year"].min()) * 12 + ts_top6["month"]
        )

        # 6 months history
        max_date = ts_top6["year_month"].max()
        min_date = max_date - pd.DateOffset(months=5)
        history_df = ts_top6[ts_top6["year_month"].between(min_date, max_date)]

        # 6 months forecast
        future_months = pd.date_range(
            start=max_date + pd.offsets.MonthBegin(1),
            periods=6, freq="MS"
        )
        future_df = pd.DataFrame({
            "year_month": future_months,
            "year": future_months.year,
            "month": future_months.month,
            "time_index": (
                (future_months.year - ts_top6["year"].min()) * 12 + future_months.month
            )
        })

        preds, metrics = [], []
        for crime in top6_types:
            sub = ts_top6[ts_top6["crime_type"] == crime]
            X = sub[["time_index", "year", "month"]]
            y = sub["count"]

            if len(sub) > 12:
                model = RandomForestRegressor(n_estimators=500, random_state=42)
                model.fit(X, y)

                # Evaluate
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                metrics.append([crime, round(r2, 3), round(mae, 2), round(rmse, 2)])

                # Forecast
                future_counts = model.predict(future_df[["time_index", "year", "month"]])
                temp = future_df.copy()
                temp["crime_type"] = crime
                temp["count"] = np.round(future_counts).astype(int)
                preds.append(temp)

        pred_df = pd.concat(preds)

        # Mark history vs prediction
        history_df = history_df.copy()
        history_df["Type"] = "History"
        pred_df["Type"] = "Prediction"
        combined = pd.concat([history_df, pred_df])

        # Chart
        forecast_line = alt.Chart(combined).mark_line(point=True).encode(
            x="year_month:T",
            y="count:Q",
            color="crime_type:N",
            strokeDash="Type:N",
            tooltip=["year_month:T", "crime_type:N", "count:Q", "Type:N"]
        ).properties(height=400)

        st.subheader("Crime Trends (6 Months History + 6 Months Forecast)")
        st.altair_chart(forecast_line, use_container_width=True)

        # Metrics table
        if metrics:
            st.subheader("Model Success Rate (Training Performance)")
            metric_df = pd.DataFrame(metrics, columns=["Crime Type", "R²", "MAE", "RMSE"])
            st.dataframe(metric_df)
else:
    st.warning("Both 'year_month' and 'crime_type' columns are required.")
