"""Streamlit dashboard for telecom churn analysis and predictions."""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

DATA_PATH = Path("data/customer_churn.csv")
MODEL_PATH = Path("models/churn_model.joblib")
METRICS_PATH = Path("reports/model_metrics.json")
IMPORTANCE_PATH = Path("reports/feature_importance.csv")
TARGET = "Churn"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Predictor")
st.caption("Predict customer churn risk and explore factors associated with leaving.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_default_data():
    return pd.read_csv(DATA_PATH)

def normalize_target(series):
    return series.astype(str).str.strip().str.lower().map(
        {"yes": "Yes", "no": "No", "1": "Yes", "0": "No", "true": "Yes", "false": "No"}
    )

if not MODEL_PATH.exists():
    st.error("Trained model not found. In the VS Code terminal, run: `python train_model.py`")
    st.stop()

saved = load_model()
pipeline = saved["pipeline"]
feature_columns = saved["feature_columns"]
threshold = float(saved.get("threshold", 0.5))

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    st.caption("For prediction, the file should contain the same input columns used during training. The Churn target column is optional.")
    st.divider()
    st.write(f"**Model:** {saved['model_name']}")
    st.write(f"**Decision threshold:** {threshold:.0%}")
    st.caption("Predictions are estimates, not guarantees.")

try:
    df = pd.read_csv(uploaded) if uploaded else load_default_data()
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

labeled = TARGET in df.columns
if labeled:
    labels = normalize_target(df[TARGET])
    valid_labels = labels.notna()
    churn_rate = (labels[valid_labels] == "Yes").mean() if valid_labels.any() else None
else:
    churn_rate = None

total = len(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers in file", f"{total:,}")
c2.metric("Observed churn rate", f"{churn_rate:.1%}" if churn_rate is not None else "Not provided")
c3.metric("Input features", str(len(feature_columns)))
c4.metric("Model", saved["model_name"])

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔮 Predictions", "🧭 Churn Drivers"])

with tab1:
    st.subheader("Customer data preview")
    st.dataframe(df.head(20), use_container_width=True)
    if labeled and churn_rate is not None:
        counts = labels[valid_labels].value_counts().reindex(["No", "Yes"], fill_value=0)
        chart_df = pd.DataFrame({"Status": counts.index, "Customers": counts.values})
        st.bar_chart(chart_df.set_index("Status"))
        st.caption("The chart shows observed labels in the loaded dataset, not predicted churn.")
    else:
        st.info("This file has no Churn column. Add actual churn labels to see the observed churn rate.")

with tab2:
    st.subheader("Score customers")
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        st.warning("Missing input columns will be filled by the trained preprocessing pipeline where possible: " + ", ".join(missing))
    X = df.reindex(columns=feature_columns).copy()
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    try:
        probabilities = pipeline.predict_proba(X)[:, 1]
        results = df.copy()
        results["ChurnProbability"] = probabilities.round(4)
        results["ChurnRisk"] = pd.cut(
            probabilities, bins=[-0.001, threshold, 0.65, 1.001],
            labels=["Low", "Medium", "High"], include_lowest=True
        ).astype(str)
        results["PredictedChurn"] = pd.Series(probabilities >= threshold, index=results.index).map({True: "Yes", False: "No"})
        st.write("**Risk definitions:** Low = below model threshold; Medium = threshold to 65%; High = 65% or above. If the threshold is 65% or higher, these bands may overlap conceptually; interpret the probability as the primary score.")
        a, b, c = st.columns(3)
        a.metric("Predicted at-risk customers", f"{(probabilities >= threshold).sum():,}")
        b.metric("Average predicted risk", f"{probabilities.mean():.1%}")
        c.metric("High-risk customers (≥65%)", f"{(probabilities >= 0.65).sum():,}")
        st.dataframe(results.sort_values("ChurnProbability", ascending=False).head(100),
                     use_container_width=True)
        st.download_button(
            "Download all predictions as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="customer_churn_predictions.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Prediction failed. Check that your CSV columns match the training dataset. Details: {exc}")

with tab3:
    st.subheader("Top churn drivers")
    if IMPORTANCE_PATH.exists():
        importance = pd.read_csv(IMPORTANCE_PATH).head(12).sort_values("importance_mean")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(importance["feature"], importance["importance_mean"], xerr=importance["importance_std"])
        ax.set_xlabel("Permutation importance (change in ROC-AUC)")
        ax.set_ylabel("Input feature")
        ax.set_title("Most influential features in the held-out test set")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Permutation importance measures how much model performance changes when a feature is shuffled. It indicates predictive influence, not proof of causation.")
        st.dataframe(importance.sort_values("importance_mean", ascending=False), use_container_width=True)
    else:
        st.info("Feature importance report not found. Run `python train_model.py` first.")

if METRICS_PATH.exists():
    with st.expander("Model evaluation details"):
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.write(f"Selected model: **{metrics['best_model']}**")
        st.write(f"Train rows: {metrics['train_rows']:,} | Test rows: {metrics['test_rows']:,}")
        st.json(metrics["models"])
        st.caption(metrics.get("note", ""))
