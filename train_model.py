"""Train and evaluate churn models; save the best pipeline and reports."""
from pathlib import Path
import json
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, precision_score,
    recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

DATA_PATH = Path("data/customer_churn.csv")
MODEL_PATH = Path("models/churn_model.joblib")
METRICS_PATH = Path("reports/model_metrics.json")
IMPORTANCE_PATH = Path("reports/feature_importance.csv")
TARGET = "Churn"
DROP_COLUMNS = ["customerID", TARGET]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python generate_data.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET}'.")

    df = df.dropna(subset=[TARGET]).copy()
    y = df[TARGET].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "1": 1, "0": 0}
    )
    valid = y.notna()
    df, y = df.loc[valid].copy(), y.loc[valid].astype(int)
    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # Coerce common telecom numeric columns to numeric where possible.
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
    }
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    results = {}
    fitted = {}
    for name, classifier in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", classifier)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        }
        results[name] = metrics
        fitted[name] = pipeline
        print(f"\n{name}")
        print(json.dumps(metrics, indent=2))
        print(classification_report(y_test, predictions, target_names=["No churn", "Churn"],
                                    zero_division=0))

    # Select by ROC-AUC, a useful ranking metric for churn-risk scoring.
    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_pipeline = fitted[best_name]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": best_pipeline,
        "feature_columns": X.columns.tolist(),
        "model_name": best_name,
        "threshold": 0.50,
    }, MODEL_PATH)

    # Permutation importance measures the impact of each original input column.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        importance = permutation_importance(
            best_pipeline, X_test, y_test, n_repeats=5,
            random_state=42, scoring="roc_auc", n_jobs=-1
        )
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    report = {
        "best_model": best_name,
        "test_rows": int(len(X_test)),
        "train_rows": int(len(X_train)),
        "churn_rate": round(float(y.mean()), 4),
        "models": results,
        "target": TARGET,
        "note": "Metrics are from a held-out test set. Dataset is synthetic if generated by generate_data.py."
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nBest model: {best_name}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved feature importance: {IMPORTANCE_PATH}")

if __name__ == "__main__":
    main()
