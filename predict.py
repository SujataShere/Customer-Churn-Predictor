"""Small command-line example for scoring one customer row."""
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("models/churn_model.joblib")

def predict_customer(customer: dict) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train_model.py` first.")
    saved = joblib.load(MODEL_PATH)
    X = pd.DataFrame([customer])
    # Add absent trained columns as missing; discard unexpected columns.
    columns = saved["feature_columns"]
    X = X.reindex(columns=columns)
    probability = float(saved["pipeline"].predict_proba(X)[0, 1])
    threshold = float(saved.get("threshold", 0.5))
    return {
        "churn_probability": round(probability, 4),
        "risk": "High" if probability >= 0.65 else ("Medium" if probability >= threshold else "Low"),
        "prediction": "Likely to churn" if probability >= threshold else "Unlikely to churn",
        "model": saved["model_name"],
    }

if __name__ == "__main__":
    example = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5,
        "TotalCharges": 191.0,
    }
    print(predict_customer(example))
