"""Generate a reproducible synthetic telecom customer churn dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_CUSTOMERS = 2000
OUTPUT = Path("data/customer_churn.csv")

def main():
    rng = np.random.default_rng(RANDOM_STATE)
    gender = rng.choice(["Female", "Male"], N_CUSTOMERS)
    senior = rng.choice([0, 1], N_CUSTOMERS, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], N_CUSTOMERS, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], N_CUSTOMERS, p=[0.30, 0.70])
    tenure = rng.integers(0, 73, N_CUSTOMERS)
    phone = rng.choice(["Yes", "No"], N_CUSTOMERS, p=[0.90, 0.10])
    multiple = np.where(phone == "No", "No phone service",
                        rng.choice(["Yes", "No"], N_CUSTOMERS))
    internet = rng.choice(["DSL", "Fiber optic", "No"], N_CUSTOMERS, p=[0.35, 0.48, 0.17])

    def service_for_internet():
        values = []
        for service in internet:
            if service == "No":
                values.append("No internet service")
            else:
                values.append(rng.choice(["Yes", "No"], p=[0.42, 0.58]))
        return np.array(values)

    online_security = service_for_internet()
    online_backup = service_for_internet()
    device_protection = service_for_internet()
    tech_support = service_for_internet()
    streaming_tv = service_for_internet()
    streaming_movies = service_for_internet()
    contract = rng.choice(["Month-to-month", "One year", "Two year"],
                          N_CUSTOMERS, p=[0.56, 0.23, 0.21])
    paperless = rng.choice(["Yes", "No"], N_CUSTOMERS, p=[0.60, 0.40])
    payment = rng.choice(["Electronic check", "Mailed check",
                          "Bank transfer (automatic)", "Credit card (automatic)"],
                         N_CUSTOMERS, p=[0.34, 0.18, 0.24, 0.24])

    monthly = rng.normal(68, 22, N_CUSTOMERS).clip(18, 120).round(2)
    monthly += np.where(internet == "Fiber optic", 15, 0)
    monthly = monthly.clip(18, 125).round(2)
    total = (monthly * tenure * rng.uniform(0.90, 1.08, N_CUSTOMERS)).round(2)
    total[tenure == 0] = 0.0

    # Synthetic churn signal: contract, short tenure, electronic check, and high bills.
    logit = (
        -1.30
        + 1.05 * (contract == "Month-to-month")
        - 0.75 * (contract == "Two year")
        - 0.035 * tenure
        + 0.012 * (monthly - 65)
        + 0.55 * (payment == "Electronic check")
        + 0.35 * (internet == "Fiber optic")
        + 0.35 * (tech_support == "No")
        + 0.30 * (online_security == "No")
        + 0.25 * (senior == 1)
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(N_CUSTOMERS) < probability, "Yes", "No")

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(1, N_CUSTOMERS + 1)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": churn,
    })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df):,} synthetic customer records to {OUTPUT}")
    print(f"Sample churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
    print("Note: this is synthetic demonstration data, not real customer data.")

if __name__ == "__main__":
    main()
