# Customer Churn Predictor

A beginner-friendly Python machine-learning project that predicts telecom customer churn, compares Logistic Regression and Random Forest, reports model metrics, identifies influential features, and provides a Streamlit dashboard.

## Features
- Reproducible synthetic telecom dataset generator
- Data preprocessing with missing-value imputation, scaling, and one-hot encoding
- Logistic Regression and Random Forest comparison
- Hold-out evaluation: accuracy, precision, recall, F1, ROC-AUC
- Saved preprocessing + model pipeline
- Permutation feature importance
- Streamlit dashboard with CSV upload and downloadable risk predictions
- Example MySQL analysis queries

> The generated data is synthetic and for learning/demo purposes only. Results should not be used to make real customer decisions.

## 1. Open in VS Code
Open the `Customer-Churn-Predictor` folder in VS Code. Open **Terminal → New Terminal**.

## 2. Create and activate a virtual environment (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

## 3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Generate sample data
```bash
python generate_data.py
```
This creates `data/customer_churn.csv` with 2,000 synthetic customer rows.

## 5. Train models
```bash
python train_model.py
```
This creates:
- `models/churn_model.joblib`
- `reports/model_metrics.json`
- `reports/feature_importance.csv`

The script chooses the model with the better test ROC-AUC. Metrics vary if you change the dataset or code.

## 6. Run the dashboard
```bash
streamlit run app.py
```
Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser.

## 7. Try one prediction from the terminal
```bash
python predict.py
```

## 8. Use your own data
Upload a CSV in the dashboard. It should contain the same predictor columns as the training data. The `Churn` target column is optional for scoring; if present, it is shown for observed-data analysis. Train a new model on your real labeled data before using it for meaningful predictions.

Required target for training: `Churn`, with values such as `Yes` and `No`. The model excludes `customerID` and `Churn` from input features.

## 9. SQL analysis (optional)
Open `sql/churn_analysis.sql` in MySQL Workbench. Import `data/customer_churn.csv` into a table named `customer_churn`, then run the queries. The table import is done in MySQL Workbench using **Table Data Import Wizard**.

## Common issues
- **Model not found:** run `python train_model.py` after generating data.
- **No module named ...:** activate `.venv` and run `pip install -r requirements.txt`.
- **Streamlit command not found:** use `python -m streamlit run app.py`.
- **Upload prediction fails:** check the CSV has the predictor columns used for training and compatible values.
- **PowerShell activation blocked:** use the Command Prompt activation command shown above.

## How to explain this project in an interview
“I built a telecom customer churn prediction system using Python and scikit-learn. I compared Logistic Regression and Random Forest using a held-out test set, selected the model using ROC-AUC, and evaluated precision, recall, F1-score, and accuracy. I used permutation importance to identify influential churn features and built a Streamlit dashboard for batch scoring and downloadable predictions.”
