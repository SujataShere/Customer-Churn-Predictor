# Customer-Churn-Predictor

A Machine Learning web application built with Python and Streamlit to predict whether a customer is likely to leave a service (churn) based on customer profile, location, subscribed services, and billing information.

🚀 Project Overview

Customer churn prediction helps businesses identify customers who may stop using their services. This project compares multiple classification models and selects the final model based on the highest F1 score.

The Streamlit app allows users to enter customer details, run a prediction, and view the predicted churn risk and churn probability.

✨ Features

- Interactive Streamlit interface for entering customer information
- Customer profile, location, service, and billing details
- Multiple machine learning classification models
- Final model selection based on the highest F1 score
- Churn prediction and probability display
- Model loading from a saved churn_model.pkl file
- Expandable section to inspect model input data
  
🧠 Machine Learning Models

The project interface lists these models:
- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- 
The final model is selected using the F1 score. The exact selected algorithm and model evaluation scores should be documented here once confirmed from the training code or evaluation results.

🧾 Input Features

The application uses 28 features. The interface groups customer information into the following sections:
- Customer Information: gender, dependents, number of dependents, senior-citizen status, partner, and tenure
- Customer Location: country, city, state, ZIP code, latitude, longitude, and latitude/longitude
- Services: phone service, multiple lines, internet service, online security, online backup, device protection, tech support, streaming TV, and streaming movies
- Billing Information: contract, paperless billing, payment method, monthly charges, total charges, and customer lifetime value (CLTV)
- 
📸 Sample Prediction

Example shown in the application output:
Result	Value
Prediction	No Churn
Churn Risk	Low Churn Risk
Churn Probability	17.07%
Customer Contract	Month-to-month
Monthly Charges	70.00
Total Charges	840.00

This is one sample prediction from the provided application output. Results can vary based on the entered customer details.

🛠️ Tech Stack

- Language: Python
- Web App: Streamlit
- Machine Learning: scikit-learn (if used in the training pipeline)
- Model File: churn_model.pkl
- Data Processing: pandas / NumPy (if used in the project code)
Keep only the libraries that are actually used by your source code in the final tech-stack list.

📁 Suggested Project Structure

Customer-Churn-Prediction/
├── app.py
├── churn_model.pkl
├── requirements.txt
├── README.md
└── data/                  # Include this only if your project uses a data folder
Your actual project may have additional files, such as a training notebook, dataset, preprocessing code, or evaluation charts. Add them to this structure if they are present in your repository.


📊 Model Evaluation

The final model is selected based on the highest F1 score, according to the application interface.

For a complete ML project README, add the measured values from your actual model evaluation:
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- ROC-AUC (if calculated)
Do not add estimated or invented scores; report values produced by your training/evaluation code.

🌟 Future Improvements

- Display model comparison results in the application
- Add confusion matrix and classification report visualizations
- Explain the main factors influencing each prediction
- Improve model performance through feature engineering and hyperparameter tuning
- Add model monitoring and retraining as new customer data becomes available
- Deploy the Streamlit app to a hosting platform
