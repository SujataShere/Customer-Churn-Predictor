import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Prediction System")
st.markdown(
    """
    ### Predict whether a customer is likely to churn
    Enter the customer details below and click **Predict Churn**.
    """
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_FILE = "churn_model.pkl"
FEATURE_FILE = "feature_columns.pkl"


if not os.path.exists(MODEL_FILE):
    st.error(
        f"❌ `{MODEL_FILE}` not found.\n\n"
        "Make sure churn_model.pkl is in the same folder as app.py."
    )
    st.stop()


if not os.path.exists(FEATURE_FILE):
    st.error(
        f"❌ `{FEATURE_FILE}` not found.\n\n"
        "Make sure feature_columns.pkl is in the same folder as app.py."
    )
    st.stop()


try:
    model = joblib.load(MODEL_FILE)
    feature_columns = joblib.load(FEATURE_FILE)

except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()


# ============================================================
# CHECK FEATURE COLUMNS
# ============================================================

if not isinstance(feature_columns, list):
    st.error("❌ feature_columns.pkl does not contain a valid list of columns.")
    st.stop()


st.success("✅ Churn model loaded successfully!")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value):
    """
    Convert a value to float.
    """
    try:
        return float(value)
    except:
        return 0.0


def safe_int(value):
    """
    Convert a value to integer.
    """
    try:
        return int(value)
    except:
        return 0


# ============================================================
# DEFAULT VALUES
# ============================================================

# These are used to construct the customer input.
# The final dataframe is always reordered according to
# feature_columns.pkl.

default_values = {

    # --------------------------------------------------------
    # Customer / Geographic information
    # --------------------------------------------------------

    "Count": 1,

    "Country": "United States",

    "State": "California",

    "City": "Los Angeles",

    "Zip Code": 90001,

    "Latitude": 33.9736,

    "Longitude": -118.2490,

    "Lat Long": "33.9736, -118.2490",

    "CLTV": 5000,

    # --------------------------------------------------------
    # Customer information
    # --------------------------------------------------------

    "Gender": "Male",

    "Senior Citizen": 0,

    "Partner": "No",

    "Dependents": "No",

    # --------------------------------------------------------
    # Services
    # --------------------------------------------------------

    "Tenure Months": 12,

    "Phone Service": "Yes",

    "Multiple Lines": "No",

    "Internet Service": "DSL",

    "Online Security": "No",

    "Online Backup": "No",

    "Device Protection": "No",

    "Tech Support": "No",

    "Streaming TV": "No",

    "Streaming Movies": "No",

    # --------------------------------------------------------
    # Contract / Billing
    # --------------------------------------------------------

    "Contract": "Month-to-month",

    "Paperless Billing": "Yes",

    "Payment Method": "Electronic check",

    "Monthly Charges": 70.0,

    "Total Charges": 840.0,

    # --------------------------------------------------------
    # Other possible columns
    # --------------------------------------------------------

    "Satisfaction Score": 3,

    "Customer Status": "Stayed",

    "Churn Category": "None",

    "Churn Score": 0,

    "Churn Value": 0
}


# ============================================================
# POSSIBLE CATEGORICAL VALUES
# ============================================================

categorical_options = {

    "Gender": [
        "Male",
        "Female"
    ],

    "Senior Citizen": [
        0,
        1
    ],

    "Partner": [
        "Yes",
        "No"
    ],

    "Dependents": [
        "Yes",
        "No"
    ],

    "Phone Service": [
        "Yes",
        "No"
    ],

    "Multiple Lines": [
        "Yes",
        "No",
        "No phone service"
    ],

    "Internet Service": [
        "DSL",
        "Fiber optic",
        "No"
    ],

    "Online Security": [
        "Yes",
        "No",
        "No internet service"
    ],

    "Online Backup": [
        "Yes",
        "No",
        "No internet service"
    ],

    "Device Protection": [
        "Yes",
        "No",
        "No internet service"
    ],

    "Tech Support": [
        "Yes",
        "No",
        "No internet service"
    ],

    "Streaming TV": [
        "Yes",
        "No",
        "No internet service"
    ],

    "Streaming Movies": [
        "Yes",
        "No",
        "No internet service"
    ],

    "Contract": [
        "Month-to-month",
        "One year",
        "Two year"
    ],

    "Paperless Billing": [
        "Yes",
        "No"
    ],

    "Payment Method": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ],

    "Customer Status": [
        "Stayed",
        "Joined",
        "Churned"
    ],

    "Churn Category": [
        "None",
        "Competitor",
        "Dissatisfaction",
        "Attitude",
        "Price",
        "Other"
    ]
}


# ============================================================
# INPUT COLLECTION
# ============================================================

customer_data = {}


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    if "Gender" in feature_columns:
        customer_data["Gender"] = st.selectbox(
            "Gender",
            categorical_options["Gender"]
        )

    if "Senior Citizen" in feature_columns:
        customer_data["Senior Citizen"] = st.selectbox(
            "Senior Citizen",
            categorical_options["Senior Citizen"]
        )

    if "Partner" in feature_columns:
        customer_data["Partner"] = st.selectbox(
            "Partner",
            categorical_options["Partner"]
        )


with col2:

    if "Dependents" in feature_columns:
        customer_data["Dependents"] = st.selectbox(
            "Dependents",
            categorical_options["Dependents"]
        )

    if "Tenure Months" in feature_columns:
        customer_data["Tenure Months"] = st.number_input(
            "Tenure Months",
            min_value=0,
            max_value=100,
            value=12,
            step=1
        )


with col3:

    if "Count" in feature_columns:
        customer_data["Count"] = st.number_input(
            "Count",
            min_value=1,
            value=1,
            step=1
        )


# ============================================================
# GEOGRAPHICAL INFORMATION
# ============================================================

st.header("🌎 Customer Location")

col1, col2, col3 = st.columns(3)


with col1:

    if "Country" in feature_columns:
        customer_data["Country"] = st.text_input(
            "Country",
            value="United States"
        )

    if "State" in feature_columns:
        customer_data["State"] = st.text_input(
            "State",
            value="California"
        )


with col2:

    if "City" in feature_columns:
        customer_data["City"] = st.text_input(
            "City",
            value="Los Angeles"
        )

    if "Zip Code" in feature_columns:
        customer_data["Zip Code"] = st.number_input(
            "Zip Code",
            min_value=0,
            value=90001,
            step=1
        )


with col3:

    if "Latitude" in feature_columns:
        customer_data["Latitude"] = st.number_input(
            "Latitude",
            value=33.9736,
            format="%.6f"
        )

    if "Longitude" in feature_columns:
        customer_data["Longitude"] = st.number_input(
            "Longitude",
            value=-118.2490,
            format="%.6f"
        )


if "Lat Long" in feature_columns:

    customer_data["Lat Long"] = st.text_input(
        "Lat Long",
        value="33.9736, -118.2490"
    )


# ============================================================
# SERVICES
# ============================================================

st.header("📱 Services")

col1, col2, col3 = st.columns(3)


with col1:

    if "Phone Service" in feature_columns:
        customer_data["Phone Service"] = st.selectbox(
            "Phone Service",
            categorical_options["Phone Service"]
        )

    if "Multiple Lines" in feature_columns:
        customer_data["Multiple Lines"] = st.selectbox(
            "Multiple Lines",
            categorical_options["Multiple Lines"]
        )

    if "Internet Service" in feature_columns:
        customer_data["Internet Service"] = st.selectbox(
            "Internet Service",
            categorical_options["Internet Service"]
        )


with col2:

    if "Online Security" in feature_columns:
        customer_data["Online Security"] = st.selectbox(
            "Online Security",
            categorical_options["Online Security"]
        )

    if "Online Backup" in feature_columns:
        customer_data["Online Backup"] = st.selectbox(
            "Online Backup",
            categorical_options["Online Backup"]
        )

    if "Device Protection" in feature_columns:
        customer_data["Device Protection"] = st.selectbox(
            "Device Protection",
            categorical_options["Device Protection"]
        )


with col3:

    if "Tech Support" in feature_columns:
        customer_data["Tech Support"] = st.selectbox(
            "Tech Support",
            categorical_options["Tech Support"]
        )

    if "Streaming TV" in feature_columns:
        customer_data["Streaming TV"] = st.selectbox(
            "Streaming TV",
            categorical_options["Streaming TV"]
        )

    if "Streaming Movies" in feature_columns:
        customer_data["Streaming Movies"] = st.selectbox(
            "Streaming Movies",
            categorical_options["Streaming Movies"]
        )


# ============================================================
# BILLING INFORMATION
# ============================================================

st.header("💳 Billing Information")

col1, col2, col3 = st.columns(3)


with col1:

    if "Contract" in feature_columns:
        customer_data["Contract"] = st.selectbox(
            "Contract",
            categorical_options["Contract"]
        )


with col2:

    if "Paperless Billing" in feature_columns:
        customer_data["Paperless Billing"] = st.selectbox(
            "Paperless Billing",
            categorical_options["Paperless Billing"]
        )


with col3:

    if "Payment Method" in feature_columns:
        customer_data["Payment Method"] = st.selectbox(
            "Payment Method",
            categorical_options["Payment Method"]
        )


col1, col2, col3 = st.columns(3)


with col1:

    if "Monthly Charges" in feature_columns:
        customer_data["Monthly Charges"] = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )


with col2:

    if "Total Charges" in feature_columns:
        customer_data["Total Charges"] = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=10.0
        )


with col3:

    if "CLTV" in feature_columns:
        customer_data["CLTV"] = st.number_input(
            "CLTV",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )


# ============================================================
# OTHER INFORMATION
# ============================================================

st.header("📋 Additional Information")

if "Satisfaction Score" in feature_columns:

    customer_data["Satisfaction Score"] = st.slider(
        "Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_df = pd.DataFrame([customer_data])


# ============================================================
# ADD MISSING COLUMNS
# ============================================================

# This section is very important.
# It ensures that the model receives exactly the same
# columns that were present during training.

for column in feature_columns:

    if column not in input_df.columns:

        if column in default_values:
            input_df[column] = default_values[column]

        else:
            input_df[column] = 0


# ============================================================
# REMOVE EXTRA COLUMNS
# ============================================================

input_df = input_df[feature_columns]


# ============================================================
# SHOW INPUT DATA
# ============================================================

with st.expander("🔍 View Model Input Data"):

    st.dataframe(
        input_df,
        use_container_width=True
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Churn",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(input_df)[0]

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = None

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(
                input_df
            )[0][1]


        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        st.divider()

        st.header("🎯 Prediction Result")


        if prediction == 1:

            st.error(
                "⚠️ HIGH CHURN RISK\n\n"
                "This customer is predicted to churn."
            )

            if probability is not None:

                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )


        else:

            st.success(
                "✅ LOW CHURN RISK\n\n"
                "This customer is predicted to stay."
            )

            if probability is not None:

                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )


        # ----------------------------------------------------
        # Prediction details
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            if prediction == 1:
                st.write("**Prediction:** Churn")
            else:
                st.write("**Prediction:** No Churn")


        with col2:

            if probability is not None:

                st.write(
                    f"**Probability:** "
                    f"{probability * 100:.2f}%"
                )


    except Exception as e:

        st.error(
            "❌ Prediction Error"
        )

        st.code(str(e))

        st.warning(
            """
            The input columns do not match the columns used
            during model training.

            Check feature_columns.pkl and make sure it was
            generated from the same training dataset.
            """
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Customer Churn")

st.sidebar.info(
    """
    **Machine Learning Models**

    • Logistic Regression  
    • Decision Tree  
    • Random Forest  
    • AdaBoost  
    • Gradient Boosting  

    The final model is selected based on
    the highest F1 Score.
    """
)


st.sidebar.success(
    "Model: churn_model.pkl"
)

st.sidebar.success(
    f"Features: {len(feature_columns)}"
)