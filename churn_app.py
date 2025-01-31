import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
def load_model():
    with open("best_churn_model.pkl", "rb") as f:
        saved_data = pickle.load(f)
    return saved_data["model"], saved_data["scaler"], saved_data["feature_cols"], saved_data["num_cols"]

def preprocess_for_prediction(df, scaler, feature_cols, num_cols):
    """
    - Scale numeric columns
    - Ensure the input data matches the trained model's column order
    - Fill missing categorical features
    """
    df[num_cols] = scaler.transform(df[num_cols])

    # Ensure all required columns exist, fill missing ones with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0  # Assign default value

    df = df[feature_cols]
    return df

def main():
    st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

    # Sidebar for navigation
    st.sidebar.title("📌 Navigation")
    page = st.sidebar.radio("Go to", ["Churn Prediction", "Insights & Analysis"])

    if page == "Churn Prediction":
        churn_prediction_page()
    else:
        insights_analysis_page()

def churn_prediction_page():
    st.title("📊 Customer Churn Prediction")
    model, scaler, feature_cols, num_cols = load_model()

    # Create two columns for side-by-side layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📌 Customer Details")

        # Categorical features (Pre-filled with low churn values)
        gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1], index=1)
        senior_citizen = st.selectbox("Senior Citizen (0=No, 1=Yes)", [0, 1], index=0)
        partner = st.selectbox("Partner (0=No, 1=Yes)", [0, 1], index=1)
        dependents = st.selectbox("Dependents (0=No, 1=Yes)", [0, 1], index=1)
        phone_service = st.selectbox("Phone Service (0=No, 1=Yes)", [0, 1], index=1)
        multiple_lines = st.selectbox("Multiple Lines (0=No, 1=Yes, 2=No phone service)", [0, 1, 2], index=1)
        internet_service = st.selectbox("Internet Service (0=DSL, 1=Fiber optic, 2=No)", [0, 1, 2], index=0)
        online_security = st.selectbox("Online Security (0=No, 1=Yes, 2=No internet service)", [0, 1, 2], index=1)
        online_backup = st.selectbox("Online Backup (0=No, 1=Yes, 2=No internet service)", [0, 1, 2], index=1)

    with col2:
        st.header("📌 Additional Features")

        device_protection = st.selectbox("Device Protection (0=No, 1=Yes, 2=No internet service)", [0, 1, 2], index=1)
        tech_support = st.selectbox("Tech Support (0=No, 1=Yes, 2=No internet service)", [0, 1, 2], index=1)
        streaming_tv = st.selectbox("Streaming TV (0=No, 1=Yes, 2=No internet service)", [0, 1, 2], index=1)
        streaming_movies = st.selectbox("Streaming Movies (0=No, 1=Yes, 2=No internet service)", [0, 1, 2], index=1)
        contract = st.selectbox("Contract (0=Month-to-month, 1=One year, 2=Two year)", [0, 1, 2], index=2)
        paperless_billing = st.selectbox("Paperless Billing (0=No, 1=Yes)", [0, 1], index=0)
        payment_method = st.selectbox("Payment Method (0=Electronic check, 1=Mailed check, 2=Bank transfer, 3=Credit card)", [0, 1, 2, 3], index=3)

        # Numeric features (Pre-filled with low churn values)
        tenure = st.slider("Tenure (months)", 0, 72, 60)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=300.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=3000.0)

    # Construct the input DataFrame
    input_data = pd.DataFrame([[
        gender, senior_citizen, partner, dependents, phone_service, multiple_lines, 
        internet_service, online_security, online_backup, device_protection, 
        tech_support, streaming_tv, streaming_movies, contract, paperless_billing, 
        payment_method, tenure, monthly_charges, total_charges
    ]], columns=feature_cols)

    # Prediction button
    st.markdown("---")
    st.markdown("### 🔍 **Churn Prediction Result**")
    if st.button("Predict Churn", use_container_width=True):
        try:
            # Preprocess input to match training data
            processed_input = preprocess_for_prediction(input_data, scaler, feature_cols, num_cols)

            # Make prediction
            churn_prob = model.predict_proba(processed_input)[:, 1][0]

            # Display result
            st.subheader(f"🔮 Churn Probability: **{churn_prob:.3f}**")
            
            if churn_prob >= 0.5:
                st.error("⚠️ High risk of churn! Consider offering discounts or better service.")
            else:
                st.success("✅ Low risk of churn. Customer is likely to stay!")

            # Additional confidence level
            confidence = (1 - churn_prob) * 100 if churn_prob < 0.5 else churn_prob * 100
            st.info(f"📌 Model Confidence: **{confidence:.2f}%**")
            
        except Exception as e:
            st.error(f"Error: {e}")
def preprocess_data_for_correlation(df):
    df = df.copy()
    
    # Drop non-numeric columns
    df.drop(["customerID"], axis=1, errors="ignore", inplace=True)

    # Convert categorical variables to numerical using Label Encoding
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = pd.factorize(df[col])[0]  # Encode categories as unique integer values
    
    return df

def insights_analysis_page():
    st.title("📊 Advanced Insights & Analytics")

    df = pd.read_csv("Telconnect_data.csv")
    df.columns = df.columns.str.lower().str.strip()  # Standardizing column names
    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

    st.subheader("📌 Checking Column Names")
    st.write(df.columns.tolist())  # ✅ Print column names to debug

    st.subheader("📌 Churn Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="churn", palette="viridis")
    st.pyplot(fig)

    st.subheader("📌 Feature Importance")
    model, _, feature_cols, _ = load_model()
    feature_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    feature_importance.plot(kind="barh", ax=ax, color="teal")
    st.pyplot(fig)

    st.subheader("📌 Correlation Heatmap")
    df_numeric = df.select_dtypes(include=["number"])  # Select only numeric columns
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    st.pyplot(fig)

    st.subheader("📌 Churn by Contract Type")
    if "contract" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="churn", hue="contract", data=df, palette="magma")
        st.pyplot(fig)
    else:
        st.warning("⚠️ 'contract' column not found in the dataset!")

    if "monthlycharges" in df.columns:
        st.subheader("📌 Monthly Charges Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["monthlycharges"], bins=30, kde=True, color="blue")
        st.pyplot(fig)
    else:
        st.warning("⚠️ 'monthlycharges' column not found!")

    if "paymentmethod" in df.columns:
        st.subheader("📌 Churn by Payment Method")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="churn", hue="paymentmethod", data=df, palette="pastel")
        st.pyplot(fig)
    else:
        st.warning("⚠️ 'paymentmethod' column not found!")


if __name__ == "__main__":
    main()
