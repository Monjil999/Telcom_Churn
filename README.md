

📡 Telecom Customer Churn Prediction
🔍 Overview
Customer churn is a critical challenge in the telecom industry. This project builds a machine learning model to predict whether a customer is likely to churn based on historical data. It includes a Streamlit web application for easy interaction, allowing users to input customer details and get churn predictions.

🎯 Objectives
Identify customers likely to churn.
Provide actionable insights for customer retention.
Build a Streamlit web app for real-time churn prediction.
Perform advanced analytics for business insights.
🛠️ Tech Stack
Python 🐍
Pandas, NumPy, Seaborn, Matplotlib (Data Analysis & Visualization) 📊
Scikit-learn, XGBoost, LightGBM, RandomForest (Machine Learning) 🤖
SMOTE (Synthetic Minority Over-sampling Technique) for handling class imbalance.
Streamlit (Web Application) 🌐
Git & GitHub (Version Control & Deployment) 🚀
📂 Project Structure
bash
Copy
Edit
📂 Telcom_Churn_Prediction/
│── 📄 churn_app.py             # Streamlit app for churn prediction
│── 📄 train_notebook.py        # Jupyter notebook for training ML models
│── 📄 requirements.txt         # Required dependencies
│── 📄 README.md                # Project documentation
│── 📂 data/
│   ├── Telconnect_data.csv     # Telecom customer dataset
│── 📂 models/
│   ├── best_churn_model.pkl    # Trained ML model
🚀 How to Run the Project
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/Monjil999/Telcom_Churn.git
cd Telcom_Churn
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Streamlit App
bash
Copy
Edit
streamlit run churn_app.py
This will launch the web app in your browser, where you can input customer details and get churn predictions.

📊 Key Features
🔹 1. Churn Prediction
Predict customer churn probability based on input features.
Supports multiple machine learning models (XGBoost, LightGBM, RandomForest, Logistic Regression, SVM).
Uses SMOTE to handle data imbalance for better model performance.
🔹 2. Advanced Insights & Analytics
Customer Demographics & Churn Behavior: Explore trends based on gender, contract type, tenure, and payment methods.
Feature Importance: Identify key factors contributing to churn.
Survival Analysis: Understand how long customers stay before churning.
Correlation Heatmap: Visualizes relationships between features.
Churn vs. Monthly Charges & Tenure: Identify high-risk segments.
📈 Results & Findings
Customers with month-to-month contracts have a higher churn rate.
Customers using Electronic Checks as a payment method are more likely to churn.
Customers with low tenure (0-12 months) have the highest churn probability.
Lack of Online Security & Tech Support contributes to customer churn.
🔥 Future Improvements
Implement deep learning models for better accuracy.
Add automated recommendations for customer retention.
Deploy the model as a REST API for integration with business systems.
🤝 Contributing
Feel free to contribute by raising issues, suggesting improvements, or submitting pull requests.


