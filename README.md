# 📡 Telecom Customer Churn Prediction


A comprehensive machine learning solution for predicting customer churn in the telecom industry, complete with advanced analytics and a Streamlit web interface.

## 🔍 Overview
Customer churn prediction system that helps telecom companies:
- Identify at-risk customers using machine learning
- Provide actionable retention insights
- Offer real-time predictions via web interface
- Analyze customer behavior patterns

## 🎯 Objectives
- Predict customer churn probability with 85%+ accuracy
- Reduce customer acquisition costs by 30% through targeted retention
- Identify key churn drivers through explainable AI (XAI)
- Provide business-friendly dashboards for strategic decision making

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![ML](https://img.shields.io/badge/Machine_Learning-XGBoost%7CLightGBM%7CCatBoost-orange)
![Frontend](https://img.shields.io/badge/Web_Framework-Streamlit-ff69b4)

**Core Technologies:**
- **Data Analysis**: Pandas, NumPy, Seaborn, Matplotlib
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, CatBoost
- **ModelOps**: SHAP, MLflow, Hyperopt
- **Web App**: Streamlit, Plotly
- **Deployment**: Docker, Git, GitHub Actions

## 📂 Project Structure

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

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip package manager

### Installation
1. Clone repository:
```bash
git clone https://github.com/Monjil999/Telcom_Churn.git
cd Telcom_Churn
```
2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Run the Streamlit App
```bash
streamlit run churn_app.py
```
This will launch the web app in your browser, where you can input customer details and get churn predictions.

## 📈 Key Features

### 🔮 Churn Prediction
- Real-time probability estimates  
- Multiple model support (XGBoost, LightGBM, CatBoost, RF, SVM)  
- SMOTE-enhanced class balancing  
- Threshold optimization for business metrics  

### 📊 Advanced Analytics
- **Customer Segmentation**:
  - Tenure-based cohorts
  - Contract type analysis
  - Payment method trends
- **Feature Importance**:
  - SHAP value visualizations
  - Partial dependence plots
- **Survival Analysis**:
  - Customer lifetime estimation
  - Retention curve modeling

## 📉 Results & Findings
- **3.2× higher churn risk** for month-to-month contracts  
- **Electronic check users** churn 2.1× more frequently  
- **Optimal retention window**: 6-18 month tenure  
- **Service bundle adoption** reduces churn by 40%  

## 🔥 Future Roadmap
- [ ] **Deep learning integration** (LSTM networks)  
- [ ] **Automated retention recommendation engine**  
- [ ] **REST API deployment** with FastAPI  
- [ ] **Customer lifetime value prediction module**  
- [ ] **Cloud deployment** (AWS/GCP pipeline)
  
## 🤝 Contributing
Feel free to contribute by raising issues, suggesting improvements, or submitting pull requests.

