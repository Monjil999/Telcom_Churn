import numpy as np
import pandas as pd
import pickle
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# ML models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import os

# Load Data
def load_data():
    df = pd.read_csv("Telconnect_data.csv")

    # Rename columns
    df.rename(columns={
        "account_tenure": "tenure",
        "Has_Partner": "Partner",
        "Has_Dependents": "Dependents"
    }, inplace=True)

    # Drop 'customerID' if present
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Convert 'Churn' to binary (1/0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Convert 'TotalCharges' to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

    return df

# Preprocessing Function
def preprocess_data(df):
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    cat_cols = [c for c in df.columns if c not in num_cols + ["Churn"]]

    df_enc = df.copy()

    # Label encoding categorical features
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    X = df_enc.drop("Churn", axis=1)
    y = df_enc["Churn"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE for balancing classes with a small percentage to avoid overfitting
    sm = SMOTE(sampling_strategy=0.7, random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    # Scale numerical features
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, scaler, X_train.columns.tolist(), num_cols

# Hyperparameter tuning using GridSearchCV
def tune_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Train Multiple Models
def train_models(X_train, y_train):
    models = {
        "XGBoost": (XGBClassifier(random_state=42), {
            "n_estimators": [500, 700],
            "learning_rate": [0.05, 0.1],
            "max_depth": [5, 7]
        }),
        "LightGBM": (LGBMClassifier(random_state=42), {
            "n_estimators": [500, 700],
            "learning_rate": [0.05, 0.1],
            "max_depth": [5, 7]
        }),
        "RandomForest": (RandomForestClassifier(random_state=42), {
            "n_estimators": [300, 500, 700],
            "max_depth": [None, 10, 15]
        }),
        "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=42), {
            "C": [0.1, 1.0, 10.0]
        }),
        "SVM": (SVC(probability=True, kernel="rbf", random_state=42), {
            "C": [0.5, 1.0, 2.0]
        })
    }

    trained_models = {}
    for name, (model, param_grid) in models.items():
        print(f"Tuning {name}...")
        best_model = tune_model(model, param_grid, X_train, y_train)
        trained_models[name] = best_model

    return trained_models

# Evaluate Models
def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability for AUC
        y_pred = model.predict(X_test)  # Predictions for other metrics

        # Compute Metrics
        auc = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Store results
        results[name] = {
            "AUC": auc,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Confusion_Matrix": conf_matrix
        }

        # Print results for each model
        print(f"\n📌 {name} Model Metrics:")
        print(f"🔹 AUC Score: {auc:.4f}")
        print(f"🔹 Accuracy: {accuracy:.4f}")
        print(f"🔹 Precision: {precision:.4f}")
        print(f"🔹 Recall: {recall:.4f}")
        print(f"🔹 F1-score: {f1:.4f}")
        print(f"🔹 Confusion Matrix:\n{conf_matrix}")

    return results

# Save Best Model
def save_best_model(models, results, scaler, feature_cols, num_cols):
    best_model_name = max(results, key=lambda k: results[k]["AUC"])
    best_model = models[best_model_name]

    save_dict = {
        "model": best_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "num_cols": num_cols
    }

    with open("best_churn_model.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    print(f"Best model '{best_model_name}' saved!")

# Main Execution
def main():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_cols, num_cols = preprocess_data(df)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    save_best_model(models, results, scaler, feature_cols, num_cols)

if __name__ == "__main__":
    main()
