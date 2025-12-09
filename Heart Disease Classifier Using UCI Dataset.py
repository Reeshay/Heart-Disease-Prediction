#!/usr/bin/env python
# coding: utf-8

# In[3]:


#================================================
# TASK 3: HEART DISEASE PREDICTION (COMPLETE)
# ================================================

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# -------------------------------------------
# Step 1 — Load Dataset
# -------------------------------------------

csv_path = r"C:\Users\HP\OneDrive\Desktop\Ai Internship\heart Disease Dataset\heart_disease_uci.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ File not found at: {csv_path}")

df = pd.read_csv(csv_path)
print("✅ Dataset Loaded Successfully")
print("Shape:", df.shape)
print("\nColumn names:", df.columns.tolist())

display(df.head())

print("\nInfo:")
print(df.info())

print("\nSummary Statistics (numeric columns):")
display(df.describe())

print("\nMissing Values per column:")
print(df.isna().sum())

# -------------------------------------------
# Step 2 — Create Binary Target from `num`
# -------------------------------------------
# num: 0 = no disease, 1–4 = disease

if "num" not in df.columns:
    raise KeyError("❌ Expected column 'num' not found in dataset.")

df["target"] = (df["num"] > 0).astype(int)

print("\nValue counts of target (0 = no disease, 1 = disease):")
print(df["target"].value_counts())

# Drop ID & original severity column from features
cols_to_drop = []
if "id" in df.columns:
    cols_to_drop.append("id")
cols_to_drop.append("num")

df = df.drop(columns=cols_to_drop)

print("\nColumns after dropping id & num:")
print(df.columns.tolist())

# -------------------------------------------
# Step 3 — EDA (Basic Plots)
# -------------------------------------------

plt.figure(figsize=(5,4))
sns.countplot(x="target", data=df)
plt.title("Distribution of Target (Heart Disease)")
plt.xlabel("Target (0 = No disease, 1 = Disease)")
plt.ylabel("Count")
plt.show()

# Correlation heatmap for numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# Optional: histogram of age by target
plt.figure(figsize=(6,4))
sns.histplot(data=df, x="age", hue="target", kde=True, bins=20)
plt.title("Age Distribution by Heart Disease Status")
plt.show()

# -------------------------------------------
# Step 4 — Handle Missing Values
# -------------------------------------------

print("\nMissing values BEFORE filling:")
print(df.isna().sum())

numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

# Fill numeric NaNs with median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical NaNs with mode
for col in categorical_cols:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values AFTER filling:")
print(df.isna().sum())

# -------------------------------------------
# Step 5 — Separate Features and Target
# -------------------------------------------

target_col = "target"
X = df.drop(columns=[target_col])
y = df[target_col]

# One-hot encode categorical variables (Gender, dataset, cp, fbs, restecg, exang, slope, thal, etc.)
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
print("\nCategorical columns detected:", list(non_numeric_cols))

if len(non_numeric_cols) > 0:
    X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)

print("\nFinal feature shape:", X.shape)

# -------------------------------------------
# Step 6 — Train/Test Split
# -------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# -------------------------------------------
# Step 7 — Scaling for Logistic Regression
# -------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -------------------------------------------
# Step 8 — Train Models
# -------------------------------------------

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

# -------------------------------------------
# Step 9 — Evaluation Function
# -------------------------------------------

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    print(f"\n==================== {name} ====================")
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    
    print("Train Accuracy:", accuracy_score(y_tr, y_pred_tr))
    print("Test Accuracy :", accuracy_score(y_te, y_pred_te))
    
    cm = confusion_matrix(y_te, y_pred_te)
    print("\nConfusion Matrix (test):\n", cm)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_proba)
        print("ROC-AUC:", auc)
        
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        plt.plot([0,1], [0,1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.grid(True)
        plt.show()

# -------------------------------------------
# Step 10 — Model Evaluation
# -------------------------------------------

# Logistic Regression (scaled data)
evaluate_model(
    "Logistic Regression",
    log_reg,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

# Decision Tree (raw data)
evaluate_model(
    "Decision Tree",
    dt,
    X_train,
    X_test,
    y_train,
    y_test
)

# -------------------------------------------
# Step 11 — Feature Importance (Decision Tree)
# -------------------------------------------

importances = pd.Series(dt.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print("\nTop 10 important features (Decision Tree):")
display(importances.head(10))

plt.figure(figsize=(8,5))
importances.head(10).plot(kind="bar")
plt.title("Top 10 Feature Importances (Decision Tree)")
plt.ylabel("Importance")
plt.show()

print("\n✅ TASK 2 COMPLETED SUCCESSFULLY")
print("Now add markdown cells for:")
print("- EDA observations")
print("- Which model performs better and why")
print("- Interpretation of confusion matrix and ROC-AUC")
print("- Most important features and medical meaning")


# In[ ]:




