# Heart-Disease-Prediction
Objective

The goal of this task is to build a machine learning model that can predict whether a person is at risk of heart disease based on clinical and demographic features. The model uses the UCI Heart Disease Dataset and performs exploratory data analysis, cleaning, preprocessing, training, evaluation, and feature importance analysis.

Dataset

Name: heart_disease_uci.csv

Rows: 920

Features: 15 original columns + 1 derived target

Target column:

Original label: num (0–4 severity levels)

Converted to binary:

0 → No heart disease

1 → Heart disease present (num > 0)

 Steps Performed
1. Dataset Loading

Loaded data using pandas.

Displayed:

Dataset shape

Column names

First 5 rows

Data types (.info())

Summary statistics (.describe())

Missing values per column

2. Data Cleaning

Removed irrelevant column id and original severity column num.

Created a binary target variable:
target = 1 if num > 0 else 0

Handled missing values:

Numeric → filled with median

Categorical → filled with mode

Converted categorical columns into numerical form using one-hot encoding.

3. Exploratory Data Analysis

Target class distribution (countplot)

Correlation heatmap for numeric features

Age distribution by heart disease status

Identification of key correlated features
(e.g., chest pain type, slope, oldpeak)

4. Feature Engineering

One-hot encoded:

Gender

dataset

chest pain (cp)

fasting blood sugar (fbs)

resting ECG (restecg)

exercise-induced angina (exang)

slope

thal

Final feature matrix (X) contains both numeric and encoded categorical features.

5. Model Training

Two models were trained:

Model	Description	Scaling Needed
Logistic Regression	Baseline linear classifier	Yes
Decision Tree Classifier	Non-linear, interpretable model	No

Scaling performed using StandardScaler (only for Logistic Regression).

6. Model Evaluation

Each model was evaluated using:

Train accuracy

Test accuracy

Confusion matrix

ROC Curve

ROC-AUC Score

These metrics show how well the model separates heart-disease-positive vs negative cases.

7. Feature Importance

Using the Decision Tree model, top contributing features were identified. These often included:

cp (chest pain type)

oldpeak (ST depression induced by exercise)

thal (thallium test result)

slope of ST segment

trestbps (resting blood pressure)

chol (cholesterol)

This helps understand which clinical features are most important for risk prediction.

 Key Insights

Logistic Regression provided stable general performance with good ROC-AUC.

Decision Tree offered interpretability but slightly higher risk of overfitting.

Chest pain type, ST depression (oldpeak), and thalassemia category were strong indicators of heart disease.

The dataset contained missing values in many clinical fields; proper imputation was critical.
