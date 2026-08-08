# Telco Customer Churn Prediction Using Machine Learning

## Project Overview

Customer churn prediction helps companies identify customers who are likely to leave their services.

This project analyzes telecom customer data and builds machine learning models to predict customer churn.

The goal is to help businesses take proactive retention actions before customers leave.

---

## Dataset

**Dataset:**
Telco Customer Churn Dataset

**Source:**
Kaggle - IBM Telco Customer Churn Dataset

The dataset contains:

* Customer demographics
* Internet services
* Contract information
* Monthly charges
* Tenure
* Churn status

---

## Project Workflow

1. Data Loading
2. Exploratory Data Analysis
3. Data Cleaning
4. Feature Engineering
5. Categorical Encoding
6. Model Training
7. Model Evaluation
8. Feature Importance Analysis
9. Business Recommendations

---

## Exploratory Data Analysis

Key analysis performed:

* Churn distribution
* Contract type impact
* Tenure analysis
* Monthly charges analysis
* Customer service relationship analysis

---

## Machine Learning Models

### 1. Decision Tree Classifier

Advantages:

* Easy interpretation
* Provides feature importance
* Useful for business explanations

### 2. Logistic Regression

Advantages:

* Strong baseline classification model
* Provides probability-based predictions

---

## Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

Accuracy alone can be misleading because churn datasets are usually imbalanced.

---

## Handling Imbalanced Data

### Class Imbalance

The target variable in the Telco Customer Churn dataset is imbalanced, with more customers belonging to the non-churn class than the churn class.

The class distribution was analyzed and visualized using a bar chart to understand the imbalance between the two classes.

### Baseline Model

The Decision Tree and Logistic Regression models were first trained using the original imbalanced training data.

Their performance was evaluated using:

* Precision
* Recall
* F1-score
* Accuracy

### SMOTE

To address the class imbalance, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training data.

SMOTE generates synthetic examples for the minority class, helping the model learn patterns from both classes more effectively.

SMOTE was applied only to the training data. The test data was kept unchanged to ensure that evaluation was performed on the original class distribution.

### Before vs After SMOTE

The models were retrained using the balanced training data and their performance was compared with the baseline models.

The comparison focuses on:

* Precision
* Recall
* F1-score

The exact results are reported in the notebook's **Before vs After SMOTE Comparison** section.

### Why Accuracy Can Be Misleading

Accuracy can be misleading when working with imbalanced data because a model can achieve high accuracy by primarily predicting the majority class while failing to correctly identify customers in the minority class.

For churn prediction, Precision, Recall, and F1-score provide a more informative evaluation of the model's performance, particularly its ability to identify customers who are likely to churn.

---

## Important Churn Factors

The Decision Tree model identifies important features such as:

* Contract type
* Customer tenure
* Monthly charges

---

## Business Impact

This solution can help telecom companies:

* Identify high-risk customers
* Create targeted retention campaigns
* Reduce customer loss
* Improve customer satisfaction

---

## Technologies Used

Python

**Libraries:**

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Joblib

---

## Author

**Warda Adan**

Machine Learning / AI Engineer
