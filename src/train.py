from pathlib import Path

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from preprocessing import preprocess_data


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Models directory
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# Load and preprocess data
X, y = preprocess_data(DATA_PATH)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale data for Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------
# Decision Tree
# -------------------------

decision_tree = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)

decision_tree.fit(X_train, y_train)

dt_predictions = decision_tree.predict(X_test)

print("=" * 50)
print("Decision Tree Results")
print("=" * 50)

print(classification_report(y_test, dt_predictions))
print("Accuracy:", accuracy_score(y_test, dt_predictions))


# -------------------------
# Logistic Regression
# -------------------------

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

logistic_model.fit(X_train_scaled, y_train)

lr_predictions = logistic_model.predict(X_test_scaled)

print("\n" + "=" * 50)
print("Logistic Regression Results")
print("=" * 50)

print(classification_report(y_test, lr_predictions))
print("Accuracy:", accuracy_score(y_test, lr_predictions))


# -------------------------
# Save Models
# -------------------------

joblib.dump(
    decision_tree,
    MODELS_DIR / "decision_tree_model.pkl"
)

joblib.dump(
    logistic_model,
    MODELS_DIR / "logistic_regression_model.pkl"
)

joblib.dump(
    scaler,
    MODELS_DIR / "scaler.pkl"
)

print("\nModels saved successfully!")
print(f"Location: {MODELS_DIR}")