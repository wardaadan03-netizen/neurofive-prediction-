from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocessing import preprocess_data

# ----------------------------------------------------
# Project Paths
# ----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------
# Load Data
# ----------------------------------------------------

X, y = preprocess_data(DATA_PATH)

# ----------------------------------------------------
# Split Data
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------
# Identify Feature Types
# ----------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object", "bool"]
).columns

# ----------------------------------------------------
# Column Transformer
# ----------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# ----------------------------------------------------
# Pipeline
# ----------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

# ----------------------------------------------------
# Train Model
# ----------------------------------------------------

pipeline.fit(X_train, y_train)

# ----------------------------------------------------
# Predictions
# ----------------------------------------------------

y_pred = pipeline.predict(X_test)

# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------

print("=" * 50)
print("Pipeline Results")
print("=" * 50)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------------------------------
# Save Pipeline
# ----------------------------------------------------

joblib.dump(
    pipeline,
    MODELS_DIR / "churn_pipeline.pkl"
)

print("\nPipeline saved successfully!")
print(f"Location: {MODELS_DIR / 'churn_pipeline.pkl'}")