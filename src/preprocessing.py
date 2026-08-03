import pandas as pd


def preprocess_data(path):

    # Load dataset
    df = pd.read_csv(path)

    # Remove customer ID
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove missing values
    df.dropna(inplace=True)

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    # Feature 1: Number of subscribed services
    df["TotalServices"] = (
        df[service_cols] != "No"
    ).sum(axis=1)

    # Feature 2: Average monthly spend
    df["AvgMonthlySpend"] = (
        df["TotalCharges"] /
        (df["tenure"] + 1)
    )

    # Target variable
    y = df["Churn"]

    # Features
    X = df.drop("Churn", axis=1)

    return X, y