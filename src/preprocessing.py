import pandas as pd


def preprocess_data(path):

    df = pd.read_csv(path)

    # Remove customer ID
    df.drop(
        "customerID",
        axis=1,
        inplace=True
    )

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df.dropna(inplace=True)

    # Encode categorical variables

    df = pd.get_dummies(
        df,
        drop_first=True
    )


    X = df.drop(
        "Churn_Yes",
        axis=1
    )

    y = df["Churn_Yes"]


    return X, y