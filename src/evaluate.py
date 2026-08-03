from sklearn.metrics import classification_report


def evaluate_model(model,X_test,y_test):

    prediction = model.predict(X_test)

    print(
        classification_report(
            y_test,
            prediction
        )
    )