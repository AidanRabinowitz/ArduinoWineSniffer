import pandas as pd
import joblib


def load_model_and_predict(feature_data_path):
    # Load the saved logistic regression model and label encoder
    model = joblib.load(
        "ML/WineSnifferDeepNetworkModel/PCAModel/logRegressionPKLFiles/logistic_regression_model.pkl"
    )
    label_encoder = joblib.load(
        "ML/WineSnifferDeepNetworkModel/PCAModel/logRegressionPKLFiles/label_encoder.pkl"
    )

    # Load the new feature data from a CSV file
    feature_data = pd.read_csv(feature_data_path, header=0)

    # Ensure the columns match the original feature columns
    feature_columns = [
        "MQ135",
        "MQ2",
        "MQ3",
        "MQ4",
        "MQ5",
        "MQ6",
        "MQ7",
        "MQ8",
        "MQ9",
        "BMPTemperature",
        "Pressure(Pa)",
        "DHTTemperature",
        "Humidity",
    ]

    # Extract the features (unlabeled data)
    X_new = feature_data[feature_columns].values

    # Use the loaded model to predict the labels
    predictions = model.predict(X_new)

    # Decode the predicted labels back to their original form
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Add the predictions to the DataFrame
    feature_data["Predicted_Label"] = predicted_labels

    # Print or return the DataFrame with predictions
    print(feature_data)
    return feature_data


if __name__ == "__main__":
    # Replace with the path to your CSV file with unlabeled feature data
    test_file_path = "ML/WineCSVs/Test/Test2309/TallHorseTest.csv"
    load_model_and_predict(test_file_path)
