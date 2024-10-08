import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


def runTrain():
    # Load the dataset
    data = pd.read_csv(
        r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/src/data_analysis_for_NN/data_analysis_for_NN.csv",
        header=0,
    )

    # Feature columns for MQ sensors and environmental sensors
    feature_columns_mq = [
        "MQ135",
        "MQ2",
        "MQ3",
        "MQ4",
        "MQ5",
        "MQ6",
        "MQ7",
        "MQ8",
        "MQ9",
    ]
    env_sensors = ["BMPTemperature", "Pressure(Pa)", "DHTTemperature", "Humidity"]
    target_column = "Target"

    # Label encoding for the target
    label_encoder = LabelEncoder()

    # Separate features and target
    X_mq = data[feature_columns_mq]
    X_env = data[env_sensors]
    y = data[target_column]

    # Encode target labels (wine names) into numerical values
    y_encoded = label_encoder.fit_transform(y)

    # Normalize both MQ sensor data and environmental sensor data
    scaler_mq = StandardScaler()
    X_mq_scaled = scaler_mq.fit_transform(X_mq)

    scaler_env = StandardScaler()
    X_env_scaled = scaler_env.fit_transform(X_env)

    # Concatenate MQ and environmental features after scaling
    X = np.hstack((X_mq_scaled, X_env_scaled))

    # Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize the linear regression model
    model = LinearRegression()

    # Training loop for K-Fold Cross Validation
    fold_train_accuracies = []
    fold_test_accuracies = []
    num_epochs = 100

    # Split into training and test sets based on K-fold indices
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    model.fit(X_train, y_train)

    # Make predictions on the training set and round to nearest integer
    y_train_pred = np.round(model.predict(X_train))
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Make predictions on the test set and round to nearest integer
    y_test_pred = np.round(model.predict(X_test))
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Test Accuracy", test_accuracy)


if __name__ == "__main__":
    runTrain()
