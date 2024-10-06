import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


# Define the neural network architecture (must match the training architecture)
class Multiclass(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=6):
        super(Multiclass, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def runTest(test_csv_path, wine_tested):
    # Load the test data
    test_data = pd.read_csv(test_csv_path)

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

    # Load the saved PCA models for both MQ and environmental data
    mq_pca = joblib.load("ML/WineSnifferDeepNetworkModel/PCAModel/pklfiles/mq_pca.pkl")
    env_pca = joblib.load(
        "ML/WineSnifferDeepNetworkModel/PCAModel/pklfiles/env_pca.pkl"
    )

    # Load the scalers
    scaler_mq = joblib.load(
        "ML/WineSnifferDeepNetworkModel/PCAModel/pklfiles/scaler_mq.pkl"
    )
    scaler_env = joblib.load(
        "ML/WineSnifferDeepNetworkModel/PCAModel/pklfiles/scaler_env.pkl"
    )

    # Load the label encoder
    label_encoder = joblib.load(
        "ML/WineSnifferDeepNetworkModel/PCAModel/pklfiles/label_encoder.pkl"
    )

    # Separate features (MQ and environmental)
    X_test_mq = test_data[feature_columns_mq]
    X_test_env = test_data[env_sensors]

    # Normalize the test data using the same scalers
    X_test_mq_scaled = scaler_mq.transform(X_test_mq)
    X_test_env_scaled = scaler_env.transform(X_test_env)

    # Apply PCA transformation on the test data
    X_test_mq_pca = mq_pca.transform(X_test_mq_scaled)
    X_test_env_pca = env_pca.transform(X_test_env_scaled)

    # Concatenate MQ and environmental features after PCA
    X_test_pca = np.hstack((X_test_mq_pca, X_test_env_pca))

    # Convert to PyTorch tensor
    X_test_pca_tensor = torch.tensor(X_test_pca, dtype=torch.float32)

    # Load the trained model
    input_dim = X_test_pca.shape[1]

    # Dynamically set output_dim based on the number of unique labels in the dataset
    num_classes = len(label_encoder.classes_)  # Get the number of unique labels
    model = Multiclass(input_dim=input_dim, hidden_dim=64, output_dim=num_classes)

    # Load the saved model state
    model.load_state_dict(torch.load("trained_model.pth"))
    model.eval()

    # Predict the wine types from the model
    with torch.no_grad():
        y_pred = model(X_test_pca_tensor)
        predicted_classes = torch.argmax(y_pred, dim=1).numpy()

    # Decode the predicted labels
    predicted_class_names = label_encoder.inverse_transform(predicted_classes)

    # Print the predicted labels and count matches with wine_tested
    print("Predicted Labels:")
    correct_count = 0
    for idx, label in enumerate(predicted_class_names):
        print(f"Sample {idx + 1}: {label}")
        if wine_tested.lower() in label.lower():
            correct_count += 1

    # Calculate accuracy as a percentage
    accuracy_label = (correct_count / len(predicted_class_names)) * 100
    print(f"Accuracy of matching '{wine_tested}' in labels: {accuracy_label:.2f}%")

    return accuracy_label


if __name__ == "__main__":
    # Provide the path to the test CSV file
    test_csv_path = r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineCSVs/Test/ControlTests/0410/TallHorse.csv"  # Update this path

    # Ask for the wine being tested
    wine_tested = "TallHorse"

    # Run the test
    accuracy = runTest(test_csv_path, wine_tested)
