import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import joblib
import numpy as np
import pandas as pd


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


def runTrain():
    # Load the dataset
    data = pd.read_csv(
        r"ArduinoWineSniffer\src\data_analysis_for_NN\data_analysis_for_NN.csv",
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
    env_sensors = ["BMPTemperature",
                   "Pressure(Pa)", "DHTTemperature", "Humidity"]
    target_column = "Target"

    # Label encoding for the target
    label_encoder = LabelEncoder()

    # Separate features and target
    X_mq = data[feature_columns_mq]
    X_env = data[env_sensors]
    y = data[target_column]

    # Encode target labels (wine names) into numerical values
    y_encoded = label_encoder.fit_transform(y)

    # Determine the number of unique classes from the target column
    num_classes = len(np.unique(y_encoded))  # Automatically sets output_dim

    # Normalize both MQ sensor data and environmental sensor data
    scaler_mq = StandardScaler()
    X_mq_scaled = scaler_mq.fit_transform(X_mq)

    scaler_env = StandardScaler()
    X_env_scaled = scaler_env.fit_transform(X_env)

    # Apply PCA to both MQ and environmental sensor data
    mq_pca = PCA(n_components=3)  # Retain 70% variance
    X_mq_pca = mq_pca.fit_transform(X_mq_scaled)

    env_pca = PCA(n_components=1)  # Retain 70% variance
    X_env_pca = env_pca.fit_transform(X_env_scaled)

    # Save the PCA models for later use
    joblib.dump(
        mq_pca,
        r"ArduinoWineSniffer\PCAModel\pklfiles\mq_pca.pkl",
    )
    joblib.dump(
        env_pca,
        r"ArduinoWineSniffer\PCAModel\pklfiles\mq_pca.pkl",
    )

    # Concatenate MQ and environmental features after PCA
    X = np.hstack((X_mq_pca, X_env_pca))

    # Save the label encoder and scalers for future use
    joblib.dump(
        label_encoder,
        r"ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/pklfiles/label_encoder.pkl",
    )
    joblib.dump(
        scaler_mq,
        r"ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/pklfiles/scaler_mq.pkl",
    )
    joblib.dump(
        scaler_env,
        r"ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/pklfiles/scaler_env.pkl",
    )

    # Initialize the model with input size as the number of features from both MQ and env sensors after PCA
    # Combined feature size after PCA (MQ_PCA + env_PCA)
    input_dim = X.shape[1]

    # Initialize the model
    model = Multiclass(input_dim=input_dim, hidden_dim=64,
                       output_dim=num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Training loop for K-Fold Cross Validation
    num_epochs = 100
    highest_train_accuracy = 0.0  # Initialize variable to track highest accuracy
    highest_test_accuracy = 0.0  # Initialize variable to track highest accuracy

    fold_train_accuracies = []
    fold_test_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
        print(f"Fold {fold+1}/{skf.n_splits}")

        # Split into training and test sets based on K-fold indices
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_accuracies = []
        test_accuracies = []

        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted_train = torch.max(outputs, 1)
            correct_train = (predicted_train == y_train_tensor).sum().item()
            train_accuracy = correct_train / y_train_tensor.size(0)
            train_accuracies.append(train_accuracy)  # Append training accuracy

            # Calculate test accuracy
            with torch.no_grad():  # No gradient calculation during testing
                outputs_test = model(X_test_tensor)
                _, predicted_test = torch.max(outputs_test, 1)
                correct_test = (predicted_test == y_test_tensor).sum().item()
                test_accuracy = correct_test / y_test_tensor.size(0)
                test_accuracies.append(test_accuracy)  # Append test accuracy

            # Check for the highest accuracy
            if train_accuracy > highest_train_accuracy:
                highest_train_accuracy = train_accuracy
            if test_accuracy > highest_test_accuracy:
                highest_test_accuracy = test_accuracy

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, "
                    f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
                )

        fold_train_accuracies.append(train_accuracies)
        fold_test_accuracies.append(test_accuracies)

    # Print the highest accuracy achieved during training
    print(f"Highest Train Accuracy: {highest_train_accuracy:.4f}")
    print(f"Highest Test  Accuracy: {highest_test_accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pth")


if __name__ == "__main__":
    runTrain()
