import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


class Multiclass(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=6):
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
        r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/src/data_analysis_for_NN/data_analysis_for_NN.csv",
        header=0,
    )

    # Feature columns for MQ and environmental sensors
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
    target_column = "Target"

    # Separate features and target
    X = data[feature_columns].values
    y = data[target_column].values

    # Label encoding for the target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save the label encoder
    joblib.dump(
        label_encoder,
        r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/pklfiles/label_encoder.pkl",
    )

    # Determine the number of unique classes
    num_classes = len(np.unique(y_encoded))

    # Initialize the model
    input_dim = X.shape[1]  # 13 features
    model = Multiclass(input_dim=input_dim, hidden_dim=64, output_dim=num_classes)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # K-Fold Cross Validation (no stratification)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    num_epochs = 1000
    highest_train_accuracy = 0.0
    highest_test_accuracy = 0.0

    fold_train_accuracies = []
    fold_test_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{kf.n_splits}")

        # Re-initialize model and optimizer inside each fold
        model = Multiclass(input_dim=input_dim, hidden_dim=64, output_dim=num_classes)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Split into train and test sets
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Scale the data within the fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
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

            # Training accuracy
            _, predicted_train = torch.max(outputs, 1)
            correct_train = (predicted_train == y_train_tensor).sum().item()
            train_accuracy = correct_train / y_train_tensor.size(0)
            train_accuracies.append(train_accuracy)

            # Test accuracy
            with torch.no_grad():
                outputs_test = model(X_test_tensor)
                _, predicted_test = torch.max(outputs_test, 1)
                correct_test = (predicted_test == y_test_tensor).sum().item()
                test_accuracy = correct_test / y_test_tensor.size(0)
                test_accuracies.append(test_accuracy)

            # Track highest accuracy
            highest_train_accuracy = max(highest_train_accuracy, train_accuracy)
            highest_test_accuracy = max(highest_test_accuracy, test_accuracy)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, "
                    f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
                )

        fold_train_accuracies.append(train_accuracies)
        fold_test_accuracies.append(test_accuracies)

        # Print classification report
        print(f"\nClassification Report (Fold {fold + 1}):")
        print(classification_report(y_test, predicted_test))

    print(f"Highest Train Accuracy: {highest_train_accuracy:.4f}")
    print(f"Highest Test Accuracy: {highest_test_accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pth")

    # Plot train and test accuracy per fold
    plt.figure(figsize=(12, 8))
    for fold in range(kf.n_splits):
        plt.plot(fold_train_accuracies[fold], label=f"Fold {fold + 1} Train Accuracy")
        plt.plot(
            fold_test_accuracies[fold],
            label=f"Fold {fold + 1} Test Accuracy",
            linestyle="--",
        )
    plt.title("Train and Test Accuracy per Fold", fontsize=30)
    plt.xlabel("Epoch (x1000)", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()


if __name__ == "__main__":
    runTrain()
