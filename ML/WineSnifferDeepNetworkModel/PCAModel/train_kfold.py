from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import joblib
import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv(
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/src/data_analysis_for_NN/data_analysis_for_NN.csv",
    header=0,
)


# Define your MLP model
class Multiclass(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super(Multiclass, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Feature columns for MQ sensors and environmental sensors
feature_columns_mq = ["MQ135", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9"]
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

# Apply PCA to both MQ and environmental sensor data
mq_pca = PCA(n_components=0.70)  # Retain 70% variance
X_mq_pca = mq_pca.fit_transform(X_mq_scaled)

env_pca = PCA(n_components=0.70)  # Retain 70% variance
X_env_pca = env_pca.fit_transform(X_env_scaled)

# Save the PCA models for later use
joblib.dump(
    mq_pca,
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/pklfiles/mq_pca.pkl",
)
joblib.dump(
    env_pca,
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/pklfiles/env_pca.pkl",
)

# Concatenate the MQ and environmental features
X = np.hstack((X_mq_pca, X_env_pca))

# Save the label encoder and scalers for future use
joblib.dump(
    label_encoder,
    "C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/label_encoder.pkl",
)
joblib.dump(
    scaler_mq,
    "C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/scaler_mq.pkl",
)
joblib.dump(
    scaler_env,
    "C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/scaler_env.pkl",
)

# Initialize the model with input size as the number of features from both MQ and env sensors after PCA
input_dim = X.shape[1]  # Combined feature size after PCA (MQ_PCA + env_PCA)
model = Multiclass(
    input_dim=input_dim, hidden_dim=64, output_dim=len(np.unique(y_encoded))
)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize StratifiedKFold cross-validation
kf = StratifiedKFold(n_splits=5)

# Track accuracy for plotting
train_accuracies = []
test_accuracies = []
highest_train_accuracy = 0.0  # Initialize variable to track highest train accuracy
highest_test_accuracy = 0.0  # Initialize variable to track highest test accuracy

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y_encoded)):
    # Split data into training and testing sets for the current fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Training loop for each fold
    num_epochs = 1000
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
        train_accuracies.append(train_accuracy)

        # Calculate test accuracy
        with torch.no_grad():
            outputs_test = model(X_test_tensor)
            _, predicted_test = torch.max(outputs_test, 1)
            correct_test = (predicted_test == y_test_tensor).sum().item()
            test_accuracy = correct_test / y_test_tensor.size(0)
            test_accuracies.append(test_accuracy)

        # Track highest accuracy
        if train_accuracy > highest_train_accuracy:
            highest_train_accuracy = train_accuracy
        if test_accuracy > highest_test_accuracy:
            highest_test_accuracy = test_accuracy

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Fold [{fold + 1}], Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
            )

# Print the highest accuracy achieved during training
print(f"Highest Train Accuracy: {highest_train_accuracy:.4f}")
print(f"Highest Test Accuracy: {highest_test_accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), "trained_model.pth")

# Plotting the train and test accuracy across epochs
plt.figure(figsize=(10, 5))
plt.plot(
    range(len(train_accuracies)), train_accuracies, label="Train Accuracy", color="blue"
)
plt.plot(
    range(len(test_accuracies)), test_accuracies, label="Test Accuracy", color="orange"
)
plt.title("Train and Test Accuracy (Stratified K-Fold)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
