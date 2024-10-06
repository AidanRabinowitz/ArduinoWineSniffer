from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
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
    def __init__(
        self, input_dim, hidden_dim=64, output_dim=3
    ):  # Adjust `output_dim` as per your number of classes
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
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/pklfilesmw_pca.pkl",
)
joblib.dump(
    env_pca,
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/pklfiles/env_pca.pkl",
)

# Split data into train and test sets for both MQ and environmental features
X_mq_train, X_mq_test, X_env_train, X_env_test, y_train, y_test = train_test_split(
    X_mq_pca, X_env_pca, y_encoded, test_size=0.2, random_state=42
)

# Concatenate MQ and environmental features after splitting
X_train = np.hstack((X_mq_train, X_env_train))
X_test = np.hstack((X_mq_test, X_env_test))

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

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
input_dim = X_train.shape[1]  # Combined feature size after PCA (MQ_PCA + env_PCA)
model = Multiclass(
    input_dim=input_dim, hidden_dim=64, output_dim=len(np.unique(y_encoded))
)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
highest_accuracy = 0.0  # Initialize variable to track highest accuracy

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
    if train_accuracy > highest_accuracy:
        highest_accuracy = train_accuracy

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, "
            f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )

# Print the highest accuracy achieved during training
print(f"Highest Train Accuracy: {highest_accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), "trained_model.pth")

# Plotting the train and test accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_accuracies, label="Train Accuracy", color="blue")
plt.plot(range(num_epochs), test_accuracies, label="Test Accuracy", color="orange")
plt.title("Train and Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
