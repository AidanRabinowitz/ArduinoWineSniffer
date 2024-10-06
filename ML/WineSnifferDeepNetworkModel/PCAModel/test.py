import pandas as pd
from sklearn.calibration import LabelEncoder
import torch
import torch.nn as nn
import joblib
from collections import Counter
from train_kfold import Multiclass
import numpy as np

# Define the wine being tested (you can hard-code or pass as argument)
wine_tested = "Moscato"

# Load the test data (assuming no labels in this dataset)
test_data = pd.read_csv(
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineCSVs/Test/ControlTests/3009/Moscato.csv"
)
# Load the dataset
train_data = pd.read_csv(
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/src/data_analysis_for_NN/data_analysis_for_NN.csv",
    header=0,
)
y = train_data["Target"]
label_encoder = LabelEncoder()

# Encode target labels (wine names) into numerical values
y_encoded = label_encoder.fit_transform(y)

# **Determine the number of unique classes from the target column**
num_classes = len(np.unique(y_encoded))  # Automatically sets output_dim

feature_columns_mq = ["MQ135", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9"]
env_sensors = ["BMPTemperature", "Pressure(Pa)", "DHTTemperature", "Humidity"]

# Separate features (MQ and environmental)
X_test_mq = test_data[feature_columns_mq]
X_test_env = test_data[env_sensors]

# Load the saved PCA models for both MQ and environmental data
mq_pca = joblib.load(
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/pklfiles/mq_pca.pkl"
)
env_pca = joblib.load(
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/pklfiles/env_pca.pkl"
)

# Apply PCA transformation on the test data
X_test_mq_pca = mq_pca.transform(X_test_mq)
X_test_env_pca = env_pca.transform(X_test_env)

# Concatenate MQ and environmental features after PCA
X_test_pca = torch.tensor(
    np.hstack((X_test_mq_pca, X_test_env_pca)), dtype=torch.float32
)

# Load the trained model
input_dim = X_test_pca.shape[1]
model = Multiclass(input_dim=input_dim, hidden_dim=64, output_dim=num_classes)

model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# Load the label encoder to decode predictions
label_encoder = joblib.load(
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/label_encoder.pkl"
)

# Predict the wine types from the model
with torch.no_grad():
    y_pred = model(X_test_pca)
    predicted_classes = torch.argmax(y_pred, dim=1).numpy()

# Decode the predicted labels
predicted_class_names = label_encoder.inverse_transform(predicted_classes)

# Compare predicted labels with the known wine being tested (wine_tested)
total_samples = len(predicted_class_names)
count_correct = sum(
    1 for name in predicted_class_names if wine_tested.lower() in name.lower()
)
label_accuracy = (count_correct / total_samples) * 100 if total_samples > 0 else 0

# Print the predicted labels and accuracy
print(f"Predicted Labels: {predicted_class_names}")
print(f"Label Accuracy for '{wine_tested}': {label_accuracy:.2f}%")

# Save the accuracy to a file
with open("label_accuracy.txt", "w") as f:
    f.write(f"Label Accuracy for '{wine_tested}': {label_accuracy:.2f}%")
