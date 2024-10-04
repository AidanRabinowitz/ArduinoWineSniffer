import pandas as pd
import torch
import torch.nn as nn
import joblib
from collections import Counter
import sys
from train import Multiclass

# Load the wine name from command-line arguments
wine_tested = "Moscato"

# Load the test data
test_data = pd.read_csv(
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineCSVs/Test/ControlTests/3009/Moscato.csv"
)

num_wines_in_dataset = 11
feature_columns = ["MQ135", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9"]
X_test = test_data[feature_columns]

# Load the PCA model
pca = joblib.load(r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/pca_model.pkl")
X_test_pca = pca.transform(X_test)


model = Multiclass()
model.load_state_dict(
    torch.load(r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/pca_wine_model.pth")
)
model.eval()

# Load the label encoder
label_encoder = joblib.load(
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/label_encoder.pkl"
)

# Predict
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)
with torch.no_grad():
    y_pred = model(X_test_tensor)
    predicted_classes = torch.argmax(y_pred, dim=1).numpy()

# Map predicted indices to class names
predicted_class_names = label_encoder.inverse_transform(predicted_classes)

# Calculate Label Accuracy using substring match
total_samples = len(predicted_class_names)
count_correct = sum(
    1 for name in predicted_class_names if wine_tested.lower() in name.lower()
)
label_accuracy = (count_correct / total_samples) * 100 if total_samples > 0 else 0

print(predicted_class_names)
# Print results
print(f"Label Accuracy for '{wine_tested}': {label_accuracy:.2f}%")

# Save the accuracy to a file (or handle it as needed)
with open("label_accuracy.txt", "w") as f:
    f.write(str(label_accuracy))
