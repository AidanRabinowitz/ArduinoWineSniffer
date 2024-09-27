import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from collections import Counter


# Load the model
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(9, 32)  # Adjust input size to your feature count
        self.bn1 = nn.BatchNorm1d(32)
        self.hidden2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.act = nn.ReLU()
        self.output = nn.Linear(16, 6)  # Adjust output size to your number of classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.act(self.bn1(self.hidden1(x)))
        x = self.dropout(self.act(self.bn2(self.hidden2(x))))
        x = self.output(x)
        return x


# Load data
train_data = pd.read_csv(
    "ML/WineCSVs/Train/SixWinesData/SixWines2309(25degEnvTemp)_cleaned.csv"
)
test_data = pd.read_csv(
    "ML/WineCSVs/Test/Test2309/SophieTest2309(25degEnvTemp).csv"
)  # Adjust path as necessary

# Extract feature columns
feature_columns = ["MQ135", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9"]
X_test = test_data[feature_columns]

# Load the trained model and the label encoder
model = Multiclass()
model.load_state_dict(torch.load("wine_model.pth"))
model.eval()

label_encoder = joblib.load("label_encoder.pkl")
class_names = label_encoder.classes_

# Preprocess test data (if needed)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

# Predict
with torch.no_grad():
    y_pred = model(X_test_tensor)
    predicted_classes = torch.argmax(y_pred, dim=1).numpy()

# Map predicted indices to class names
predicted_class_names = label_encoder.inverse_transform(predicted_classes)

# Print results
for idx, class_name in enumerate(predicted_class_names):
    print(f"Sample {idx + 1}: Classified as {class_name}")

# Calculate modal classification
modal_class = Counter(predicted_class_names).most_common(1)[0][0]
print(f"\nModal classification wine name: {modal_class}")
