import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from collections import Counter

# Train data
data = pd.read_csv(
    "ML/WineCSVs/Train/SixWinesData/SixWines2309(25degEnvTemp)_cleaned.csv",
    header=0,
)

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
    # "BMPTemperature",
    # "Pressure(Pa)",
    # "DHTTemperature",
    # "Humidity",
]
target_column = "Target"
# For adjusted CSV (environmental control)
X = data[feature_columns]
y = data[[target_column]]
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(y)
y = ohe.transform(y)
y = torch.tensor(y, dtype=torch.float32)
num_outputs = y.shape[1]  # Number of columns after one-hot encoding
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)


# Load the model
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(
            X_train.shape[1], 8
        )  # Input layer size based on X columns
        self.act = nn.ReLU()
        self.output = nn.Linear(
            8, num_outputs
        )  # Output layer size based on target classes

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


# Load data
train_data = pd.read_csv(
    "ML/WineCSVs/Train/SixWinesData/SixWines2309(25degEnvTemp)_cleaned.csv"
)
test_data = pd.read_csv(
    "ML/WineCSVs/Test/ControlTests/TallHorse2309control.csv"
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
