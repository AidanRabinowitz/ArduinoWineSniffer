import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder

from WineSnifferDeepNetwork import data, feature_columns, Multiclass


# # Load the training data to fit the LabelEncoder (to map predictions back to class labels)
# train_data = pd.read_csv(
#     "ML/WineCSVs/Train/SixWinesData/SixWinesCombined.csv", header=0
# )

# # Feature columns used for training and testing
# feature_columns = [
#     "MQ135",
#     "MQ2",
#     "MQ3",
#     "MQ4",
#     "MQ5",
#     "MQ6",
#     "MQ7",
#     "MQ8",
#     "MQ9",
#     "BMPTemperature",
#     "Pressure(Pa)",
#     "DHTTemperature",
#     "Humidity",
# ]

# # Target column from the training data
# target_column = "Target"
y_train = data["Target"]

# Fit the LabelEncoder using the training data labels
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

# Load the test data (the one you are testing the model on)
test_data = pd.read_csv(
    "ML/WineCSVs/Test/ControlTests/TallHorse2509(20degEnvTemp)_control.csv", header=0
)
X_test = test_data[feature_columns]

# Convert test data to PyTorch tensor
X_test = torch.tensor(X_test.values, dtype=torch.float32)


# Load the trained model
model = Multiclass()
model.load_state_dict(torch.load("wine_model.pth"))
model.eval()

# Make predictions on test data
with torch.no_grad():
    y_pred_test = model(X_test)

# Get the predicted class indices
predicted_classes = torch.argmax(y_pred_test, dim=1)

# Decode predictions back to wine labels using the LabelEncoder
predicted_wine_names = label_encoder.inverse_transform(predicted_classes.cpu().numpy())

# Print predicted wine labels for each test sample
print("Predicted Wine Labels:")
for i, wine_name in enumerate(predicted_wine_names):
    print(f"Sample {i+1}: {wine_name}")

# Calculate and print the mode (most frequent class) using numpy
unique_wines, counts = np.unique(predicted_wine_names, return_counts=True)
most_frequent_wine = unique_wines[np.argmax(counts)]

print(
    f"\nThe wine that appeared the most frequently is: {most_frequent_wine} (appeared {np.max(counts)} times)"
)
