import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load the training data to fit the OneHotEncoder
train_data = pd.read_csv("SixWines2509(20degEnvTemp)cleaned.csv", header=0)

# Feature columns used for training and testing
feature_columns = ["MQ135", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9"]

# Target column from the training data
target_column = "Target"
y_train = train_data[[target_column]]

# Fit the OneHotEncoder using the training data labels
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(y_train)

# Load the test data (the one you are testing the model on)
test_data = pd.read_csv(
    "ML/WineCSVs/Test/Test2509/SophieTest2509(20degEnvTemp).csv", header=0
)
X_test = test_data[feature_columns]

# Convert test data to PyTorch tensor
X_test = torch.tensor(X_test.values, dtype=torch.float32)


# Define the same model architecture used in training
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(X_test.shape[1], 8)  # Input layer for 9 features
        self.act = nn.ReLU()
        self.output = nn.Linear(
            8, ohe.categories_[0].size
        )  # Use num_classes instead of 1

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


# Load the trained model
model = Multiclass()
model.load_state_dict(torch.load("wine_model.pth"))
model.eval()

# Make predictions on test data
with torch.no_grad():
    y_pred_test = model(X_test)

# Get the predicted class indices
predicted_classes = torch.argmax(y_pred_test, dim=1)

# Decode predictions back to wine labels
wine_labels = ohe.categories_[
    0
]  # Extract labels from OneHotEncoder fitted on training data

# Print predicted wine labels for each test sample
print("Predicted Wine Labels:")
predicted_wine_names = []
for i, pred_class in enumerate(predicted_classes):
    wine_name = wine_labels[pred_class]
    predicted_wine_names.append(wine_name)
    print(f"Sample {i+1}: {wine_name}")

# Calculate and print the mode using numpy.unique
unique_wines, counts = np.unique(predicted_wine_names, return_counts=True)
most_frequent_wine = unique_wines[np.argmax(counts)]

print(
    f"\nThe wine that appeared the most frequently is: {most_frequent_wine} (appeared {np.max(counts)} times)"
)
