import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn


# Load the pre-trained model class
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(9, 8)  # Adjust input size based on your data
        self.act = nn.ReLU()
        self.output = nn.Linear(8, 6)  # Adjust output size based on number of classes

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


# Load the trained model
model = Multiclass()
model.load_state_dict(torch.load("wine_model.pth"))
model.eval()

# Load the test data
test_data = pd.read_csv("ML\TestCSVs\TestCSV1309.csv", header=0)
X_test = test_data.iloc[:, 1:10]  # MQ sensor data (columns 1 to 9)

# Convert to PyTorch tensor
X_test = torch.tensor(X_test.values, dtype=torch.float32)

# OneHotEncoder from training for decoding predictions back to labels
# Ensure the same encoder is used as in training
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(
    pd.read_csv("ML/TrainCSVs/SixWines1609.csv", header=0).iloc[:, [14]]
)  # Load target column from training data

# Make predictions on test data
with torch.no_grad():
    y_pred_test = model(X_test)

# Get the predicted class indices
predicted_classes = torch.argmax(y_pred_test, dim=1)

# Create a mapping of class indices to wine labels (from OHE used in training)
wine_labels = ohe.categories_[
    0
]  # Extracting the labels (assuming OHE was fitted similarly during training)

# Print predicted wine labels for each test sample
print("Predicted Wine Labels:")
for i, pred_class in enumerate(predicted_classes):
    print(f"Sample {i+1}: {wine_labels[pred_class]}")
