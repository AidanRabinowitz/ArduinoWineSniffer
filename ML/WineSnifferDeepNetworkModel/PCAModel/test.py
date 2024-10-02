import pandas as pd
import torch
import torch.nn as nn
import joblib
from collections import Counter

# Load the test data
test_data = pd.read_csv(
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineCSVs/Test/Test2509/SophieTest2509(20degEnvTemp).csv"
)

num_wines_in_dataset = 11
# Extract feature columns
feature_columns = ["MQ135", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9"]
X_test = test_data[feature_columns]

# Load the PCA model
pca = joblib.load(r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/pca_model.pkl")

# Apply PCA transformation to the test data
X_test_pca = pca.transform(X_test)

# Convert the transformed test data to a tensor
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)


# Load the trained neural network model
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(
            X_test_pca.shape[1], 32
        )  # Adjust input size to match PCA output
        self.act = nn.ReLU()
        self.output = nn.Linear(32, num_wines_in_dataset)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


# Initialize the model and load weights
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
with torch.no_grad():
    y_pred = model(X_test_tensor)
    predicted_classes = torch.argmax(y_pred, dim=1).numpy()

# Map predicted indices to class names
predicted_class_names = label_encoder.inverse_transform(predicted_classes)

# Print results
for idx, class_name in enumerate(predicted_class_names):
    print(f"Sample {idx + 1}: Classified as {class_name}")

# Optional: Calculate modal classification
modal_class = Counter(predicted_class_names).most_common(1)[0][0]
print(f"\nModal classification wine name: {modal_class}")

# Input for WineTested
wine_tested = input("\nEnter the name of the wine tested: ")

# Calculate Label Accuracy using substring match
total_samples = len(predicted_class_names)
count_correct = sum(
    1 for name in predicted_class_names if wine_tested.lower() in name.lower()
)
label_accuracy = (count_correct / total_samples) * 100 if total_samples > 0 else 0

print(f"Label Accuracy for '{wine_tested}': {label_accuracy:.2f}%")
