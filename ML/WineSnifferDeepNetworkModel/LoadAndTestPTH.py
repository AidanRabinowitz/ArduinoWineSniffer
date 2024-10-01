import pandas as pd
import torch
import torch.nn as nn
import joblib

# Load the test data
test_data = pd.read_csv("ML/WineCSVs/Test/ControlTests/2309/namaqua2309control.csv")

# Extract feature columns
feature_columns = ["MQ135", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9"]
X_test = test_data[feature_columns]


# Load the trained model
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(X_test.shape[1], 32)
        self.act = nn.ReLU()
        self.output = nn.Linear(32, 6)  # 6 wine labels

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


model = Multiclass()
model.load_state_dict(torch.load("wine_model.pth"))
model.eval()

# Load the label encoder
label_encoder = joblib.load("label_encoder.pkl")

# Preprocess test data
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

# Optional: Calculate modal classification
from collections import Counter

modal_class = Counter(predicted_class_names).most_common(1)[0][0]
print(f"\nModal classification wine name: {modal_class}")
