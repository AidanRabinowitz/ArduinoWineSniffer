from flask import Flask, request, jsonify
import pandas as pd
import torch
import torch.nn as nn
import joblib
from collections import Counter

app = Flask(__name__)


# Define the PyTorch model
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(9, 32)  # Assuming 9 input features (like MQ135, MQ2...)
        self.act = nn.ReLU()
        self.output = nn.Linear(32, 6)  # Assuming 6 wine classes

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


# Load model and label encoder
model = Multiclass()
model.load_state_dict(torch.load("../wine_model.pth"))
model.eval()

label_encoder = joblib.load("../label_encoder.pkl")
class_names = label_encoder.classes_


# Define a route to get the modal classification
@app.route("/classify", methods=["POST"])
def classify_wine():
    # Load input data from request
    data = request.json
    df = pd.DataFrame(data)

    # Convert to tensor
    X_test_tensor = torch.tensor(df.values, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        predicted_classes = torch.argmax(y_pred, dim=1).numpy()

    # Map indices to class names
    predicted_class_names = label_encoder.inverse_transform(predicted_classes)

    # Calculate modal class
    modal_class = Counter(predicted_class_names).most_common(1)[0][0]

    # Return modal classification
    return jsonify({"modal_class": modal_class})


if __name__ == "__main__":
    app.run(debug=True)
