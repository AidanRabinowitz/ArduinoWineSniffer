from flask import Flask, jsonify, request
import pandas as pd
import torch
import torch.nn as nn
import joblib
from collections import Counter
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)
CORS(app)

# Load the PCA model and other resources
pca = joblib.load(r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/pca_model.pkl")
label_encoder = joblib.load(
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/label_encoder.pkl"
)

# Load the test data
test_data = pd.read_csv(
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineCSVs/Test/Test2309/silvermyntestforpcatest.csv"
)

num_wines_in_dataset = 11
feature_columns = ["MQ135", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9"]
X_test = test_data[feature_columns]

# Load the PCA model
pca = joblib.load(r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/pca_model.pkl")
X_test_pca = pca.transform(X_test)


class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(X_test_pca.shape[1], 32)
        self.act = nn.ReLU()
        self.output = nn.Linear(32, num_wines_in_dataset)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


model = Multiclass()
model.load_state_dict(
    torch.load(r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/pca_wine_model.pth")
)
model.eval()


@app.route("/run-test", methods=["POST"])
def run_test():
    wine_name = request.json.get("wine_name", "")
    if not wine_name:
        return jsonify({"error": "Wine name is required."}), 400

    try:
        subprocess.run(
            [
                "python",
                r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel/test.py",
                wine_name,
            ],
            check=True,
        )

        # Read the accuracy from the file
        with open("label_accuracy.txt", "r") as f:
            label_accuracy = float(f.read().strip())

        return (
            jsonify(
                {
                    "message": "Script executed successfully.",
                    "label_accuracy": label_accuracy,
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["GET"])
def predict():
    # Load the test data
    test_data = pd.read_csv(
        r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineCSVs/Test/Test2309/silvermyntestforpcatest.csv"
    )

    feature_columns = ["MQ135", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9"]
    X_test = test_data[feature_columns]

    # Apply PCA transformation
    X_test_pca = pca.transform(X_test)
    X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)

    with torch.no_grad():
        y_pred = model(X_test_tensor)
        predicted_classes = torch.argmax(y_pred, dim=1).numpy()

    predicted_class_names = label_encoder.inverse_transform(predicted_classes)
    modal_class = Counter(predicted_class_names).most_common(1)[0][0]

    # Read the label accuracy from the file
    try:
        with open("label_accuracy.txt", "r") as f:
            label_accuracy = float(f.read().strip())
    except Exception as e:
        label_accuracy = None  # Handle the case where the file is not available

    return jsonify(
        {
            "predictions": predicted_class_names.tolist(),
            "modal_class": modal_class,
            "label_accuracy": label_accuracy,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
