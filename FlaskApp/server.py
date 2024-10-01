from flask import Flask, jsonify
import pandas as pd
import torch
import torch.nn as nn
import joblib
from collections import Counter
from flask_cors import CORS

app = Flask(__name__)

CORS(app)


# Load the model
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(9, 32)
        self.act = nn.ReLU()
        self.output = nn.Linear(32, 6)  # 6 wine labels

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


model = Multiclass()
model.load_state_dict(torch.load("../wine_model.pth"))
model.eval()

# Load the label encoder
label_encoder = joblib.load("../label_encoder.pkl")

# Load the test data (for demonstration, this should be dynamic)
test_data = pd.read_csv(
    "../ML/WineCSVs/Test/ControlTests/2509/silvermyn2509control.csv"
)
feature_columns = ["MQ135", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9"]
X_test = test_data[feature_columns]


@app.route("/predict", methods=["GET"])
def predict():
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

    with torch.no_grad():
        y_pred = model(X_test_tensor)
        predicted_classes = torch.argmax(y_pred, dim=1).numpy()

    predicted_class_names = label_encoder.inverse_transform(predicted_classes)
    modal_class = Counter(predicted_class_names).most_common(1)[0][0]

    return jsonify(
        {"predictions": predicted_class_names.tolist(), "modal_class": modal_class}
    )


if __name__ == "__main__":
    app.run(debug=True)
