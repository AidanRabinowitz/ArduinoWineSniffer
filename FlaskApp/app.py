from flask import Flask, render_template, jsonify
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load model and encoders globally
# Read the training data and apply one-hot encoding
train_data = pd.read_csv("ML\WineCSVs\TrainCSV.csv", header=0)

X_train = train_data.iloc[:, 1:10]  # Features from columns 1 to 9
y_train = train_data.iloc[:, [14]]  # Target in column 14
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(y_train)
y_train = ohe.transform(y_train)  # One-hot encode the targets

# Convert pandas DataFrame (X_train) and numpy array (y_train) into PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Split the training data for training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, train_size=0.8, shuffle=True
)

# Get the number of output classes (one-hot encoded)
num_outputs = y_train.shape[1]


class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(X_train.shape[1], 8)  # Input layer
        self.act = nn.ReLU()
        self.output = nn.Linear(8, num_outputs)  # Output layer

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


# Define model, loss function, and optimizer
model = Multiclass()
model.load_state_dict(torch.load("wine_model.pth"))
model.eval()
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit([[0], [1], [2]])  # Adjust with actual wine classes


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Simulate loading a CSV
        test_data = pd.read_csv("ML\TestCSVs\TestCSV_no_target.csv", header=0)
        X_test = test_data.iloc[:, 1:10]  # Assuming MQ sensor data in columns 1-9

        # Convert to PyTorch tensor
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

        # Make predictions
        with torch.no_grad():
            y_pred_test = model(X_test)

        # Convert predictions back to wine labels (class names)
        predicted_classes = torch.argmax(y_pred_test, dim=1)
        wine_labels = ohe.inverse_transform(predicted_classes.unsqueeze(1).numpy())

        # Send predicted wine labels to frontend
        return jsonify({"wine_labels": [label[0] for label in wine_labels]})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
