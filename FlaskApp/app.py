from flask import Flask, render_template, jsonify, request
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import copy

app = Flask(__name__)

# Load and prepare the model and encoders globally
train_data = pd.read_csv("ML/WineCSVs/TrainCSV.csv", header=0)

X_train = train_data.iloc[:, 1:10]  # Features from columns 1 to 9
y_train = train_data.iloc[:, [14]]  # Target in column 14

# One-hot encode the target
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(y_train)
y_train = ohe.transform(y_train)  # One-hot encode the targets

# Convert pandas DataFrame (X_train) and numpy array (y_train) into PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)


# Define the PyTorch model
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(X_train.shape[1], 8)  # Input layer (9 features)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, y_train.shape[1])  # Output layer

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


# Load pre-trained model
model = Multiclass()
model.load_state_dict(torch.load("wine_model.pth"))
model.eval()


# Flask routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load the test CSV data (ensure it has the correct number of columns)
        test_data = pd.read_csv("ML/TestCSVs/TestCSV_no_target.csv", header=0)
        X_test = test_data.iloc[:, 1:10]  # Ensure it has 9 features (columns 1-9)

        # Convert to PyTorch tensor
        X_test = torch.tensor(X_test.values, dtype=torch.float32)

        # Make predictions
        with torch.no_grad():
            y_pred_test = model(X_test)

        # Get the predicted class indices
        predicted_classes = torch.argmax(y_pred_test, dim=1)

        # Convert predictions back to wine labels (class names)
        wine_labels = ohe.categories_[0]  # Map indices to original labels
        predicted_wine_labels = [wine_labels[idx] for idx in predicted_classes]

        # Return the predicted wine labels as JSON
        return jsonify({"wine_labels": predicted_wine_labels})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
