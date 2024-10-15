import torch
import torch.nn as nn
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin


# Define the DataFrameSelector class
class DataFrameSelector(TransformerMixin):
    """Custom transformer to select columns from a DataFrame"""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns].values


# Define the Multiclass model
class Multiclass(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=7):
        super(Multiclass, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Load the trained model
input_dim = 3 + 1  # Adjust this based on your PCA output dimensions
model = Multiclass(input_dim=input_dim)
model.load_state_dict(torch.load("final_trained_model.pth", weights_only=True))
model.eval()

# Load the preprocessing pipeline and label encoder
preprocessing_pipeline = joblib.load(
    "ML/WineSnifferDeepNetworkModel/PCAModel/NNPKLFiles/preprocessing_pipeline.pkl"
)
label_encoder = joblib.load(
    "ML/WineSnifferDeepNetworkModel/PCAModel/NNPKLFiles/label_encoder.pkl"
)


def predict_labels(input_csv):
    # Load the data
    data = pd.read_csv(input_csv)

    # Preprocess the data
    X_preprocessed = preprocessing_pipeline.transform(data)

    # Convert to PyTorch tensor
    X_tensor = torch.tensor(X_preprocessed, dtype=torch.float32)

    # Make predictions
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted_classes = torch.max(outputs, 1)

    # Decode the predicted labels
    predicted_labels = label_encoder.inverse_transform(predicted_classes.numpy())

    # Add predictions to the original dataframe
    data["Predicted_Target"] = predicted_labels
    print(predicted_labels)


if __name__ == "__main__":
    input_csv = "ML/WineCSVs/Test/Test2309/silvermyntestforpcatest.csv"  # Replace with your input CSV path
    predict_labels(input_csv)
