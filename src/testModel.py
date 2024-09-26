import os
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from data_cleaner import DataCleaner
from deep_learning import DeepLearning


def load_test_data(test_folder, cleaned_output_file):
    data_cleaner = DataCleaner(test_folder, cleaned_output_file, 'iqr')
    data_cleaner.clean_data()


def test_model(model_path, cleaned_data_file):
    # Load the cleaned test data
    cleaned_data = pd.read_csv(cleaned_data_file)

    # Assuming feature columns are from 1 to 10 and target column is in the 14th position
    X_test = cleaned_data.iloc[:, 1:10]
    y_test = cleaned_data.iloc[:, 14]

    # One-hot encode the target
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    y_test_encoded = ohe.fit_transform(y_test.values.reshape(-1, 1))

    # Convert to PyTorch tensors
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.float32)

    # Instantiate the model class
    model = DeepLearning(
        input_size=X_test.shape[1], output_size=y_test_encoded.shape[1])

    # Load the saved state_dict into the model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Make predictions
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        accuracy = (torch.argmax(y_pred, dim=1) == torch.argmax(
            y_test_tensor, dim=1)).float().mean()

    print(f"Model accuracy on test data: {accuracy.item() * 100:.2f}%")


if __name__ == "__main__":
    test_folder = "src/TestData"  # Path to your test data folder
    cleaned_output_file = "src/cleaned_test_data.csv"
    model_path = "src/wine_model.pth"

    # Load and clean the test data
    load_test_data(test_folder, cleaned_output_file)

    # Test the model
    test_model(model_path, cleaned_output_file)
