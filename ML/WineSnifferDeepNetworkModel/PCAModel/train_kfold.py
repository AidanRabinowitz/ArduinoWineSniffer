import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.metrics import f1_score
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Multiclass(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=6):
        super(Multiclass, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DataFrameSelector(TransformerMixin):
    """Custom transformer to select columns from a DataFrame"""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns].values


def runTrain():
    # Load the dataset
    data = pd.read_csv(
        "ML/WineCSVs/Train/OldWinesDataset/OldWinesDataset_combined_cleaned_data.csv",
        header=0,
    )

    # Define feature columns for MQ sensors and environmental sensors
    feature_columns_mq = [
        "MQ135",
        "MQ2",
        "MQ3",
        "MQ4",
        "MQ5",
        "MQ6",
        "MQ7",
        "MQ8",
        "MQ9",
    ]
    env_sensors = ["BMPTemperature", "Pressure(Pa)", "DHTTemperature", "Humidity"]
    target_column = "Target"

    # Prepare label encoder for the target
    label_encoder = LabelEncoder()
    y = data[target_column]
    y_encoded = label_encoder.fit_transform(y)

    # Save the label encoder for later use
    joblib.dump(
        label_encoder,
        "ML/WineSnifferDeepNetworkModel/PCAModel/NNPKLFiles/label_encoder.pkl",
    )

    # Number of classes
    num_classes = len(np.unique(y_encoded))

    # Define a pipeline for preprocessing (scaling only)
    preprocessing_pipeline = ColumnTransformer(
        [
            (
                "mq",
                Pipeline(
                    [
                        ("selector", DataFrameSelector(feature_columns_mq)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns_mq,
            ),
            (
                "env",
                Pipeline(
                    [
                        ("selector", DataFrameSelector(env_sensors)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                env_sensors,
            ),
        ]
    )

    # Apply the preprocessing pipeline to the data
    X_preprocessed = preprocessing_pipeline.fit_transform(data)

    # Save the preprocessing pipeline
    joblib.dump(
        preprocessing_pipeline,
        "ML/WineSnifferDeepNetworkModel/PCAModel/NNPKLFiles/preprocessing_pipeline.pkl",
    )

    # Initialize the model with the number of features after preprocessing
    input_dim = X_preprocessed.shape[1]
    model = Multiclass(input_dim=input_dim, hidden_dim=64, output_dim=num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    num_epochs = 1000
    best_model_state = None
    highest_test_accuracy = 0.0  # Track highest test accuracy

    fold_train_accuracies = []
    fold_test_accuracies = []
    fold_train_f1_scores = []
    fold_test_f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_preprocessed, y_encoded)):
        print(f"Fold {fold+1}/{skf.n_splits}")

        # Re-initialize the model for each fold
        model = Multiclass(input_dim=input_dim, hidden_dim=64, output_dim=num_classes)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Split the data
        X_train, X_test = X_preprocessed[train_idx], X_preprocessed[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_accuracies = []
        test_accuracies = []
        train_f1_scores = []
        test_f1_scores = []

        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate training accuracy and F1-score
            _, predicted_train = torch.max(outputs, 1)
            correct_train = (predicted_train == y_train_tensor).sum().item()
            train_accuracy = correct_train / y_train_tensor.size(0)
            train_f1 = f1_score(
                y_train_tensor.cpu().numpy(),
                predicted_train.cpu().numpy(),
                average="weighted",
            )
            train_accuracies.append(train_accuracy)
            train_f1_scores.append(train_f1)

            # Calculate test accuracy and F1-score
            with torch.no_grad():
                outputs_test = model(X_test_tensor)
                _, predicted_test = torch.max(outputs_test, 1)
                correct_test = (predicted_test == y_test_tensor).sum().item()
                test_accuracy = correct_test / y_test_tensor.size(0)
                test_f1 = f1_score(
                    y_test_tensor.cpu().numpy(),
                    predicted_test.cpu().numpy(),
                    average="weighted",
                )
                test_accuracies.append(test_accuracy)
                test_f1_scores.append(test_f1)

            if test_accuracy > highest_test_accuracy:
                highest_test_accuracy = test_accuracy
                best_model_state = model.state_dict().copy()  # Save best model

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, "
                    f"Train Accuracy: {train_accuracy:.4f}, Train F1-Score: {train_f1:.4f}, "
                    f"Test Accuracy: {test_accuracy:.4f}, Test F1-Score: {test_f1:.4f}"
                )

        fold_train_accuracies.append(train_accuracies)
        fold_test_accuracies.append(test_accuracies)
        fold_train_f1_scores.append(train_f1_scores)
        fold_test_f1_scores.append(test_f1_scores)

    # Retrain the model on the entire dataset using the best model
    print(
        f"Retraining on entire dataset using the best model parameters from cross-validation"
    )
    model.load_state_dict(best_model_state)  # Load the best model's state

    # Convert full dataset to PyTorch tensors
    X_full_tensor = torch.tensor(X_preprocessed, dtype=torch.float32)
    y_full_tensor = torch.tensor(y_encoded, dtype=torch.long)

    for epoch in range(num_epochs):
        outputs = model(X_full_tensor)
        loss = criterion(outputs, y_full_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy on the full dataset
        _, predicted_full = torch.max(outputs, 1)
        correct_full = (predicted_full == y_full_tensor).sum().item()
        full_train_accuracy = correct_full / y_full_tensor.size(0)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Full Train Accuracy: {full_train_accuracy:.4f}"
            )

    # Save the final model trained on the entire dataset
    torch.save(model.state_dict(), "final_trained_model.pth")
    print(f"Final model saved as 'final_trained_model.pth'")

    # Plot accuracy and F1-score results from cross-validation
    plt.figure(figsize=(12, 8))
    for fold in range(skf.n_splits):
        plt.plot(fold_train_accuracies[fold], label=f"Fold {fold+1} Train Accuracy")
        plt.plot(
            fold_test_accuracies[fold],
            label=f"Fold {fold+1} Test Accuracy",
            linestyle="--",
        )
    plt.title("Train and Test Accuracy per Fold", fontsize=30)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.legend(fontsize=18)
    plt.show()

    plt.figure(figsize=(12, 8))
    for fold in range(skf.n_splits):
        plt.plot(fold_train_f1_scores[fold], label=f"Fold {fold+1} Train F1-Score")
        plt.plot(
            fold_test_f1_scores[fold],
            label=f"Fold {fold+1} Test F1-Score",
            linestyle="--",
        )
    plt.title("Train and Test F1-Score per Fold", fontsize=30)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("F1-Score", fontsize=20)
    plt.legend(fontsize=18)
    plt.show()


if __name__ == "__main__":
    runTrain()
