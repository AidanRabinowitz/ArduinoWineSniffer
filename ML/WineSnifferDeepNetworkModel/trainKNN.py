import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import joblib  # Used for saving and loading models
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


def save_model(knn, label_encoder, scaler):
    # Save the KNN model
    joblib.dump(
        knn,
        "ML/WineSnifferDeepNetworkModel/PCAModel/shallowPKLFiles/knn_model.pkl",
    )
    # Save the label encoder
    joblib.dump(
        label_encoder,
        "ML/WineSnifferDeepNetworkModel/PCAModel/shallowPKLFiles/label_encoder.pkl",
    )
    # Save the scaler
    joblib.dump(
        scaler,
        "ML/WineSnifferDeepNetworkModel/PCAModel/shallowPKLFiles/standard_scaler.pkl",
    )
    print("Model, label encoder, and scaler saved successfully.")


def run_knn():
    # Load your wine dataset
    file_path = (
        "ML/WineCSVs/Train/SimilarDataset/SimilarDataset_combined_cleaned_data.csv"
    )
    try:
        data = pd.read_csv(file_path, header=0)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return

    # Define feature columns and target column (adjust according to your dataset)
    feature_columns = [
        "MQ135",
        "MQ2",
        "MQ3",
        "MQ4",
        "MQ5",
        "MQ6",
        "MQ7",
        "MQ8",
        "MQ9",
        "BMPTemperature",
        "Pressure(Pa)",
        "DHTTemperature",
        "Humidity",
    ]
    target_column = "Target"

    # Create feature matrix X and target vector y
    X = data[feature_columns].values
    y = data[target_column].values

    # Label encode the target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Standardize the features (normalize)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5)
    accuracy_scores = []
    fold = 1

    for train_index, test_index in skf.split(X_scaled, y_encoded):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # Train the KNN classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

        # Test the model on the test set
        y_pred = knn.predict(X_test)

        # Evaluate the model
        accuracy = knn.score(X_test, y_test)
        accuracy_scores.append(accuracy)
        print(f"Fold {fold} Accuracy: {accuracy:.4f}")
        fold += 1

        # Confusion Matrix using `from_predictions`
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=label_encoder.classes_
        )
        plt.title(f"Confusion Matrix - Fold {fold}")
        plt.show()

        # Classification Report
        print(
            metrics.classification_report(
                y_test, y_pred, target_names=label_encoder.classes_
            )
        )

    # Print average accuracy across all folds
    print(
        f"Average Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}"
    )

    # Save the trained model and label encoder (on the last fold)
    save_model(knn, label_encoder, scaler)


if __name__ == "__main__":
    run_knn()
