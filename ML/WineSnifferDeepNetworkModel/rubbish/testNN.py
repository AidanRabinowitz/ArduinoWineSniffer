import torch
import pandas as pd
import torch.nn as nn
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Define the neural network model (same architecture as your training model)
class Multiclass(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Multiclass, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def load_model(model_path, input_dim, num_classes):
    model = Multiclass(input_dim, num_classes)  # Initialize the model architecture
    model.load_state_dict(torch.load(model_path))  # Load the model weights
    model.eval()  # Set the model to evaluation mode
    return model


def classify_new_data(model, data):
    with torch.no_grad():
        inputs = torch.tensor(
            data.values, dtype=torch.float32
        )  # Convert data to tensor
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class labels
    return predicted


def main(model_path, real_world_test_data, training_data_path):
    # Load and preprocess new data from CSV (unlabeled)
    new_data = pd.read_csv(real_world_test_data)

    # Only keep the feature columns (adjust this if necessary)
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
    ]
    new_data = new_data[feature_columns]

    # Feature normalization (using the same scaler you used during training)
    scaler = StandardScaler()
    new_data = scaler.fit_transform(new_data)  # Normalize the data

    # Load the saved model (adjust the number of classes to match your model)
    input_dim = new_data.shape[1]
    model = load_model(
        model_path, input_dim, 6
    )  # Set num_classes to 6 initially; will be adjusted later

    # Load training data to get the target labels
    training_data = pd.read_csv(training_data_path)
    target_column = "Target"
    wine_labels = training_data[target_column].unique()

    # Initialize the label encoder with the training labels
    le = LabelEncoder()
    le.fit(wine_labels)

    # Classify the new data
    predictions = classify_new_data(model, pd.DataFrame(new_data))

    # Convert predictions to numpy array
    predictions = predictions.numpy()

    # Convert numerical predictions back to string labels
    predicted_labels = le.inverse_transform(predictions)

    # Append predictions to the original data
    new_data_with_predictions = pd.DataFrame(new_data, columns=feature_columns)
    new_data_with_predictions["Predicted Label"] = predicted_labels

    # Save the classified data to a new CSV file
    new_data_with_predictions.to_csv("classified_new_data.csv", index=False)

    # Print out the predicted labels for each wine
    for i, label in enumerate(predicted_labels):
        print(f"Wine {i + 1}: Predicted Label = {label}")

    # Find the most frequent (modal) predicted label
    modal_label = Counter(predictions).most_common(1)[0][0]

    # Print modal label
    print(f"Modal (most common) wine label: {le.inverse_transform([modal_label])[0]}")


if __name__ == "__main__":
    model_path = "wine_model.pth"  # Replace with your actual model path
    real_world_test_data = "ML/WineCSVs/Test/ControlTests/Namaqua2309(25degenvtemp)_control.csv"  # Path to the new CSV without target labels
    training_data_path = "ML/WineCSVs/Train/SixWinesData/SixWinesCombined.csv"  # Path to your training data CSV
    main(model_path, real_world_test_data, training_data_path)
