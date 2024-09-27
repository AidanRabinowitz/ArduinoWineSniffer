import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


# Define the neural network model outside the main block
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


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    data = pd.read_csv(
        "ML/WineCSVs/Train/SixWinesData/SixWines2309(25degEnvTemp)_cleaned.csv",
        header=0,
    )

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
        # "DHTTemperature",
        # "Humidity",
    ]
    target_column = "Target"

    X = data[feature_columns].values
    y = data[target_column].values  # Assuming target is categorical

    # Feature normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode target labels if they are not numeric
    if y.dtype == "object" or not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Classes: {le.classes_}")

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)  # CrossEntropyLoss expects class indices

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, shuffle=True, stratify=y, random_state=42
    )

    # Create TensorDatasets and DataLoaders
    batch_size = 64  # Adjust based on GPU memory

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Define the model
    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y))
    model = Multiclass(input_dim, num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )  # Removed verbose=True

    # Training parameters
    n_epochs = 100
    patience = 10  # For early stopping
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    # For tracking metrics
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []

    # Training loop
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(
                device, non_blocking=True
            )

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

            progress_bar.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

        avg_train_loss = epoch_loss / total
        train_accuracy = correct / total
        train_loss_hist.append(avg_train_loss)
        train_acc_hist.append(train_accuracy)

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(
                    device, non_blocking=True
                )
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        avg_test_loss = test_loss / total
        test_accuracy = correct / total
        test_loss_hist.append(avg_test_loss)
        test_acc_hist.append(test_accuracy)

        print(
            f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy*100:.2f}%, "
            f"Test Loss={avg_test_loss:.4f}, Test Acc={test_accuracy*100:.2f}%"
        )

        # Scheduler step
        scheduler.step(avg_test_loss)

        # Check for improvement
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_weights = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model weights
    model.load_state_dict(best_weights)

    # Save the trained model
    torch.save(model.state_dict(), "wine_model_optimized.pth")

    # Plotting loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(test_loss_hist, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.show()

    # Plotting accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_hist, label="Train Accuracy")
    plt.plot(test_acc_hist, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()
    plt.show()

    print(f"Best Test Accuracy: {best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
