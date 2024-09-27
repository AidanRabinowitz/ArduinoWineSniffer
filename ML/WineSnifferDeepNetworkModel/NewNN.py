import copy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
data = pd.read_csv("ML/WineCSVs/Train/SixWinesData/SixWinesCombined.csv", header=0)

# Define feature columns and target
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

# Prepare input (X) and target (y)
X = data[feature_columns].values
y = data[target_column].values

# Encode the target classes as integer labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convert data into PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(
    y, dtype=torch.long
)  # Note: CrossEntropyLoss expects class indices, not one-hot encoded

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

# Move data to the correct device (GPU or CPU)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Create DataLoader for batching
batch_size = 32
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)


# Define the model
class MulticlassClassifier(nn.Module):
    def __init__(self):
        super(MulticlassClassifier, self).__init__()
        self.hidden1 = nn.Linear(X_train.shape[1], 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.hidden2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.output = nn.Linear(
            16, len(label_encoder.classes_)
        )  # Output size = number of classes

    def forward(self, x):
        x = torch.relu(self.bn1(self.hidden1(x)))
        x = torch.relu(self.bn2(self.hidden2(x)))
        x = self.output(x)  # Raw logits, no softmax needed
        return x


# Initialize the model, loss function, and optimizer
model = MulticlassClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multiclass classification
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Initialize variables for tracking the best accuracy
best_acc = -np.inf
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []
patience = 10
epochs_without_improvement = 0

# Training loop
n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    epoch_loss = []
    epoch_acc = []

    # Training phase
    with tqdm.tqdm(train_loader, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch + 1}/{n_epochs}")
        for X_batch, y_batch in bar:
            # Forward pass
            y_pred = model(X_batch)

            # Compute the loss
            loss = loss_fn(y_pred, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            preds = torch.argmax(y_pred, dim=1)
            acc = (preds == y_batch).float().mean()

            # Log the loss and accuracy for this batch
            epoch_loss.append(loss.item())
            epoch_acc.append(acc.item())
            bar.set_postfix(loss=loss.item(), acc=acc.item())

    # Scheduler step
    scheduler.step()

    # Evaluation phase
    model.eval()
    test_loss = []
    test_acc = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)

            # Compute the test loss
            loss = loss_fn(y_pred, y_batch)
            test_loss.append(loss.item())

            # Calculate test accuracy
            preds = torch.argmax(y_pred, dim=1)
            acc = (preds == y_batch).float().mean()
            test_acc.append(acc.item())

    # Calculate mean loss and accuracy
    train_loss_hist.append(np.mean(epoch_loss))
    train_acc_hist.append(np.mean(epoch_acc))
    test_loss_hist.append(np.mean(test_loss))
    test_acc_hist.append(np.mean(test_acc))

    # Log epoch results
    print(
        f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {np.mean(epoch_loss):.4f}, Train Acc: {np.mean(epoch_acc):.4f}, "
        f"Test Loss: {np.mean(test_loss):.4f}, Test Acc: {np.mean(test_acc):.4f}"
    )

    # Early stopping logic
    if np.mean(test_acc) > best_acc:
        best_acc = np.mean(test_acc)
        best_weights = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Stopping early at epoch {epoch + 1}")
        break

# Restore the best model weights
model.load_state_dict(best_weights)

# Save the model
torch.save(model.state_dict(), "wine_multiclass_model.pth")

# Plot the loss and accuracy curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_hist, label="Train Loss")
plt.plot(test_loss_hist, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_hist, label="Train Accuracy")
plt.plot(test_acc_hist, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

print(f"Highest achieved test accuracy: {best_acc * 100:.2f}%")
