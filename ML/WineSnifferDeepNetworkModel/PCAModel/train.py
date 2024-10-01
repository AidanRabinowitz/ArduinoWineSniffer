import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# Train data
data = pd.read_csv(
    r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineCSVs/Train/SixWinesData/SixWines2509(20degEnvTemp).csv_cleaned.csv",
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
]
target_column = "Target"

# Prepare features and targets
X = data[feature_columns]
y = data[target_column]

# Use LabelEncoder for target encoding (class labels instead of one-hot encoding)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=2)  # Modify n_components based on desired variance retention
X_pca = pca.fit_transform(X)

# Convert to PyTorch tensors
X_pca = torch.tensor(X_pca, dtype=torch.float32)
y = torch.tensor(
    y, dtype=torch.long
)  # Use long for class indices, required by CrossEntropyLoss

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, train_size=0.8, shuffle=True
)

# Save label encoder and PCA for later use
import joblib

joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(pca, "pca_model.pkl")  # Save PCA model for future use

# Define the number of output classes
num_outputs = len(np.unique(y))  # Number of classes

# Set device for GPU usage
device = torch.device("cuda")


class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(X_train.shape[1], 32)
        self.act = nn.ReLU()
        self.output = nn.Linear(32, num_outputs)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


# Initialize the model, loss function, and optimizer
model = Multiclass().to(device)  # Move model to GPU
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare DataLoader for batching
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Prepare model training parameters
n_epochs = 100

best_acc = -np.inf
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

# Training loop
for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []

    model.train()
    with tqdm.trange(len(train_loader), unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for X_batch, y_batch in train_loader:
            # Move batch to GPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute and store metrics
            acc = (torch.argmax(y_pred, 1) == y_batch).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            bar.set_postfix(loss=float(loss), acc=float(acc))

    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device))  # Move test data to GPU
        ce = loss_fn(y_pred, y_test.to(device))  # Move test labels to GPU
        acc = (torch.argmax(y_pred, 1) == y_test.to(device)).float().mean()

    train_loss_hist.append(np.mean(epoch_loss))
    train_acc_hist.append(np.mean(epoch_acc))
    test_loss_hist.append(float(ce))
    test_acc_hist.append(float(acc))

    if acc > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(model.state_dict())

    print(
        f"Epoch {epoch} validation: Cross-entropy={float(ce):.2f}, Accuracy={float(acc)*100:.1f}%"
    )

# Restore the best model
model.load_state_dict(best_weights)

# Save the trained model with PCA applied
torch.save(model.state_dict(), "pca_wine_model.pth")

print(f"Highest achieved accuracy: {best_acc * 100:.2f}%")
