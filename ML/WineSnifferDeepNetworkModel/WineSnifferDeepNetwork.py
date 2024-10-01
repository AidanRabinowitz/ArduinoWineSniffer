import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Train data
data = pd.read_csv(
    "ML/WineCSVs/Train/SixWinesData/SixWines2509(20degEnvTemp).csv_cleaned.csv",
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

# Convert to PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(
    y, dtype=torch.long
)  # Use long for class indices, required by CrossEntropyLoss

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

# Save label encoder for later use
import joblib

joblib.dump(label_encoder, "label_encoder.pkl")

# Define the number of output classes
num_outputs = len(np.unique(y))  # Number of classes


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
model = Multiclass()
loss_fn = (
    nn.CrossEntropyLoss()
)  # CrossEntropyLoss expects raw class labels, not one-hot
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare model training parameters
n_epochs = 100
batch_size = 5
batches_per_epoch = len(X_train) // batch_size

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
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # Take a batch
            start = i * batch_size
            X_batch = X_train[start : start + batch_size]
            y_batch = y_train[start : start + batch_size]

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
        y_pred = model(X_test)
        ce = loss_fn(y_pred, y_test)
        acc = (torch.argmax(y_pred, 1) == y_test).float().mean()

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

# Save the trained model for later use
torch.save(model.state_dict(), "wine_model.pth")

print(f"Highest achieved accuracy: {best_acc * 100:.2f}%")
