import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import StratifiedKFold
import joblib

# Load training data
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
]
target_column = "Target"
X = data[feature_columns]
y = data[[target_column]]

# Fit the LabelEncoder using the training data labels
label_encoder = LabelEncoder()
label_encoder.fit(y.values.ravel())
joblib.dump(label_encoder, "label_encoder.pkl")  # Save the encoder
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(y)
y = ohe.transform(y)

# Convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Determine the number of output classes dynamically
num_outputs = y.shape[1]  # Number of columns after one-hot encoding


class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(X.shape[1], 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.hidden2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.act = nn.ReLU()
        self.output = nn.Linear(16, num_outputs)
        self.dropout = nn.Dropout(0.5)  # 50% dropout rate

    def forward(self, x):
        x = self.act(self.bn1(self.hidden1(x)))
        x = self.dropout(self.act(self.bn2(self.hidden2(x))))
        x = self.output(x)
        return x


# Loss metric and optimizer
model = Multiclass()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare model and training parameters
n_epochs = 10
batch_size = 5

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Prepare for metrics recording
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

# Stratified K-Fold training
for fold, (train_index, test_index) in enumerate(
    skf.split(X.numpy(), y.argmax(axis=1))
):
    print(f"Fold {fold + 1}/{skf.n_splits}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    best_acc = -np.inf  # Initialize best accuracy for this fold
    best_weights = None

    # Training loop for the current fold
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = []
        epoch_acc = []
        batches_per_epoch = len(X_train) // batch_size

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
                acc = (
                    (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
                )
                epoch_loss.append(float(loss))
                epoch_acc.append(float(acc))
                bar.set_postfix(loss=float(loss), acc=float(acc))

        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            ce = loss_fn(y_pred, y_test)
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
            ce = float(ce)
            acc = float(acc)

        train_loss_hist.append(np.mean(epoch_loss))
        train_acc_hist.append(np.mean(epoch_acc))
        test_loss_hist.append(ce)
        test_acc_hist.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%"
        )

    # Restore best model for the current fold
    model.load_state_dict(best_weights)

# Save the trained model for later use
torch.save(model.state_dict(), "wine_model.pth")

# Plot the loss and accuracy
plt.plot(train_loss_hist, label="train")
plt.plot(test_loss_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.show()

plt.plot(train_acc_hist, label="train")
plt.plot(test_acc_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

print(f"Highest achieved accuracy across folds: {max(test_acc_hist) * 100:.2f}%")
