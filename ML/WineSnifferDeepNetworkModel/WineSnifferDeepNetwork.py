import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Train data
data = pd.read_csv(
    "ML\WineCSVs\Train\SixWinesData\SixWinesCombined.csv",
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
    # "Pressure(Pa)",
    # "DHTTemperature",
    # "Humidity",
]
target_column = "Target"
# For adjusted CSV (environmental control)
X = data[feature_columns]
y = data[[target_column]]

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(y)
y = ohe.transform(y)

# convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# X = torch.nn.functional.normalize(X, p=1.0, dim=1)
# y = torch.nn.functional.normalize(y, p=1.0, dim=1)
# split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

# Determine the number of output classes dynamically
num_outputs = y.shape[1]  # Number of columns after one-hot encoding


# class Multiclass(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(
#             X_train.shape[1], 8
#         )  # Input layer size based on X columns
#         self.act = nn.ReLU()
#         self.output = nn.Linear(
#             8, num_outputs
#         )  # Output layer size based on target classes

#     def forward(self, x):
#         x = self.act(self.hidden(x))
#         x = self.output(x)
#         return x


class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(X_train.shape[1], 32)
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


# loss metric and optimizer
model = Multiclass()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 50
batch_size = 5
batches_per_epoch = len(X_train) // batch_size

best_acc = -np.inf  # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

# training loop
for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []
    # set model in training mode and run through each batch
    model.train()
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            X_batch = X_train[start : start + batch_size]
            y_batch = y_train[start : start + batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # compute and store metrics
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            bar.set_postfix(loss=float(loss), acc=float(acc))
    # set model in evaluation mode and run through the test set
    model.eval()
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
    print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")

    # Restore best model
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

print(f"Highest achieved accuracy: {best_acc * 100:.2f}%")
