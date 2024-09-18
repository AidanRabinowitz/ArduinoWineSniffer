import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Read the training data and apply one-hot encoding
train_data = pd.read_csv("ML/WineCSVs/SixWineData.csv", header=0)

X_train = train_data.iloc[:, 1:10]  # Features from columns 1 to 9
y_train = train_data.iloc[:, [14]]  # Target in column 14
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(y_train)
y_train = ohe.transform(y_train)  # One-hot encode the targets

# Convert pandas DataFrame (X_train) and numpy array (y_train) into PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# Split the training data for training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, train_size=0.8, shuffle=True
)

# Get the number of output classes (one-hot encoded)
num_outputs = y_train.shape[1]


class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(X_train.shape[1], 8)  # Input layer
        self.act = nn.ReLU()
        self.output = nn.Linear(8, num_outputs)  # Output layer

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


# Define model, loss function, and optimizer
model = Multiclass()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
n_epochs = 100
batch_size = 5
batches_per_epoch = len(X_train) // batch_size

best_acc = -np.inf
best_weights = None
train_loss_hist = []
train_acc_hist = []
val_loss_hist = []
val_acc_hist = []

# Training loop
for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []
    model.train()
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            start = i * batch_size
            X_batch = X_train[start : start + batch_size]
            y_batch = y_train[start : start + batch_size]

            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            bar.set_postfix(loss=float(loss), acc=float(acc))

    # Validation
    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_val)
        val_loss = loss_fn(y_pred_val, y_val)
        val_acc = (torch.argmax(y_pred_val, 1) == torch.argmax(y_val, 1)).float().mean()

    val_loss = float(val_loss)
    val_acc = float(val_acc)
    train_loss_hist.append(np.mean(epoch_loss))
    train_acc_hist.append(np.mean(epoch_acc))
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())

    print(
        f"Epoch {epoch}: Validation Loss = {val_loss:.2f}, Validation Accuracy = {val_acc*100:.1f}%"
    )

# Load the best weights
model.load_state_dict(best_weights)

# Save the trained model for later use
torch.save(model.state_dict(), "wine_model.pth")

# Plot training history
plt.plot(train_loss_hist, label="train")
plt.plot(val_loss_hist, label="val")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.plot(train_acc_hist, label="train")
plt.plot(val_acc_hist, label="val")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

print(f"Highest achieved validation accuracy: {best_acc * 100:.2f}%")

# BELOW CODE CAN BE MOVE TO A SEPARATE CLASS OR PYTHON SCRIPT

# Load new data (without target column) for prediction
test_data = pd.read_csv("ML\TestCSVs\TestCSV1309.csv", header=0)
X_test = test_data.iloc[:, 1:10]  # MQ sensor data (columns 1 to 9)

# Convert to PyTorch tensor
X_test = torch.tensor(X_test.values, dtype=torch.float32)

# Load the trained model
model = Multiclass()  # Assuming the Multiclass model is already defined
model.load_state_dict(torch.load("wine_model.pth"))
model.eval()

X_test = test_data.iloc[:, 1:10]  # MQ sensor data (columns 1 to 9)

# Convert to PyTorch tensor
X_test = torch.tensor(X_test.values, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    y_pred_test = model(X_test)

# Get the predicted class indices
predicted_classes = torch.argmax(y_pred_test, dim=1)

# Create a mapping of class indices to wine labels
# Assuming ohe is the OneHotEncoder from training
wine_labels = ohe.categories_[0]

# Print predicted wine labels
print("Predicted Wine Labels:")
for i, pred_class in enumerate(predicted_classes):
    print(f"Sample {i+1}: {wine_labels[pred_class]}")
