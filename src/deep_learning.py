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


class DeepLearning:
    def __init__(self, input_csv, output_model_path="src/wine_model.pth", n_epochs=20, batch_size=5, lr=0.001):
        self.input_csv = input_csv
        self.output_model_path = output_model_path
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load and process the data
        self.X_train, self.X_test, self.y_train, self.y_test, self.num_outputs = self._prepare_data()

        # Initialize the model, loss function, and optimizer
        self.model = self.Multiclass(
            self.X_train.shape[1], self.num_outputs).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _prepare_data(self):
        # Read and preprocess data
        data = pd.read_csv(self.input_csv)
        X = data.iloc[:, 1:10]
        y = data.iloc[:, [14]]

        # One-hot encode the target
        ohe = OneHotEncoder(handle_unknown="ignore",
                            sparse_output=False).fit(y)
        y = ohe.transform(y)

        # Convert to PyTorch tensors
        X = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, shuffle=True)

        # Returning number of output classes dynamically
        return X_train, X_test, y_train, y_test, y.shape[1]

    class Multiclass(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            # Input size based on X columns
            self.hidden = nn.Linear(input_size, 8)
            self.act = nn.ReLU()
            # Output size based on one-hot encoded classes
            self.output = nn.Linear(8, output_size)

        def forward(self, x):
            x = self.act(self.hidden(x))
            x = self.output(x)
            return x

    def train_and_evaluate(self):
        # Tracking metrics
        best_acc = -np.inf
        best_weights = None
        train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = [], [], [], []

        batches_per_epoch = len(self.X_train) // self.batch_size

        for epoch in range(self.n_epochs):
            epoch_loss, epoch_acc = [], []
            self.model.train()

            with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
                bar.set_description(f"Epoch {epoch}")
                for i in bar:
                    start = i * self.batch_size
                    X_batch = self.X_train[start: start + self.batch_size]
                    y_batch = self.y_train[start: start + self.batch_size]

                    # Forward pass
                    y_pred = self.model(X_batch)
                    loss = self.loss_fn(y_pred, y_batch)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Compute metrics
                    acc = (torch.argmax(y_pred, 1) ==
                           torch.argmax(y_batch, 1)).float().mean()
                    epoch_loss.append(float(loss))
                    epoch_acc.append(float(acc))
                    bar.set_postfix(loss=float(loss), acc=float(acc))

            # Evaluation on the test set
            self.model.eval()
            y_pred = self.model(self.X_test)
            ce = self.loss_fn(y_pred, self.y_test)
            acc = (torch.argmax(y_pred, 1) == torch.argmax(
                self.y_test, 1)).float().mean()

            train_loss_hist.append(np.mean(epoch_loss))
            train_acc_hist.append(np.mean(epoch_acc))
            test_loss_hist.append(float(ce))
            test_acc_hist.append(float(acc))

            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(self.model.state_dict())

            print(
                f"Epoch {epoch} validation: Cross-entropy={float(ce):.2f}, Accuracy={float(acc) * 100:.1f}%")

        # Restore best weights
        self.model.load_state_dict(best_weights)
        torch.save(self.model.state_dict(), self.output_model_path)

        # Plot the results
        self._plot_metrics(train_loss_hist, test_loss_hist,
                           train_acc_hist, test_acc_hist)
        print(f"Highest achieved accuracy: {best_acc * 100:.2f}%")

    def _plot_metrics(self, train_loss, test_loss, train_acc, test_acc):
        # Loss plot
        plt.plot(train_loss, label="train")
        plt.plot(test_loss, label="test")
        plt.xlabel("epochs")
        plt.ylabel("cross entropy")
        plt.legend()
        plt.show()

        # Accuracy plot
        plt.plot(train_acc, label="train")
        plt.plot(test_acc, label="test")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()
