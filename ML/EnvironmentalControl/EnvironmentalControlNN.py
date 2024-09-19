import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def adjust_mq_sensors(file_path, env_columns, mq_columns, output_csv):
    """
    Adjust MQ sensor values based on environmental factors using a neural network.

    Parameters:
    file_path (str): Path to the input CSV file containing sensor data.
    env_columns (list): List of column names representing environmental factors.
    mq_columns (list): List of column names representing MQ sensor data.
    output_csv (str): Path to save the adjusted data as a CSV file.

    Returns:
    DataFrame: A DataFrame with adjusted MQ sensor values.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Extract environmental and sensor columns
    X_env = data[env_columns].values
    y_mq = data[mq_columns].values

    # Prepare a new DataFrame to store adjusted sensor values
    adjusted_mq = pd.DataFrame(index=data.index)

    # Normalize the environmental data
    scaler = StandardScaler()
    X_env_scaled = scaler.fit_transform(X_env)

    # Train a neural network for each MQ sensor
    for i, sensor in enumerate(mq_columns):
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X_env_scaled)
        y_tensor = torch.FloatTensor(y_mq[:, i]).view(-1, 1)

        # Initialize the model, loss function, and optimizer
        model = MLP(input_size=X_tensor.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training the model on the entire dataset
        model.train()
        for epoch in tqdm(range(100), desc=f"Training {sensor}", unit="epoch"):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        # Predict the MQ sensor's values based on environmental conditions
        model.eval()
        with torch.no_grad():
            predicted_mq = model(X_tensor).numpy()

        # Subtract the predicted values from the actual sensor values
        adjusted_mq[sensor] = data[sensor] - predicted_mq.flatten()

    # Save adjusted MQ data to a CSV file
    adjusted_mq.to_csv(output_csv, index=False)

    return adjusted_mq


def remove_invalid_rows(file_path, output_cleaned_csv):
    """
    Removes rows that contain the substring '_A' in any columns
    except the last one. The cleaned data is saved to a new CSV.

    Parameters:
    file_path (str): Path to the input CSV file containing sensor data.
    output_cleaned_csv (str): Path to save the cleaned data as a CSV file.

    Returns:
    DataFrame: A DataFrame with the cleaned data.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Get the last column name
    last_column = data.columns[-1]

    # Get the columns to check (all except the last one)
    non_last_columns = data.columns[:-1]

    # Create a mask to filter out rows containing '_A' in non-last columns
    mask = data[non_last_columns].apply(
        lambda row: ~row.astype(str).str.contains("_A").any(), axis=1
    )

    # Keep only the valid rows
    cleaned_data = data[mask]

    # Save the cleaned data to a new CSV
    cleaned_data.to_csv(output_cleaned_csv, index=False)

    return cleaned_data


# Define the file paths and column names
input_csv = "ML\EnvironmentalControl\TemperatureRangeTest.csv"
cleaned_csv = "ML\EnvironmentalControl\TemperatureRangeTest_cleaned.csv"
output_csv = "ML\EnvironmentalControl\TemperatureRangeTest_adjusted.csv"

# Use only the BMPTemperature column as the environmental factor
env_columns = ["BMPTemperature"]
mq_columns = ["MQ135", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9"]

# Call the function to clean the data
cleaned_data = remove_invalid_rows(input_csv, cleaned_csv)
print(f"Cleaned data saved to {cleaned_csv}")

# Call the function to adjust the MQ sensor data and save to CSV
adjusted_data = adjust_mq_sensors(cleaned_csv, env_columns, mq_columns, output_csv)
print(f"Adjusted data saved to {output_csv}")
