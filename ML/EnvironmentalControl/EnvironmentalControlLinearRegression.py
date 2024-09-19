import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


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


def adjust_mq_sensors(file_path, env_columns, mq_columns, output_csv):
    """
    Adjust MQ sensor values based on environmental factors (temperature, humidity, pressure).

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
    X_env = data[env_columns]
    X_mq = data[mq_columns]

    # Prepare a new DataFrame to store adjusted sensor values
    adjusted_mq = pd.DataFrame(index=data.index)

    # Fit a linear regression model for each MQ sensor using the environmental factors
    for sensor in mq_columns:
        model = LinearRegression()
        model.fit(X_env, data[sensor])

        # Predict the MQ sensor's values based on environmental conditions
        predicted_mq = model.predict(X_env)

        # Subtract the predicted values (environmentally influenced) from the actual sensor values
        adjusted_mq[sensor] = data[sensor] - predicted_mq

    # Save adjusted MQ data to a CSV file
    adjusted_mq.to_csv(output_csv, index=False)

    return adjusted_mq


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
