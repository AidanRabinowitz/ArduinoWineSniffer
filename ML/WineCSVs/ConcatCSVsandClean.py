import os
import pandas as pd


def remove_invalid_rows(file_path):
    """
    Removes rows that contain any non-numeric value (excluding the first row and the target column),
    as well as outliers in MQ and other specified columns using the IQR method.

    Parameters:
    file_path (str): Path to the input CSV file containing sensor data.

    Returns:
    DataFrame: A DataFrame with the cleaned data.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Define the columns to clean (MQ columns + BMPTemperature, Pressure(Pa), DHTTemperature, Humidity)
    columns_to_clean = [col for col in data.columns if col.startswith("MQ")]
    columns_to_clean += ["BMPTemperature", "Pressure(Pa)", "DHTTemperature", "Humidity"]

    # Replace non-numeric values with NaN in the specified columns
    data[columns_to_clean] = data[columns_to_clean].apply(
        pd.to_numeric, errors="coerce"
    )

    # Drop rows with NaN values in any of the specified columns
    cleaned_data = data.dropna(subset=columns_to_clean)

    # Remove outliers from MQ and other columns using the IQR method
    for column in columns_to_clean:
        # Calculate IQR
        Q1 = cleaned_data[column].quantile(0.25)
        Q3 = cleaned_data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 5 * IQR
        upper_bound = Q3 + 5 * IQR

        # Keep only rows within the IQR range (removing outliers)
        non_outlier_mask = (cleaned_data[column] >= lower_bound) & (
            cleaned_data[column] <= upper_bound
        )
        cleaned_data = cleaned_data[non_outlier_mask]

    return cleaned_data


def shuffle_data(data):
    """
    Shuffles the rows of the DataFrame to prevent overfitting.

    Parameters:
    data (DataFrame): The DataFrame to shuffle.

    Returns:
    DataFrame: A shuffled DataFrame.
    """
    return data.sample(frac=1).reset_index(drop=True)


def process_folder(input_folder, output_file, shuffle=False):
    """
    Processes all CSV files in a folder, removes invalid rows and outliers,
    combines the cleaned data, and optionally shuffles it.

    Parameters:
    input_folder (str): Path to the folder containing CSV files.
    output_file (str): Path to save the combined cleaned data.
    shuffle (bool): If True, shuffle the combined cleaned data before saving.
    """
    combined_data = (
        pd.DataFrame()
    )  # Initialize an empty DataFrame to store combined data

    # Iterate through all CSV files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing {file_path}...")

            # Clean the data
            cleaned_data = remove_invalid_rows(file_path)

            # Append cleaned data to the combined DataFrame
            combined_data = pd.concat([combined_data, cleaned_data], ignore_index=True)

    # Shuffle the data if shuffle is True
    if shuffle:
        combined_data = shuffle_data(combined_data)

    # Save the combined cleaned data to the output CSV
    combined_data.to_csv(output_file, index=False)
    print(f"Combined cleaned data saved to {output_file}")


input_folder = "ML/WineCSVs/Train/FinalTrainSet"
output_file = "ML/WineCSVs/Train/cleanedCombinedTrainSet/combined_cleaned_data.csv"
process_folder(input_folder, output_file, shuffle=True)  # Shuffle the data
