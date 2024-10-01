import os
import pandas as pd


def remove_invalid_rows(file_path):
    """
    Removes rows that contain any letter (excluding the first row and the target column),
    as well as outliers in MQ columns using the IQR method.

    Parameters:
    file_path (str): Path to the input CSV file containing sensor data.

    Returns:
    DataFrame: A DataFrame with the cleaned data.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Get the columns that start with 'MQ'
    mq_columns = [col for col in data.columns if col.startswith("MQ")]

    # Create a mask to filter out rows containing any letters in MQ columns
    mask = data[mq_columns].apply(
        lambda row: not row.astype(str).str.contains(r"[a-zA-Z]").any(), axis=1
    )

    # Keep only valid rows from the initial check
    cleaned_data = data[mask]

    # Remove outliers from MQ columns using the IQR method
    for column in mq_columns:
        # Convert column to numeric, forcing errors to NaN
        cleaned_data[column] = pd.to_numeric(cleaned_data[column], errors="coerce")

        # Drop rows with NaN values created by the conversion
        cleaned_data = cleaned_data.dropna(subset=[column])

        # Calculate IQR
        Q1 = cleaned_data[column].quantile(0.25)
        Q3 = cleaned_data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 5 * IQR
        upper_bound = Q3 + 5 * IQR

        # Create a mask for non-outliers
        non_outlier_mask = (cleaned_data[column] >= lower_bound) & (
            cleaned_data[column] <= upper_bound
        )
        cleaned_data = cleaned_data[non_outlier_mask]

    return cleaned_data


def process_folder(input_folder, output_file):
    """
    Processes all CSV files in a folder, removes invalid rows and outliers,
    and combines the cleaned data into one CSV file.

    Parameters:
    input_folder (str): Path to the folder containing CSV files.
    output_file (str): Path to save the combined cleaned data.
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

    # Save the combined cleaned data to the output CSV
    combined_data.to_csv(output_file, index=False)
    print(f"Combined cleaned data saved to {output_file}")


# Example usage
input_folder = "ML\WineCSVs\Train\Train3009"  # Replace with your folder path
output_file = "combined_cleaned_data.csv"  # Replace with your desired output file name
process_folder(input_folder, output_file)
