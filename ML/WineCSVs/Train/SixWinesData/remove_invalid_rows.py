import pandas as pd


def remove_invalid_rows(file_path, output_cleaned_csv):
    """
    Removes rows that contain any letter (excluding the first row and the target column),
    as well as outliers in MQ columns using the IQR method.
    The cleaned data is saved to a new CSV.

    Parameters:
    file_path (str): Path to the input CSV file containing sensor data.
    output_cleaned_csv (str): Path to save the cleaned data as a CSV file.

    Returns:
    DataFrame: A DataFrame with the cleaned data.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Get the last column name (assumed to be the target column)
    last_column = data.columns[-1]

    # Get the columns to check (all except the last one)
    non_last_columns = data.columns[:-1]

    # Create a mask to filter out rows containing any letters in non-last columns
    mask = data[non_last_columns].apply(
        lambda row: not row.astype(str).str.contains(r"[a-zA-Z]").any(), axis=1
    )

    # Keep only valid rows from the initial check
    cleaned_data = data[mask | (data.index == 0)]

    # Remove outliers from MQ columns using the IQR method
    mq_columns = cleaned_data.columns[
        cleaned_data.columns.str.startswith("MQ")
    ]  # Adjust if necessary

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

    # Save the cleaned data to a new CSV
    cleaned_data.to_csv(output_cleaned_csv, index=False)

    return cleaned_data


input_csv = "ML\WineCSVs\Train\SixWinesData\SixWinesCombined.csv"
cleaned_csv = "ML\WineCSVs\Train\SixWinesData\SixWinesCombined_cleaned.csv"

cleaned_data = remove_invalid_rows(input_csv, cleaned_csv)
print(f"Cleaned data saved to {cleaned_csv}")
