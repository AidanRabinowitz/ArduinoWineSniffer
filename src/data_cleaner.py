import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class DataCleaner:
    def __init__(self, input_folder=None, cleaning_method='z_score'):
        """
        Constructor with optional parameters for initialization.
        Allows default initialization without inputs.
        """
        self.input_folder = input_folder
        self.cleaning_method = cleaning_method

    # Setters and Getters for input_folder
    def set_input_folder(self, input_folder):
        self.input_folder = input_folder

    def get_input_folder(self):
        return self.input_folder

    # Setters and Getters for cleaning_method
    def set_cleaning_method(self, cleaning_method):
        self.cleaning_method = cleaning_method

    def get_cleaning_method(self):
        return self.cleaning_method

    def clean_data(self):
        """
        Cleans the data from all CSV files in the input folder.
        Returns a cleaned DataFrame.
        """
        # Initialize an empty dataframe to store all cleaned data
        all_cleaned_data = pd.DataFrame()

        # Loop through all CSV files in the folder
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".csv"):
                print(f"Processing file: {filename}")
                file_path = os.path.join(self.input_folder, filename)
                df = pd.read_csv(file_path)

                # Clean the data for each CSV file
                cleaned_df = self._clean_individual_file(df)

                # Append the cleaned data to the final dataframe
                all_cleaned_data = pd.concat([all_cleaned_data, cleaned_df])
                print("____________________")

        print(
            f"Data cleaning complete. Rows in cleaned data: {len(all_cleaned_data)}")
        return all_cleaned_data

    def save_cleaned_data(self, cleaned_df, output_file):
        """
        Saves the cleaned DataFrame to a specified output file.
        """
        cleaned_df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")

    def _clean_individual_file(self, df):
        print(f"Original number of rows: {len(df)}")
        l_before = len(df)

        # Step 1: Remove null values
        df = df.dropna()
        print(f"Rows after dropping nulls: {len(df)}")

        # Step 2: Convert sensor columns to numeric, forcing errors to NaN
        sensor_columns = [col for col in df.columns if col not in [
            'yyyy-mm-dd timestamp', 'Target']]
        for col in sensor_columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

        # Step 3: Remove rows with any NaN values
        df = df.dropna()
        print(f"Rows after removing NaNs from sensor columns: {len(df)}")

        # Step 3.5: Handle infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Step 4: Remove outliers from the sensor columns using the appropriate method
        if self.cleaning_method == 'z_score':
            df = self._remove_outliers_zscore(df, sensor_columns)
        elif self.cleaning_method == 'iqr':
            df = self._remove_outliers_iqr(df, sensor_columns)

        print(f"Total rows cleaned: {l_before - len(df)}")
        return df

    def _remove_outliers_zscore(self, df, columns):
        # Calculate the z-scores for all columns at once
        z_scores = np.abs(stats.zscore(df[columns]))
        # Create a mask to keep rows where all z-scores are less than 3
        mask = (z_scores < 3).all(axis=1)
        cleaned_df = df[mask]

        print(
            f"Rows before: {len(df)}, Rows cleaned after Z-Score: {len(df)-len(cleaned_df)}")
        return cleaned_df

    def _remove_outliers_iqr(self, df, columns):
        # Use IQR to detect and remove outliers
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            cleaned_df = df[~((df[col] < (Q1 - 1.5 * IQR)) |
                              (df[col] > (Q3 + 1.5 * IQR)))]
        print(
            f"Rows before: {len(df)}, Rows cleaned after IQR: {len(df)-len(cleaned_df)}")
        return cleaned_df

    def plot_histograms(self, df):
        """
        Generates histograms for sensor data, color-coded by 'Target'.
        """
        sensor_columns = [col for col in df.columns if col.startswith('MQ')]
        target_values = df['Target'].unique()

        for col in sensor_columns:
            plt.figure(figsize=(10, 6))
            for target in target_values:
                subset = df[df['Target'] == target]
                plt.hist(subset[col], bins=30, alpha=0.5,
                         label=f'Target {target}')

            # Set title font size
            plt.title(f'Distribution of {col} values by Target', fontsize=30)
            # Set x-axis label font size
            plt.xlabel(f'{col} Value', fontsize=24)
            plt.ylabel('Frequency', fontsize=24)  # Set y-axis label font size

            # Set the font size for tick labels
            # Adjust label size for both axes
            plt.tick_params(axis='both', labelsize=22)

            plt.legend(fontsize=12)  # Set legend font size
            plt.show()
