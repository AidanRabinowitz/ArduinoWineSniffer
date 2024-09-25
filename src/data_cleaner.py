import os
import pandas as pd
import numpy as np
from scipy import stats


class DataCleaner:
    def __init__(self, input_folder, output_file):
        self.input_folder = input_folder
        self.output_file = output_file

    def clean_data(self):
        # Initialize an empty dataframe to store all cleaned data
        all_cleaned_data = pd.DataFrame()

        # Loop through all CSV files in the folder
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.input_folder, filename)
                df = pd.read_csv(file_path)

                # Clean the data for each CSV file
                cleaned_df = self._clean_individual_file(df)

                # Append the cleaned data to the final dataframe
                all_cleaned_data = pd.concat([all_cleaned_data, cleaned_df])

        # Save the final cleaned dataframe to the output file
        all_cleaned_data.to_csv(self.output_file, index=False)
        print(f"All cleaned data saved to {self.output_file}")

    def _clean_individual_file(self, df):
        # Step 1: Remove null values
        df = df.dropna()

        # Step 2: Convert sensor columns to numeric, forcing errors to NaN
        sensor_columns = [col for col in df.columns if col not in [
            'yyyy-mm-dd timestamp', 'Target']]
        for col in sensor_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Step 3: Remove rows with any NaN values
        df = df.dropna()

        # Step 4: Remove outliers from the sensor columns using z-score method
        df = self._remove_outliers(df, sensor_columns)

        return df

    def _remove_outliers(self, df, columns):
        # Use z-score to detect and remove outliers (z-score > 3)
        for col in columns:
            z_scores = np.abs(stats.zscore(df[col]))
            df = df[z_scores < 3]
        return df
