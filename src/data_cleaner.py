import os
import pandas as pd
import numpy as np
from scipy import stats


class DataCleaner:
    def __init__(self, input_folder, output_file, cleaning_method='z_score'):
        self.input_folder = input_folder
        self.output_file = output_file
        # 'z_score' for training, 'iqr' for testing
        self.cleaning_method = cleaning_method

    def clean_data(self):
        # Initialize an empty dataframe to store all cleaned data
        all_cleaned_data = pd.DataFrame()

        # Loop through all CSV files in the folder
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".csv"):
                print(filename)
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
        print(f"Original number of rows: {len(df)}")

        # Step 1: Remove null values
        df = df.dropna()
        print(f"Rows after dropping nulls: {len(df)}")

        # Step 2: Convert sensor columns to numeric, forcing errors to NaN
        sensor_columns = [col for col in df.columns if col not in [
            'yyyy-mm-dd timestamp', 'Target']]
        for col in sensor_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Step 3: Remove rows with any NaN values
        df = df.dropna()
        print(f"Rows after removing NaNs from sensor columns: {len(df)}")

        # Step 4: Remove outliers from the sensor columns using the appropriate method
        if self.cleaning_method == 'z_score':
            df = self._remove_outliers_zscore(df, sensor_columns)
            print(
                f"Rows after removing outliers with Z-Score method: {len(df)}")
        elif self.cleaning_method == 'iqr':
            df = self._remove_outliers_iqr(df, sensor_columns)
            print(f"Rows after removing outliers with IQR method: {len(df)}")

        return df

    def _remove_outliers_zscore(self, df, columns):
        # Calculate the z-scores for all columns at once
        z_scores = np.abs(stats.zscore(df[columns]))
        # Create a mask to keep rows where all z-scores are less than 3
        mask = (z_scores < 3).all(axis=1)
        cleaned_df = df[mask]

        print(
            f"Original rows: {len(df)}, Cleaned rows after Z-Score: {len(cleaned_df)}")
        return cleaned_df

    def _remove_outliers_iqr(self, df, columns):
        # Use IQR to detect and remove outliers
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 1.5 * IQR)) |
                      (df[col] > (Q3 + 1.5 * IQR)))]
        return df
