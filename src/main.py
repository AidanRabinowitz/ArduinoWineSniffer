import os
from data_cleaner import DataCleaner
from deep_learning import DeepLearning


def process_folder(folder_path, output_file):
    # Initialize the DataCleaner with folder path and output file
    cleaner = DataCleaner(folder_path)
    df = cleaner.clean_data()  # Call the cleaning function
    cleaner.save_cleaned_data(df, output_file)


def main():
    # Path to the folder containing CSV files
    folder_path = "ArduinoWineSniffer/src/Data/2309_data"
    output_file = (
        "src/cleaned_data_allwines.csv"  # Path where cleaned data will be saved
    )

    # Process the folder to clean data
    process_folder(folder_path, output_file)

    # Now pass the cleaned data to the DeepLearning class
    model = DeepLearning(output_file)
    model.train_and_evaluate()  # Assuming you have a method to train and evaluate


if __name__ == "__main__":
    main()
