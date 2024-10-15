import pandas as pd


def calculate_average_mq_values(data):
    # Ensure that the required columns are present
    feature_columns = ["MQ135", "MQ2", "MQ3", "MQ4", "MQ5", "MQ6", "MQ7", "MQ8", "MQ9"]
    target_column = "Target"

    # Check if required columns are in the DataFrame
    if target_column not in data.columns or not all(
        col in data.columns for col in feature_columns
    ):
        print("Error: Required columns are missing from the dataset.")
        return

    # Calculate average sensor values for each target class
    average_values = data.groupby(target_column)[feature_columns].mean().reset_index()

    # Print the average values
    print("Average MQ Sensor Values for Each Target:")
    print(average_values)


# Example usage in your main code
if __name__ == "__main__":
    # Load your dataset
    file_path = r"C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/src/data_analysis_for_NN/data_analysis_for_NN.csv"  # Update with your CSV file path
    data = pd.read_csv(file_path, header=0)

    # Call the function to calculate and print average MQ sensor values
    calculate_average_mq_values(data)
