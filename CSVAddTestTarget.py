import csv

# Load the data from the CSV file
input_file = "MQSensorData.csv"
output_file = "MQSensorData_updated.csv"

with open(input_file, "r", newline="") as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)

    # Get the header and data separately
    header = rows[0]
    data = rows[1:]

    # Remove the 'timestamp' column
    header = header[1:]
    for row in data:
        del row[0]

    # Populate the 'Target' column with "airControl"
    for row in data:
        row.append("airControl")

# Write the updated data back to a new CSV file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(data)

print(f"File updated successfully. The new file is saved as {output_file}.")
