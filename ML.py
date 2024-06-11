import json


def load_data_from_file(filename="mq2data.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


MQ2Data = load_data_from_file()


def process_data():
    if not MQ2Data:
        print(
            "No data available. Please ensure that the serial_reader script is running and data has been collected."
        )
        return

    # Example processing - just printing the data
    for entry in MQ2Data:
        timestamp, data = entry
        print(f"Time: {timestamp}, Data: {data}")


if __name__ == "__main__":
    process_data()
