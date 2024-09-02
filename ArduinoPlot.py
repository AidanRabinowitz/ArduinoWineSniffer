import json
import matplotlib.pyplot as plt
from datetime import datetime


def load_data_from_file(filename="MQSensorData.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return []


def plot_sensor_data(data):
    timestamps = []
    sensor_values = {
        "MQ-4": [],
        "MQ-7": [],
        "MQ-2": [],
        "MQ-135": [],
        "MQ-8": [],
        "MQ-6": [],
        "MQ-9": [],
        "MQ-3": [],
        "MQ-5": [],
    }

    # Extract timestamps and sensor values from data
    for entry in data:
        timestamps.append(datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S"))
        for sensor in sensor_values.keys():
            sensor_values[sensor].append(int(entry[sensor]))

    # Plot each sensor's data
    fig, axs = plt.subplots(len(sensor_values), 1, figsize=(10, 15), sharex=True)
    fig.suptitle("MQ Sensor Readings Over Time")

    for i, (sensor, values) in enumerate(sensor_values.items()):
        axs[i].plot(timestamps, values, label=sensor)
        axs[i].set_ylabel(sensor)
        axs[i].legend(loc="upper right")
        axs[i].grid(True)

    axs[-1].set_xlabel("Time")
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    data = load_data_from_file()
    if data:
        plot_sensor_data(data)
