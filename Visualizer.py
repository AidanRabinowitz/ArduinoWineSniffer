import json
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Path to the JSON file
JSON_FILE = "mq2data.json"

# Initialize data storage
timestamps = []
sensor1_values = []
sensor2_values = []


def load_data():
    global timestamps, sensor1_values, sensor2_values
    try:
        with open(JSON_FILE, "r") as f:
            data = json.load(f)
            timestamps = [entry[0] for entry in data]
            sensor1_values = [
                float(entry[1][0].split(":")[-1].strip()) for entry in data
            ]
            # If you have two sensors, adjust the indexing accordingly
            # sensor2_values = [float(entry[1][1].split(':')[-1].strip()) for entry in data]
            print("Loaded data successfully.")
            print("Timestamps:", len(timestamps))
            print("Sensor 1 Values:", len(sensor1_values))
            # print("Sensor 2 Values:", len(sensor2_values))

    except FileNotFoundError:
        print("Error: File not found.")
    except (ValueError, IndexError) as e:
        print(f"Error loading data: {e}")


def update(frame):
    load_data()

    if len(timestamps) != len(sensor1_values):
        print("Error: Inconsistent data dimensions.")
        return

    plt.cla()  # Clear the previous plot
    plt.plot(timestamps, sensor1_values, label="Sensor 1")
    # If you have two sensors, add another plot for sensor2_values
    # plt.plot(timestamps, sensor2_values, label='Sensor 2')
    plt.xlabel("Time")
    plt.ylabel("Sensor Values")
    plt.title("Real-time Sensor Data")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="upper left")
    plt.tight_layout()


if __name__ == "__main__":
    # Create a figure and an axes
    fig = plt.figure()
    ani = FuncAnimation(
        fig, update, interval=1000
    )  # Update every 1000 milliseconds (1 second)
    plt.show()
