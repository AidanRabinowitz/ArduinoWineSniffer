import datetime as dt
import csv
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def load_data_from_file(filename="MQSensorData.csv"):
    """Load data from the CSV file."""
    try:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return []


def animate(frame, xs, ys_dict):
    """Function called periodically by Matplotlib as an animation."""
    data = load_data_from_file()
    if data:
        latest_entry = data[-1]
        try:
            timestamp = dt.datetime.strptime(
                latest_entry["timestamp"], "%Y-%m-%d %H:%M:%S"
            )
            # Append timestamp
            xs.append(timestamp)

            # Update sensor readings
            for sensor in ys_dict.keys():
                value_str = latest_entry.get(sensor)
                if value_str is not None:
                    try:
                        value = float(value_str)
                    except ValueError:
                        value = None
                else:
                    value = None

                # Only append valid values
                if value is not None:
                    ys_dict[sensor].append(value)

        except ValueError:
            print(f"Skipping entry due to invalid data: {latest_entry}")

    # Limit x and y lists to the more recent items
    size_limit = 20
    xs = xs[-size_limit:]
    for sensor in ys_dict.keys():
        ys_dict[sensor] = ys_dict[sensor][-size_limit:]

    # Draw x and y lists
    ax.clear()
    for sensor, ys in ys_dict.items():
        if ys:  # Only plot if there is data
            ax.plot(xs, ys, label=sensor)

    # (Re)Format plot
    plt.grid()
    plt.xticks(rotation=45, ha="right")
    plt.subplots_adjust(bottom=0.30)
    plt.title("Sensor Data over Time")
    plt.ylabel("Sensor Reading")
    plt.xlabel("Time")
    plt.legend()


if __name__ == "__main__":
    # Create figure
    fig, ax = plt.subplots()

    # Create empty data series
    x_data = []
    y_data_dict = {
        "MQ3": [],
        "MQ135": [],
        "MQ8": [],
        "MQ5": [],
        "MQ7": [],
        "MQ4": [],
        "MQ6": [],
        "MQ2": [],
        "MQ9": [],
    }

    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(
        fig, animate, fargs=(x_data, y_data_dict), interval=100
    )
    plt.show()
