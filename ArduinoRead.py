import serial
import time
import json

# Initialize the list to store data
MQ2Data = []


def load_data_from_file(filename="mq2data.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_data_to_file(data, filename="mq2data.json"):
    with open(filename, "w") as f:
        json.dump(data, f)


def read_serial_data(port="COM3", baudrate=9600, save_interval=1):
    ser = serial.Serial(port, baudrate)
    time.sleep(2)  # Allow some time for the connection to establish

    entry_count = 0  # Counter for entries to manage save interval

    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8").rstrip()
                data = line.split(",")
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Get current time

                # Append data and timestamp to the list
                MQ2Data.append((timestamp, data))
                print(MQ2Data[-1])  # Print the latest entry to verify

                # Increment the entry counter
                entry_count += 1

                # Save data to file at the specified interval
                if entry_count >= save_interval:
                    save_data_to_file(MQ2Data)
                    entry_count = 0  # Reset the counter

    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ser.close()
        save_data_to_file(MQ2Data)  # Ensure data is saved when exiting


if __name__ == "__main__":
    # Load existing data
    MQ2Data = load_data_from_file()
    read_serial_data(port="COM3", baudrate=9600)
