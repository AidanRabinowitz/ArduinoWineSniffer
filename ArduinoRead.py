import serial
import time
import csv

# Initialize the list to store data
MQSensorData = []


def load_data_from_file(filename):
    try:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        return []


def save_data_to_file(data, filename):
    with open(filename, "w", newline="") as f:
        fieldnames = [
            "yyyy-mm-dd timestamp",
            "MQ6",
            "MQ5",
            "MQ4",
            "MQ7",
            "MQ3",  # COM3
            "MQ8",
            "MQ2",
            "MQ135",
            "MQ9",  # COM5
            "BMPTemperature",
            "Pressure(Pa)",
            "DHTTemperature",
            "Humidity",
            "Target",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def parse_sensor_data(line):
    try:
        sensors = line.split(",")
        data_dict = {
            "MQ6": sensors[0].split(":")[1],  # COM3 A0
            "MQ5": sensors[1].split(":")[1],  # COM3 A1
            "MQ4": sensors[2].split(":")[1],  # COM3 A2
            "MQ7": sensors[3].split(":")[1],  # COM3 A3
            "MQ3": sensors[4].split(":")[1],  # COM3 A4
            "MQ8": sensors[5].split(":")[1],  # COM5 A5
            "MQ2": sensors[6].split(":")[1],  # COM5 A6
            "MQ135": sensors[7].split(":")[1],  # COM5 A7
            "MQ9": sensors[8].split(":")[1],  # COM5 A8
            "BMPTemperature": sensors[9].split(":")[1],  # BMP temperature
            "Pressure(Pa)": sensors[10].split(":")[1],  # Pressure
            "DHTTemperature": sensors[11].split(":")[1],  # DHT temperature
            "Humidity": sensors[12].split(":")[1],  # Humidity
        }
        return data_dict
    except IndexError:
        print(f"Error: Line format incorrect: {line}")
        return None


def read_serial_data(port="COM6", baudrate=115200, save_interval=1, filename="data.csv"):
    ser = serial.Serial(port, baudrate)
    entry_count = 0  # Counter for entries to manage save interval

    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8").rstrip()
                data = parse_sensor_data(line)

                if data:
                    timestamp = time.strftime(
                        "%Y-%m-%d %H:%M:%S")  # Get current time
                    sensor_data = {"yyyy-mm-dd timestamp": timestamp}
                    sensor_data.update(data)

                    # Use the filename (without extension) as the "Target"
                    target_value = filename.split(".")[0]
                    sensor_data["Target"] = target_value

                    MQSensorData.append(sensor_data)
                    print(MQSensorData[-1])  # Print the latest entry to verify

                    # Increment the entry counter
                    entry_count += 1

                    # Save data to file at the specified interval
                    if entry_count >= save_interval:
                        save_data_to_file(MQSensorData, filename)
                        entry_count = 0  # Reset the counter

            # # Wait 0.5 seconds before reading the next set of data
            # time.sleep(0.5)

    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ser.close()
        # Ensure data is saved when exiting
        save_data_to_file(MQSensorData, filename)


if __name__ == "__main__":
    # Prompt for COM port:
    # COM6 - AIDAN
    # COM9 -JESS
    COM_port = "COM"+input("Enter the COM port number:")

    # Prompt the user for the filename
    filename = input("Enter the filename (e.g., BlackTieR2_3009.csv): ")

    # Load existing data if any
    MQSensorData = load_data_from_file(filename)

    # Start reading serial data and saving to the given filename
    read_serial_data(port=COM_port, baudrate=115200, filename=filename)
