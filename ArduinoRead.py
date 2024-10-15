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
    with open(filename, "a", newline="") as f:  # Open in append mode
        fieldnames = [
            "yyyy-mm-dd timestamp",
            "MQ6",
            "MQ5",
            "MQ4",
            "MQ7",
            "MQ3",
            "MQ8",
            "MQ2",
            "MQ135",
            "MQ9",
            "BMPTemperature",
            "Pressure(Pa)",
            "DHTTemperature",
            "Humidity",
            "Target",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only if the file is new (not existing)
        if f.tell() == 0:  # Check if the file is empty
            writer.writeheader()

        writer.writerows(data)


def parse_sensor_data(line):
    try:
        sensors = line.split(",")
        data_dict = {
            "MQ6": sensors[0].split(":")[1],
            "MQ5": sensors[1].split(":")[1],
            "MQ4": sensors[2].split(":")[1],
            "MQ7": sensors[3].split(":")[1],
            "MQ3": sensors[4].split(":")[1],
            "MQ8": sensors[5].split(":")[1],
            "MQ2": sensors[6].split(":")[1],
            "MQ135": sensors[7].split(":")[1],
            "MQ9": sensors[8].split(":")[1],
            "BMPTemperature": sensors[9].split(":")[1],  # BMP temperature
            "Pressure(Pa)": sensors[10].split(":")[1],  # Pressure
            "DHTTemperature": sensors[11].split(":")[1],  # DHT temperature
            "Humidity": sensors[12].split(":")[1],  # Humidity
        }
        return data_dict
    except IndexError:
        print(f"Error: Line format incorrect: {line}")
        return None


def read_serial_data(
    port="COM6",
    baudrate=115200,
    save_interval=5000,
    filename="Shir_Shiraz_2021_R2.csv",
    max_entries=15000,
):
    global MQSensorData  # Use global variable
    ser = serial.Serial(port, baudrate)
    entry_count = 0  # Counter for entries to manage save interval

    try:
        while entry_count < max_entries:  # Loop until max_entries is reached
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8").rstrip()
                data = parse_sensor_data(line)

                if data:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Get current time
                    sensor_data = {"yyyy-mm-dd timestamp": timestamp}
                    sensor_data.update(data)

                    # Use the filename (without extension) as the "Target"
                    target_value = filename.split(".")[0]
                    sensor_data["Target"] = target_value

                    MQSensorData.append(sensor_data)
                    print(MQSensorData[-1])  # Print the latest entry to verify

                    entry_count += 1  # Increment the entry counter

                    # Save data to file when reaching the specified limit
                    if entry_count % save_interval == 0:
                        save_data_to_file(MQSensorData, filename)
                        MQSensorData.clear()  # Clear data for the next batch

        print("Max entries reached. Stopping data collection.")

    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ser.close()
        # Ensure any remaining data is saved when exiting
        if MQSensorData:  # Only save if there is data
            save_data_to_file(MQSensorData, filename)


if __name__ == "__main__":
    COM_port = "COM" + input("Enter the COM port number: ")
    filename = input("Enter the filename for the final output (e.g., final_data.csv): ")

    # Load existing data if any
    MQSensorData = load_data_from_file(filename)

    # Start reading serial data and saving to the given filename
    read_serial_data(port=COM_port, baudrate=115200, filename=filename)
