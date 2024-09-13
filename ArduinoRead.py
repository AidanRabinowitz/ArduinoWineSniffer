import serial
import time
import csv

# Initialize the list to store data
MQSensorData = []


def load_data_from_file(filename="MQSensorData.csv"):
    try:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        return []


def save_data_to_file(data, filename="MQSensorData.csv"):
    with open(filename, "w", newline="") as f:
        fieldnames = [
            "timestamp",
            "MQ3",
            "MQ135",
            "MQ8",
            "MQ5",
            "MQ7",
            "MQ4",
            "MQ6",
            "MQ2",
            "MQ9",
            "BMPTemperature",
            "Pressure(Pa)",
            "DHTTemperature",
            "Humidity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def parse_sensor_data(line, port):
    try:
        data_dict = {}
        sensors = line.split(",")
        if port == "COM3":
            data_dict = {
                "MQ3": sensors[0].split(":")[1],  # COM3 A0
                "MQ135": sensors[1].split(":")[1],  # COM3 A1
                "MQ8": sensors[2].split(":")[1],  # COM3 A2
                "MQ5": sensors[3].split(":")[1],  # COM3 A3
                "MQ7": sensors[4].split(":")[1],  # COM3 A4
            }
        elif port == "COM5":
            data_dict = {
                "MQ4": sensors[0].split(":")[1],  # COM5 A0
                "MQ6": sensors[1].split(":")[1],  # COM5 A1
                "MQ2": sensors[2].split(":")[1],  # COM5 A2
                "MQ9": sensors[3].split(":")[1],  # COM5 A3
                "BMPTemperature": sensors[4].split(":")[1],  # COM5 A3
                "Pressure(Pa)": sensors[5].split(":")[1],  # COM5 A3
                "DHTTemperature": sensors[6].split(":")[1],  # COM5 A3
                "Humidity": sensors[7].split(":")[1],  # COM5 A3
            }
        return data_dict
    except IndexError:
        print(f"Error: Line format incorrect from {port}: {line}")
        return None


def read_serial_data(ports=["COM5", "COM3"], baudrate=9600, save_interval=1):
    ser1 = serial.Serial(ports[0], baudrate)  # COM3
    ser2 = serial.Serial(ports[1], baudrate)  # COM5
    time.sleep(2)  # Allow some time for the connection to establish

    entry_count = 0  # Counter for entries to manage save interval

    try:
        while True:
            if ser1.in_waiting > 0 and ser2.in_waiting > 0:
                line1 = ser1.readline().decode("utf-8").rstrip()
                line2 = ser2.readline().decode("utf-8").rstrip()

                data1 = parse_sensor_data(line1, ports[0])
                data2 = parse_sensor_data(line2, ports[1])

                if data1 and data2:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Get current time
                    sensor_data = {"timestamp": timestamp}
                    sensor_data.update(data1)
                    sensor_data.update(data2)

                    MQSensorData.append(sensor_data)
                    print(MQSensorData[-1])  # Print the latest entry to verify

                    # Increment the entry counter
                    entry_count += 1

                    # Save data to file at the specified interval
                    if entry_count >= save_interval:
                        save_data_to_file(MQSensorData)
                        entry_count = 0  # Reset the counter

                # Wait 1 seconds before reading the next set of data
                time.sleep(1)

    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ser1.close()
        ser2.close()
        save_data_to_file(MQSensorData)  # Ensure data is saved when exiting


if __name__ == "__main__":
    # Load existing data
    MQSensorData = load_data_from_file()
    read_serial_data(ports=["COM5", "COM3"], baudrate=9600)
