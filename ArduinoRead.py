import serial
import time
import csv

# Initialize the list to store data
MQSensorData = []

# AIDAN COM5 = JESS COM3
# AIDAN COM3 = JESS COM4


def load_data_from_file(filename="warmuptest(20degEnvTemp).csv"):
    try:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        return []


def save_data_to_file(data, filename="warmuptest(20degEnvTemp).csv"):
    with open(filename, "w", newline="") as f:
        fieldnames = [
            "yyyy-mm-dd timestamp",
            "MQ6",  #
            "MQ5",
            "MQ4",
            "MQ7",
            "MQ3",  #
            "MQ8",  #
            "MQ2",  #
            "MQ135",  #
            "MQ9",  #
            "BMPTemperature",
            "Pressure(Pa)",
            "DHTTemperature",
            "Humidity",
            "Target",
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
                "MQ6": sensors[0].split(":")[1],  # COM3 A0
                "MQ5": sensors[1].split(":")[1],  # COM3 A1
                "MQ4": sensors[2].split(":")[1],  # COM3 A2
                "MQ7": sensors[3].split(":")[1],  # COM3 A3
                "MQ3": sensors[4].split(":")[1],  # COM3 A4
            }
        elif port == "COM5":
            data_dict = {
                "MQ8": sensors[0].split(":")[1],  # COM5 A0
                "MQ2": sensors[1].split(":")[1],  # COM5 A1
                "MQ135": sensors[2].split(":")[1],  # COM5 A2
                "MQ9": sensors[3].split(":")[1],  # COM5 A3
                "BMPTemperature": sensors[4].split(":")[1],  # COM5 A4
                "Pressure(Pa)": sensors[5].split(":")[1],  # COM5 A5
                "DHTTemperature": sensors[6].split(":")[1],  # COM5 A6
                "Humidity": sensors[7].split(":")[1],  # COM5 A7
            }
        return data_dict
    except IndexError:
        print(f"Error: Line format incorrect from {port}: {line}")
        return None


def read_serial_data(ports=["COM3", "COM5"], baudrate=9600, save_interval=1):
    ser1 = serial.Serial(ports[0], baudrate)  # COM3
    ser2 = serial.Serial(ports[1], baudrate)  # COM5
    # time.sleep(2)  # Allow some time for the connection to establish

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
                    sensor_data = {"yyyy-mm-dd timestamp": timestamp}
                    sensor_data.update(data1)
                    sensor_data.update(data2)

                    # Add the hardcoded "Target" value
                    sensor_data["Target"] = "warmuptest(20degEnvTemp)"

                    MQSensorData.append(sensor_data)
                    print(MQSensorData[-1])  # Print the latest entry to verify

                    # Increment the entry counter
                    entry_count += 1

                    # Save data to file at the specified interval
                    if entry_count >= save_interval:
                        save_data_to_file(MQSensorData)
                        entry_count = 0  # Reset the counter

            # Wait 0.5 seconds before reading the next set of data
            # time.sleep(0.5)

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
