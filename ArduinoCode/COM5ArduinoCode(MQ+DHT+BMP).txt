// Blue cable Uno with COM5
// #include <DHT11.h>
// DHT11 dht11(A7);
#include <Wire.h>
#include <SPI.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP280.h>
#include <dht.h>

#define BMP_SCK 13    //SCL
#define BMP_MISO 12   //SDO
#define BMP_MOSI 11   //SDA
#define BMP_CS 10     //CSB
#define DHT11_PIN 4

//Adafruit_BMP280 bme; // I2C
Adafruit_BMP280 bme(BMP_CS, BMP_MOSI, BMP_MISO,  BMP_SCK);
dht DHT; 

void setup() {
  Serial.begin(9600);
  if (!bme.begin()) {  
  Serial.println("Could not find a valid BMP280 sensor, check wiring!");
  while (1);
  }
  pinMode(A0, INPUT);
  pinMode(A1, INPUT);
  pinMode(A2, INPUT);
  pinMode(A3, INPUT);
  // pinMode(A4, INPUT); 
  // pinMode(A5, INPUT); 

}

void loop() {
  // Pressure sensor
  Serial.print("BMP.Temperature(*C):");
  Serial.print(bme.readTemperature());
  
  Serial.print(",Pressure(Pa):");
  Serial.print(bme.readPressure());

  //       // Humidity sensor
  int chk = DHT.read11(DHT11_PIN);
  Serial.print(",DHT.Temperature:");
  Serial.print(DHT.temperature);
  Serial.print(",Humidity:");
  Serial.print(DHT.humidity);
  //   Serial.println();
  int sensorValue0 = analogRead(A0); 
  int sensorValue1 = analogRead(A1); 
  int sensorValue2 = analogRead(A2); 
  int sensorValue3 = analogRead(A3); 
  // int sensorValue4 = analogRead(A4); 
  // int sensorValue5 = analogRead(A5); 

  Serial.print(",Sensor_A0:"); 
  Serial.print(sensorValue0);
  Serial.print(",Sensor_A1:"); 
  Serial.print(sensorValue1);
  Serial.print(",Sensor_A2:"); 
  Serial.print(sensorValue2);
  Serial.print(",Sensor_A3:"); 
  Serial.print(sensorValue3);
  // Serial.print(",Sensor_A4:");
  // Serial.println(sensorValue4);
  // Serial.print(",Sensor_A5:");
  // Serial.println(sensorValue5);
  Serial.println();

  delay(1000); // Wait 0.1 second before the next reading
}