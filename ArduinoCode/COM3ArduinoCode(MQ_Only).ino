// Black cable Uno with COM3



void setup() {
  Serial.begin(9600);
  pinMode(A0, INPUT);
  pinMode(A1, INPUT);
  pinMode(A2, INPUT);
  pinMode(A3, INPUT);
  pinMode(A4, INPUT); 
}

void loop() {

  int sensorValue0 = analogRead(A0); 
  int sensorValue1 = analogRead(A1); 
  int sensorValue2 = analogRead(A2); 
  int sensorValue3 = analogRead(A3); 
  int sensorValue4 = analogRead(A4); 
  // int sensorValue5 = analogRead(A5); 

  Serial.print("Sensor_A0:"); 
  Serial.print(sensorValue0);
  Serial.print(",Sensor_A1:"); 
  Serial.print(sensorValue1);
  Serial.print(",Sensor_A2:"); 
  Serial.print(sensorValue2);
  Serial.print(",Sensor_A3:"); 
  Serial.print(sensorValue3);
  Serial.print(",Sensor_A4:");
  Serial.println(sensorValue4);
  // Serial.print(",Sensor_A5:");
  // Serial.println(sensorValue5);

  delay(1000); // Wait 0.1 second before the next reading
}