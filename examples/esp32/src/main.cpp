#include <Arduino.h>
#include <SPI.h>
#include <SD.h>

const int SENSOR_PIN = 34; // ADC pin
const int SD_CS = 5;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("ESP32 sensor logger starting...");
  if (!SD.begin(SD_CS)) {
    Serial.println("SD card mount failed");
  } else {
    Serial.println("SD mounted.");
  }
}

void loop() {
  int val = analogRead(SENSOR_PIN);
  float voltage = (val / 4095.0) * 3.3;
  String line = String(millis()) + "," + String(voltage, 4);
  Serial.println(line);

  if (SD.begin(SD_CS)) {
    File f = SD.open("/log.csv", FILE_APPEND);
    if (f) {
      f.println(line);
      f.close();
    }
  }
  delay(500);
}
