#include <Arduino.h>

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println();
  Serial.println("ESP32 test firmware starting...");
}

void loop() {
  // Print a simple heartbeat with millis() and a test analog read
  unsigned long t = millis();
  int raw = analogRead(34); // ADC pin (change if your board uses different pin)
  float voltage = (raw / 4095.0) * 3.3;
  Serial.printf("time=%lu ms, adc34=%d, v=%.3f V\n", t, raw, voltage);
  delay(1000);
}
