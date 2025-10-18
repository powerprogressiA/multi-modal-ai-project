#!/usr/bin/env python3
"""
Read NMEA/UBX lines from a u-blox F9P connected via serial.
Edit PORT variable to match your device node.
"""
import serial

PORT = "/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_GNSS_receiver-if00"
BAUD = 38400

def main():
    with serial.Serial(PORT, BAUD, timeout=1) as s:
        print("Connected to", PORT)
        while True:
            line = s.readline().decode(errors="ignore").strip()
            if line:
                print(line)

if __name__ == "__main__":
    main()
