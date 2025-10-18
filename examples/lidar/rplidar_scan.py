#!/usr/bin/env python3
"""
Simple RPLIDAR scan example. Adjust PORT variable to /dev/serial-lidar or by-id path.
"""
from rplidar import RPLidar

PORT = '/dev/serial-lidar'

def main():
    lidar = RPLidar(PORT)
    print("Connected to", PORT)
    try:
        for i, scan in enumerate(lidar.iter_scans()):
            print(i, scan[:8])
            if i > 10:
                break
    finally:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()

if __name__ == "__main__":
    main()
