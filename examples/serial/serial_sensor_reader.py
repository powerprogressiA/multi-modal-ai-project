#!/usr/bin/env python3
"""
Generic serial reader for boards emitting lines (CSV/JSON). Prints each parsed line.
"""
import serial, argparse, json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    args = p.parse_args()

    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                print(json.dumps(obj))
            except Exception:
                print(line)

if __name__ == "__main__":
    main()
