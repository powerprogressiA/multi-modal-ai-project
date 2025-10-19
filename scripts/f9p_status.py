#!/usr/bin/env python3
"""
Live status reader for F9P over serial. Prints GGA-derived fix and sats.

Usage:
  PORT=/dev/ttyACM1 BAUD=38400 python3 scripts/f9p_status.py
"""
import os, sys
import serial

PORT = os.environ.get('PORT', '/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_GNSS_receiver-if00')
BAUD = int(os.environ.get('BAUD', '38400'))

def nmea_to_dd(value, hemi):
    if not value:
        return None
    try:
        f = float(value)
    except:
        return None
    deg = int(f // 100)
    minutes = f - deg*100
    dd = deg + minutes/60.0
    if hemi in ('S','W'):
        dd = -dd
    return dd

def parse_gga(line):
    parts = line.split(',')
    timeutc = parts[1] if len(parts) > 1 else ''
    lat = nmea_to_dd(parts[2], parts[3]) if len(parts) > 3 else None
    lon = nmea_to_dd(parts[4], parts[5]) if len(parts) > 5 else None
    fix = parts[6] if len(parts) > 6 else ''
    sats = parts[7] if len(parts) > 7 else ''
    alt = parts[9] if len(parts) > 9 else ''
    return timeutc, fix, sats, lat, lon, alt

def main():
    try:
        s = serial.Serial(PORT, BAUD, timeout=1)
    except Exception as e:
        print("ERROR opening", PORT, ":", e, file=sys.stderr)
        sys.exit(1)
    print("Listening on", PORT, "baud", BAUD)
    print("CTRL-C to quit")
    try:
        while True:
            line = s.readline().decode('ascii', errors='ignore').strip()
            if not line:
                continue
            if line.startswith('$'):
                if 'GGA' in line:
                    timeutc, fix, sats, lat, lon, alt = parse_gga(line)
                    print(f"[GGA] time={timeutc} fix={fix} sats={sats} lat={lat} lon={lon} alt={alt}")
                else:
                    print(f"[NMEA] {line[:120]}")
    except KeyboardInterrupt:
        print("\nExiting")
    finally:
        s.close()

if __name__ == "__main__":
    main()
