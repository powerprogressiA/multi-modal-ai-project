#!/usr/bin/env python3
"""
Query u-blox for basic UBX information. Requires pyubx2 in the venv.

Usage:
  source .venv/bin/activate
  python3 scripts/f9p_ubx_probe.py
"""
import os, sys, time
import serial
from pyubx2 import UBXReader

PORT = os.environ.get('PORT', '/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_GNSS_receiver-if00')
BAUD = int(os.environ.get('BAUD', '38400'))

def main():
    try:
        s = serial.Serial(PORT, BAUD, timeout=1)
    except Exception as e:
        print("ERROR opening", PORT, ":", e, file=sys.stderr)
        sys.exit(1)

    ubr = UBXReader(s, readmode='poll', timeout=1)
    print("Listening UBX/NMEA on", PORT)
    start = time.time()
    try:
        while time.time() - start < 5:
            try:
                (raw, parsed) = ubr.read()
                if parsed is None:
                    continue
                print("MSG:", parsed.identity, getattr(parsed, 'timeGPS', ''))
            except Exception:
                pass
    finally:
        s.close()

if __name__ == "__main__":
    main()
