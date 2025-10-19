#!/usr/bin/env python3
"""
Send UBX polls for MON-ANT and NAV-PVT and print responses (hex + parsed when possible).

Usage:
  source .venv/bin/activate
  PORT=/dev/ttyACM1 BAUD=38400 python3 scripts/f9p_ubx_poll.py

Adjust PORT/BAUD env vars as needed.
"""
import os, sys, time, binascii
try:
    import serial
except Exception as e:
    print("Please install pyserial (pip install pyserial) in the venv", file=sys.stderr)
    raise

PORT = os.environ.get("PORT", "/dev/ttyACM1")
BAUD = int(os.environ.get("BAUD", "38400"))

# UBX poll frames (no payload)
# MON-ANT: class 0x0A id 0x13
MON_ANT = bytes([0xB5,0x62, 0x0A,0x13, 0x00,0x00, 0x1D,0x44])
# NAV-PVT: class 0x01 id 0x07
NAV_PVT = bytes([0xB5,0x62, 0x01,0x07, 0x00,0x00, 0x08,0x11])

print("Opening", PORT, "baud", BAUD)
try:
    s = serial.Serial(PORT, BAUD, timeout=1)
except Exception as e:
    print("ERROR opening port:", e, file=sys.stderr)
    sys.exit(1)

# optional UBX parsing helper
use_pyubx2 = False
try:
    from pyubx2 import UBXReader
    use_pyubx2 = True
    ubr = UBXReader(s)
except Exception:
    ubr = None

def read_for(seconds=4):
    """Read bytes for 'seconds', print any UBX packets (hex) and plain NMEA lines."""
    end = time.time() + seconds
    buf = b""
    while time.time() < end:
        try:
            chunk = s.read(256)
        except Exception as e:
            print("read error:", e, file=sys.stderr)
            break
        if not chunk:
            continue
        buf += chunk
        # print any NMEA lines
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip(b"\r")
            if line.startswith(b'$'):
                try:
                    print("NMEA:", line.decode('ascii', errors='ignore'))
                except:
                    print("NMEA (raw):", binascii.hexlify(line))
            elif line.startswith(b'\xb5\x62'):
                # full UBX packet may be in 'line' or split; use hex
                print("UBX (raw):", binascii.hexlify(b'\xb5\x62'+line[2:]) if not line.startswith(b'\xb5\x62') else binascii.hexlify(line))
            else:
                # maybe partial binary, print hex
                print("BIN (hex):", binascii.hexlify(line))
        # also try reading UBX via pyubx2 if available
        if use_pyubx2:
            try:
                raw, parsed = ubr.read()  # non-blocking if stream has data
                if parsed:
                    identity = getattr(parsed, "identity", None)
                    if identity:
                        print("UBX parsed:", identity)
                        print(parsed)
            except Exception:
                # ignore parse errors
                pass

try:
    print("Sending MON-ANT poll...")
    s.write(MON_ANT)
    s.flush()
    read_for(4)

    print("Sending NAV-PVT poll...")
    s.write(NAV_PVT)
    s.flush()
    read_for(4)

    print("Now switching to passive read for a few seconds to capture NMEA/UBX stream...")
    read_for(6)

finally:
    s.close()
    print("Done.")
