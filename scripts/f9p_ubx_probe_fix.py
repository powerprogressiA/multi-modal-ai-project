#!/usr/bin/env python3
# UBX probe tolerant of pyubx2 versions. Prints detected UBX message types.
import os, sys, time
import serial

PORT = os.environ.get('PORT', '/dev/ttyACM1')
BAUD = int(os.environ.get('BAUD', '38400'))

try:
    from pyubx2 import UBXReader
except Exception as e:
    print("pyubx2 not available or import error:", e, file=sys.stderr)
    sys.exit(1)

def main():
    try:
        s = serial.Serial(PORT, BAUD, timeout=1)
    except Exception as e:
        print("ERROR opening", PORT, ":", e, file=sys.stderr)
        sys.exit(1)

    # Try creating UBXReader with a couple of common signatures
    ubr = None
    for args in ((), (s,), (s,)):
        try:
            # try common init forms; UBXReader usually accepts the stream only
            ubr = UBXReader(s)
            break
        except TypeError:
            try:
                ubr = UBXReader(s)  # fallback
                break
            except Exception:
                ubr = None
                break
        except Exception:
            ubr = None
            break

    if ubr is None:
        print("Failed to instantiate UBXReader. Exiting.", file=sys.stderr)
        s.close()
        sys.exit(1)

    print("Listening for UBX & NMEA on", PORT, "baud", BAUD)
    start = time.time()
    try:
        while time.time() - start < 8:   # read for 8 seconds
            try:
                raw, parsed = ubr.read()   # returns (raw, parsed) or (None, None)
                if parsed is None:
                    continue
                # print a short summary
                identity = getattr(parsed, "identity", None)
                if identity is None:
                    # fallback: show class ID / msg ID if available
                    print("Parsed:", parsed)
                else:
                    print("MSG:", identity)
                    # if MON-ANT or NAV-PVT print the whole message
                    if identity in ("MON-ANT", "NAV-PVT"):
                        print(parsed)
            except Exception as ex:
                # ignore isolated parse errors
                # print the exception once for debugging
                print("read error:", ex)
                time.sleep(0.1)
    finally:
        s.close()

if __name__ == "__main__":
    main()
