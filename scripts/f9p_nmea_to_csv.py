#!/usr/bin/env python3
"""
Convert NMEA (stdin or file) to CSV with columns:
 time_utc,src,fix,lat_dd,lon_dd,alt_m,num_sats,hdop,raw_sentence

Usage:
  python3 scripts/f9p_nmea_to_csv.py ~/f9p_stream.log > ~/f9p_parsed.csv
  or pipe live:
  tail -f ~/f9p_stream.log | python3 scripts/f9p_nmea_to_csv.py > ~/f9p_parsed.csv
"""
import sys
from datetime import datetime

def nmea_to_dd(value, hemi):
    if not value:
        return ''
    try:
        f = float(value)
    except:
        return ''
    deg = int(f // 100)
    minutes = f - deg*100
    dd = deg + minutes/60.0
    if hemi in ('S','W'):
        dd = -dd
    return f"{dd:.8f}"

def parse_gga(fields):
    timeutc = fields[1] if len(fields) > 1 else ''
    lat = nmea_to_dd(fields[2] if len(fields) > 2 else '', fields[3] if len(fields) > 3 else '')
    lon = nmea_to_dd(fields[4] if len(fields) > 4 else '', fields[5] if len(fields) > 5 else '')
    fix = fields[6] if len(fields) > 6 else ''
    sats = fields[7] if len(fields) > 7 else ''
    hdop = fields[8] if len(fields) > 8 else ''
    alt = fields[9] if len(fields) > 9 else ''
    return timeutc, 'GGA', fix, lat, lon, alt, sats, hdop

def parse_rmc(fields):
    timeutc = fields[1] if len(fields) > 1 else ''
    status = fields[2] if len(fields) > 2 else ''
    lat = nmea_to_dd(fields[3] if len(fields) > 3 else '', fields[4] if len(fields) > 4 else '')
    lon = nmea_to_dd(fields[5] if len(fields) > 5 else '', fields[6] if len(fields) > 6 else '')
    date = fields[9] if len(fields) > 9 else ''
    ts = ''
    if date and timeutc:
        try:
            ts = datetime.strptime(date + timeutc.split('.')[0], "%d%m%y%H%M%S").isoformat()
        except:
            ts = ''
    return timeutc, 'RMC', status, lat, lon, '', '', ts

def process_line(line):
    line = line.strip()
    if not line.startswith('$'):
        return None
    if '*' in line:
        sentence, _ = line.split('*',1)
    else:
        sentence = line
    parts = sentence.split(',')
    tag = parts[0][1:]
    if tag.endswith('GGA'):
        return parse_gga(parts) + (line,)
    if tag.endswith('RMC'):
        return parse_rmc(parts) + (line,)
    return None

def main():
    print("time_utc,src,fix,lat_dd,lon_dd,alt_m,num_sats,hdop,raw")
    if len(sys.argv) > 1:
        f = open(sys.argv[1], 'r', errors='ignore')
    else:
        f = sys.stdin
    for l in f:
        try:
            r = process_line(l)
            if r:
                if len(r) == 9:  # GGA
                    timeutc, src, fix, lat, lon, alt, sats, hdop, raw = r
                    print(f'{timeutc},{src},{fix},{lat},{lon},{alt},{sats},{hdop},"{raw}"')
                else:  # RMC-like
                    timeutc, src, fix, lat, lon, alt, sats, hdop, raw = r
                    print(f'{timeutc},{src},{fix},{lat},{lon},{alt},{sats},{hdop},"{raw}"')
        except Exception as e:
            print(f'# parse error: {e}', file=sys.stderr)
    if f is not sys.stdin:
        f.close()

if __name__ == "__main__":
    main()
