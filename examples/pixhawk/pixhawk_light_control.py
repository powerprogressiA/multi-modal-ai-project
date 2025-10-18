#!/usr/bin/env python3
"""
Simple pymavlink example: waits for heartbeat then sends a single DO_SET_SERVO.
Make sure flight controller is disarmed and props are removed.
"""
from pymavlink import mavutil
import os, time

CONN = os.environ.get("CONN") or "/dev/serial/by-id/usb-ArduPilot_Pixhawk1_17003A000451333239393630-if00"
BAUD = int(os.environ.get("BAUD") or 57600)

def main():
    master = mavutil.mavlink_connection(CONN, baud=BAUD)
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print("Heartbeat received")
    servo_channel = 9
    pwm = 1500
    print("Setting servo channel", servo_channel, "to PWM", pwm)
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        servo_channel,
        pwm,0,0,0,0,0
    )
    time.sleep(1)

if __name__ == "__main__":
    main()
