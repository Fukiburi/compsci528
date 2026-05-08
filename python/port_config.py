"""
Set up ESP32 connection with python code.
"""
import serial
import serial.tools.list_ports
import sys
import argparse

from python.SerialReader import SerialReader
from python.constants import SAMPLE_HZ, BAUD_RATE, WINDOW_SEC

def find_port() -> str:
    """Auto-detect the first USB-serial port."""
    ports = serial.tools.list_ports.comports()
    usb = [p for p in ports if "usb" in p.device.lower() or "usbserial" in p.device.lower()]
    if usb:
        return usb[0].device
    if ports:
        return ports[0].device
    print("[ERROR] No serial ports found. Plug in your ESP32 or specify --port.", file=sys.stderr)
    sys.exit(1)
    
def configure_port():
    parser = argparse.ArgumentParser(description="Real-time MPU6050 recorder")
    parser.add_argument("--port",   default=None,      help="Serial port (auto-detected if omitted)")
    args = parser.parse_args()

    port = args.port or find_port()

    buf_size = int(WINDOW_SEC * SAMPLE_HZ) # sample for WINDOW_SEC seconds with SAMPLE_HZ data/sec

    print(f"[INFO] Opening port {port} ")

    reader = SerialReader(port, BAUD_RATE, buf_size)
    reader.start()
    return reader
