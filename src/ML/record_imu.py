"""
Capture imu data stream while program is running and record in a txt file.
"""

import serial
import serial.tools.list_ports
import argparse
import time

from src.ML.SerialReader import SerialReader
from src.ML.constants import WINDOW_SEC, SAMPLE_HZ, BAUD_RATE, DATA_PATH
from src.ML.plot_data import display_time_domain, display_spectrogram
from src.ML.utils import find_port

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

def record_wave_data(output_file, reader):
    print(f"[INFO] Starting data collection for {WINDOW_SEC}s ...")

    time.sleep(WINDOW_SEC)

    print(f"[INFO] Finished data collection. Copying to file {output_file}.")

    t, ax_, ay_, az_, gx_, gy_, gz_, tmp = reader.snapshot()

    with open(DATA_PATH + output_file + ".txt", 'w') as file:
        all_data = "\n".join([" ".join([str(num) for num in line]) for line in zip(ax_, ay_, az_, gx_, gy_, gz_)]) # double join

        bytes_written = file.write(all_data)
        print(f"[INFO] Number of bytes written to file: {bytes_written}")

def start_imu_data_collection(gesture_name):
    reader = configure_port()
    for iteration in range(20):
        print("Starting iteration", iteration, "in 2 seconds...")
        time.sleep(2)
        record_wave_data(f"{gesture_name}_{iteration:02}.txt", reader)

if __name__ == "__main__":
    # run python3 ./record_imu.py --port COM8
    # start_imu_data_collection("down")
    gesture = "right"
    file_num_1 = 10
    file_num_2 = 17
    file_1 = f"{gesture}_{file_num_1}.txt"
    file_2 = f"{gesture}_{file_num_2}.txt"
    sensor = "Acceleration"
    display_time_domain(file_1, sensor)
    display_time_domain(file_2, sensor)
    # display_spectrogram(file_1, sensor)
    # display_spectrogram(file_2, sensor)
