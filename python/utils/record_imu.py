"""
Capture imu data stream while record in a .txt file.
Intended to capture training data for testing the ML model
"""
import time
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.constants import WINDOW_SEC, ROOT
from python.utils.plot_data import display_time_domain
from python.port_config import configure_port

def record_wave_data(output_file, reader):
    print(f"[INFO] Starting data collection for {WINDOW_SEC}s ...")

    time.sleep(WINDOW_SEC)

    print(f"[INFO] Finished data collection. Copying to file {output_file}.")

    t, ax_, ay_, az_, gx_, gy_, gz_, tmp = reader.snapshot()

    with open(ROOT / "gesture_data_lrud2/" / output_file, 'w') as file:
        all_data = "\n".join([" ".join([str(num) for num in line]) for line in zip(ax_, ay_, az_, gx_, gy_, gz_)]) # double join

        bytes_written = file.write(all_data)
        print(f"[INFO] Number of bytes written to file: {bytes_written}")

def start_imu_data_collection(gesture_name):
    reader = configure_port()
    for iteration in range(20, 40):
        print("Starting iteration", iteration, "in 2 seconds...")
        time.sleep(2)
        record_wave_data(f"{gesture_name}_{iteration:02}.txt", reader)

if __name__ == "__main__":
    # run python3 ./record_imu.py --port COM8
    classes = ["back", "up", "down", "ccw", "raise_hand"]
    for label in classes:
        print(f"NEW: Starting collection for gesture {label}")
        start_imu_data_collection(label)
    # gesture = "right"
    # file_num_1 = 10
    # file_num_2 = 17
    # file_1 = f"{gesture}_{file_num_1}.txt"
    # file_2 = f"{gesture}_{file_num_2}.txt"
    # sensor = "Acceleration"
    # display_time_domain(file_1, sensor)
    # display_time_domain(file_2, sensor)
    # display_spectrogram(file_1, sensor)
    # display_spectrogram(file_2, sensor)
