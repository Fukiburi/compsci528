from enum import Enum, auto
from pathlib import Path
import sys
import time
import numpy as np
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from python.ML.SVM import SVM
from python.SerialReader import SerialReader
from python.constants import ROOT, WINDOW_SEC, SAMPLE_HZ
from python.port_config import configure_port

ACC_THRESHOLD = 1.1
GYRO_THRESHOLD = 50
WINDOW_SIZE = WINDOW_SEC * SAMPLE_HZ

class ReaderState(Enum):
    WAITING = auto()
    COLLECTING = auto()

def detect(reader: SerialReader, classes, svm_model):
    state = ReaderState.WAITING

    print("Please wait 2 seconds before starting.") # this guarantees we have at least a 4 second window to grab data from
    time.sleep(2)
    print("Starting live gesture detection...")

    while True:
        time.sleep(1 / SAMPLE_HZ) # match up loop to sample hz (doesn't have to be exact)
        gyro_mag = reader.mean_magnitude_gyroscope
        acc_mag = reader.mean_magnitude_accelerometer
        print("Gyro magnitude: ", gyro_mag) #debug
        print("Acc magnitude: ", acc_mag) #debug
        
        if state == ReaderState.WAITING:
            if gyro_mag > GYRO_THRESHOLD and acc_mag > ACC_THRESHOLD:
                # print("Changing state to COLLECTING.") #debug
                state = ReaderState.COLLECTING

        if state == ReaderState.COLLECTING:
            if gyro_mag < GYRO_THRESHOLD and acc_mag < ACC_THRESHOLD:
                # print("Changing state to WAITING") #debug
                print("Detected gesture!")
                state = ReaderState.WAITING
                time.sleep(2)

                # === signal preprocessing ===
                t, ax_, ay_, az_, gx_, gy_, gz_, tmp = reader.snapshot()
                raw = np.array(list(zip(ax_, ay_, az_, gx_, gy_, gz_)))[-WINDOW_SIZE:]
                sample_time_series = raw[np.newaxis, :, :]

                # === prediction using SVM ===
                predicted_class = svm_model.predict(sample_time_series)
                print("Prediction: ", predicted_class, classes[predicted_class[0]])
                yield predicted_class
                time.sleep(1) # just for prediction readability in stdout
                print("Please enter next gesture.")
                # display_time_domain("accelerometer", data = raw) #debug

if __name__ == "__main__":
    # python -m python.live_detection --port COM8
    classes = ["left", "right", "forward", "land"]
    samples_per_class = 20
    reader = configure_port()

    data_path = ROOT / "gesture_data"
    svm_model = SVM(
        classes, data_path,
        feature_extraction_method='METRICS',
        n_samples_per_class=samples_per_class,
        source_file="svm_model.pkl"
    )

    for predicted_class in detect(reader, classes, svm_model):
        print(predicted_class)