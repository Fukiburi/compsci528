from enum import Enum, auto
import joblib
import time
import numpy as np

from SerialReader import SerialReader
from python.constants import WINDOW_SEC, SAMPLE_HZ
from python.port_config import configure_port

ACC_THRESHOLD = 1.1
GYRO_THRESHOLD = 50
WINDOW_SIZE = WINDOW_SEC * SAMPLE_HZ

CLASSES = ["left", "right", "up", "down"]

class ReaderState(Enum):
    WAITING = auto()
    COLLECTING = auto()

def detect(reader: SerialReader):
    state = ReaderState.WAITING
    svm_model = joblib.load("svm_model_pipeline.pkl")

    print("Please wait 2 seconds before starting.") # this guarantees we have at least a 4 second window to grab data from
    time.sleep(2)
    print("Starting live gesture detection...")

    while True:
        time.sleep(1 / SAMPLE_HZ) # match up loop to sample hz (doesn't have to be exact)
        gyro_mag = reader.mean_magnitude_gyroscope
        acc_mag = reader.mean_magnitude_accelerometer
        # print("Gyro magnitude: ", gyro_mag) #debug
        # print("Acc magnitude: ", acc_mag) #debug
        
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
                print("Prediction: ", predicted_class, CLASSES[predicted_class[0]])

                time.sleep(1) # just for prediction readability in stdout
                print("Please enter next gesture.")
                # display_time_domain("accelerometer", data = raw) #debug

reader = configure_port()
detect(reader)