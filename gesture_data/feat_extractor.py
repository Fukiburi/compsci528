import numpy as np
from enum import Enum

class Gesture(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    FORWARD = "forward"
    BACK = "back"
    RAISE_HAND = "raise_hand"
    CCW = "ccw"

def calculate_sma(imu_data, offsets=None):
    """
    imu_data: (n_samples, 6) array for [ax, ay, az, gx, gy, gz]
    offsets: optional static offsets (e.g., gravity for z-axis)
    """
    if offsets is None:
        offsets = np.mean(imu_data, axis=0)  # Estimate bias
    corrected = np.abs(imu_data - offsets)
    sma = np.mean(np.sum(corrected, axis=1))
    return sma

def calculate_rms(imu_data):
    return np.sqrt(np.mean(np.square(imu_data), axis=0))

def extract_feat(filename):
    """
    Function to return feature vector (7 features). Input: txt file of 6 axes floats, comma separated
    """

    data = np.loadtxt(filename, delimiter=",")

    mean_vec = np.mean(data, axis=0)
    min_vec = np.min(data, axis=0)
    max_vec = np.max(data, axis=0)
    range_vec = np.ptp(data, axis=0)
    std_dev_vec = np.ptp(data, axis=0)
    sma = calculate_sma(data, offsets=None)
    rms_vec = calculate_rms(data)

    feature_vector = np.concatenate([
    mean_vec, min_vec, max_vec, range_vec, std_dev_vec, rms_vec  # all shape (6,)
    ])

    feature_vector = np.append(feature_vector, sma)

    return feature_vector

def assemble_csv(floats):
    result = ", ".join(str(x) for x in floats)
    with open("sample_feat_vec.txt", "a") as file:
        file.write(result + "\n")

# go through sample files

for gesture in Gesture:
    for i in range(21):
        filename = f"{gesture.value}_{i:02d}.txt"
        floats = extract_feat(filename)
        assemble_csv(floats)

for gesture in Gesture:
    for i in range(21):
        filename = f"sample_class.txt"
        with open(filename, "a") as file:
            if f"{gesture.value}" == "up":
                file.write(str(0) + "\n")
            elif f"{gesture.value}" == "down":
                file.write(str(1) + "\n")
            elif f"{gesture.value}" == "left":
                file.write(str(2) + "\n")
            elif f"{gesture.value}" == "right":
                file.write(str(3) + "\n")
            elif f"{gesture.value}" == "forward":
                file.write(str(4) + "\n")
            elif f"{gesture.value}" == "back":
                file.write(str(5) + "\n")
            elif f"{gesture.value}" == "raise_hand":
                file.write(str(6) + "\n")
            elif f"{gesture.value}" == "ccw":
                file.write(str(7) + "\n")
