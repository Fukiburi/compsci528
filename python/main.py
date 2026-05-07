# Example Drone Automatic Maneuvering

# Place at top of gesture recognition file
######################################################################

import threading 
import socket
import sys
import time
import platform

import joblib

from python.ML.SVM import SVM
from python.constants import ROOT
from python.live_detection import detect
from python.port_config import configure_port  

host = ''
port = 9000
locaddr = (host,port) 

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

tello_address = ('192.168.10.1', 8889)

sock.bind(locaddr)

def recv():
    count = 0
    while True: 
        try:
            data, server = sock.recvfrom(1518)
            print(data.decode(encoding="utf-8"))
        except Exception:
            print ('\nExit . . .\n')
            break

#recvThread create
recvThread = threading.Thread(target=recv)
recvThread.start()

def act(msg):
    """
    Function for sending commands to Tello drone. Use Tello SDK for command options.
    """
    print("sending " + msg)
    msg = msg.encode(encoding="utf-8")
    sent = sock.sendto(msg, tello_address)
    time.sleep(5)

######################################################################

# After prediction: Use assigned enumeration
######################################################################
def demo():
    act("command")
    act("takeoff")
    act("cw 360")
    act("land")


######################################################################
# Begin gesture prediction
######################################################################
classes = ["left", "right", "up", "down"]
samples_per_class = 20
reader = configure_port()

# get model, either pre-trained or new
try:
    svm_model = joblib.load("svm_model.pkl")
except:
    data_path = ROOT / "test_gesture_data"
    svm_model = SVM(classes, data_path, feature_extraction_method='METRICS', n_samples_per_class=samples_per_class)
    svm_model.fit()
    threshold = 0.95
    svm_model.evaluate(accuracy_threshold=threshold, save_pipeline=True)

detector = detect(reader, classes, svm_model)

for predicted_gesture in detector:
    act(classes[predicted_gesture[0]])

sock.close()

######################################################################
