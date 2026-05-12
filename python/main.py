# Example Drone Automatic Maneuvering

# Place at top of gesture recognition file
######################################################################

from pathlib import Path
import threading 
import socket
import sys
import time
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

def act_return_to_origin(x, y, z):
    # TODO: collision detection + path correction?
    if x > 0:
        act(f"right {x}")
    elif x < 0:
        act(f"left {x}")
    
    if y > 0:
        act(f"forward {y}")
    elif y < 0:
        act(f"back {y}")

    return 0, 0

def act_sweep(sweep_range):
    #sweep in 1 direction, move up, sweep in other direction, move up
    sweeps = 2
    left = True
    for _ in range(sweeps):
        if left:
            act(f"left {sweep_range}")
            left = False
        else:
            act(f"right {sweep_range}")
            left = True
        act('forward 60')

######################################################################
# Begin gesture prediction
######################################################################
classes = ["left", "right", "forward", "back", "up", "down", "ccw", "raise_hand"]
samples_per_class = 20
reader = configure_port()

data_path = ROOT / "gesture_data"
svm_model = SVM(
    classes, data_path,
    feature_extraction_method='METRICS',
    n_samples_per_class=samples_per_class,
    source_file="svm_model.pkl"
)

detector = detect(reader, classes, svm_model)
distance_cm = 80
x_traveled = 0
y_traveled = 0
z_traveled = 0
sweep_range_cm = 120

act("command")
act("takeoff")

for prediction in detector:
    predicted_gesture = classes[prediction[0]]
    print("Detected gesture for command: ", predicted_gesture)
    match predicted_gesture:
        case "down":
            act("land")
            break
        case "left":
            x_traveled -= distance_cm
        case "right":
            x_traveled += distance_cm
        case "forward":
            y_traveled += distance_cm
        case "back":
            y_traveled -= distance_cm
        case "up":
            act("up 40")
            continue
        case "raise_hand":
            x_traveled, y_traveled = act_return_to_origin(x_traveled, y_traveled, z_traveled)
        case "ccw":
            act_sweep(sweep_range_cm)
    act(f"{predicted_gesture} {distance_cm}")

act("land")

sock.close()

######################################################################
