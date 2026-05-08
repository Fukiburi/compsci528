COMPSCI 528: Mobile and Ubiquitous Computing
University of Massachusetts Amherst
Homework #2 (100 pt)
Steven Tang

How to Initialize:

Physical set up:
ESP32-S3-DevKitC-1 pins -> ITG/MPU
pin-to-pin pairs:
G, GND
3V3, VCC
1, SCL
0, SDA

Connect to computer (Linux)

in terminal where running ESP-IDF will be running, run:

. $HOME/esp/esp-idf/export.sh

cd ~/Downloads/compsci528_tang_a1
idf.py set-target esp32-s3
idf.py menuconfig

idf.py build

sudo chmod a+rw /dev/ttyUSB0
idf.py -p /dev/ttyUSB0 flash

python -m venv venv
source venv/bin/activate
python3 compsci528_tang_a1/assignment1.py

##################################################

If initialization is setup, how to train:

cd gesture_data/
source XXX/bin/activate (whatever virtual environment you have)
rm sample_class.txt sample_feat_vec.txt
python3 feat_extractor.py

This will produce 2 files:
sample_class.txt
sample_feat_vec.txt

cp sample_*.txt ..

##################################################

How to run:

source XXX/bin/activate (whatever virtual environment you have)
python3 demo.py
