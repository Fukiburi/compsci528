from pathlib import Path

# IMU configurations
SAMPLE_HZ  = 100      # assumed sampling rate of sensor
BAUD_RATE  = 115200   # data/sec
WINDOW_SEC = 4       # num of seconds to collect at once

# Paths
ROOT = Path(__file__).resolve().parents[1]