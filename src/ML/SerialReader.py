import threading
from collections import deque
import time
import serial
from src.ML.utils import parse_line



class SerialReader(threading.Thread):
    """Background thread that fills shared deques from the serial port."""

    def __init__(self, port: str, baud: int, buf_size: int):
        super().__init__(daemon=True)
        self.port     = port
        self.baud     = baud
        self.buf_size = buf_size
        self.lock     = threading.Lock()

        self.t    = deque(maxlen=buf_size)
        self.ax   = deque(maxlen=buf_size)
        self.ay   = deque(maxlen=buf_size)
        self.az   = deque(maxlen=buf_size)
        self.gx   = deque(maxlen=buf_size)
        self.gy   = deque(maxlen=buf_size)
        self.gz   = deque(maxlen=buf_size)
        self.temp = deque(maxlen=buf_size)

        self.connected = False
        self.status    = "Connecting…"

    def run(self):
        while True:
            try:
                with serial.Serial(self.port, self.baud, timeout=1) as ser:
                    self.connected = True
                    self.status    = f"Connected  {self.port}  @{self.baud} baud"
                    t0 = time.perf_counter()
                    while True:
                        raw = ser.readline()
                        try:
                            line = raw.decode("utf-8", errors="replace").strip()
                        except Exception:
                            continue
                        parsed = parse_line(line)
                        if parsed is None:
                            continue
                        ax, ay, az, gx, gy, gz, temp = parsed
                        now = time.perf_counter() - t0
                        with self.lock:
                            self.t.append(now)
                            self.ax.append(ax);   self.ay.append(ay);   self.az.append(az)
                            self.gx.append(gx);   self.gy.append(gy);   self.gz.append(gz)
                            self.temp.append(temp)
            except serial.SerialException as e:
                self.connected = False
                self.status    = f"Disconnected — {e}  (retrying…)"
                time.sleep(2)

    def snapshot(self):
        """Thread-safe copy of all buffers."""
        with self.lock:
            return (
                list(self.t),
                list(self.ax), list(self.ay), list(self.az),
                list(self.gx), list(self.gy), list(self.gz),
                list(self.temp),
            )