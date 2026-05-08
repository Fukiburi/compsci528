#!/usr/bin/env python3
"""
Real-time IMU plotter for MPU6050 streamed from an ESP32.

Expected serial line format (ESP_LOGI):
  I (ticks) mpu6050 stream: AX:0.123 AY:-0.456 AZ:9.789 | GX:1.23 GY:-0.45 GZ:0.67 | T:25.30 C

Usage:
  python plot_imu.py                        # auto-detects port
  python plot_imu.py --port /dev/tty.usbserial-11130
  python plot_imu.py --port /dev/tty.usbserial-11130 --baud 115200 --window 5
"""

import argparse
import re
import sys
import threading
import time
from collections import deque
import numpy as np

import matplotlib
# matplotlib.use("TkAgg")          # change to "Qt5Agg" if TkAgg is unavailable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import serial
import serial.tools.list_ports

# for machine learning
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ── Configuration ────────────────────────────────────────────────────────────
BAUD_RATE    = 115200
WINDOW_SEC   = 4 #5      # seconds of history to display
SAMPLE_HZ    = 400    # expected sample rate (used only for buffer sizing)
buffer = deque(maxlen=400)  # Fixed-size buffer for sliding window

# Regex that matches both raw ESP_LOGI lines and plain printed lines
LINE_RE = re.compile(
    r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)"
    r"\s*\|\s*"
    r"GX:(?P<gx>[-\d.]+)\s+GY:(?P<gy>[-\d.]+)\s+GZ:(?P<gz>[-\d.]+)"
    r"\s*\|\s*"
    r"T:(?P<t>[-\d.]+)"
)

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "bg":     "#0f1117",
    "panel":  "#1a1d27",
    "grid":   "#2a2d3a",
    "ax":     "#4fc3f7",   # blue  – accel X
    "ay":     "#81d4fa",   # light blue  – accel Y
    "az":     "#b3e5fc",   # pale blue   – accel Z
    "gx":     "#f48fb1",   # pink  – gyro X
    "gy":     "#f06292",   # rose  – gyro Y
    "gz":     "#e91e63",   # deep rose   – gyro Z
    "temp":   "#ffcc02",   # gold  – temperature
    "text":   "#e0e0e0",
    "title":  "#ffffff",
}

# set up machine learning model
sample_feat_vec = np.loadtxt("sample_feat_vec.txt", delimiter=",")
sample_class = np.loadtxt("sample_class.txt", delimiter=",")

# Split and scale data
scaler = StandardScaler()
sample_feat_vec_scaled = scaler.fit_transform(sample_feat_vec)

# Train SVC with RBF kernel
clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(sample_feat_vec_scaled, sample_class)

def find_port() -> str:
    """Auto-detect the first USB-serial port."""
    ports = serial.tools.list_ports.comports()
    usb = [p for p in ports if "usb" in p.device.lower() or "usbserial" in p.device.lower()]
    if usb:
        return usb[0].device
    if ports:
        return ports[0].device
    print("[ERROR] No serial ports found. Plug in your ESP32 or specify --port.", file=sys.stderr)
    sys.exit(1)


def parse_line(line: str):
    """Return (ax, ay, az, gx, gy, gz, temp) floats or None."""
    m = LINE_RE.search(line)
    if m:
        return tuple(float(m.group(k)) for k in ("ax", "ay", "az", "gx", "gy", "gz", "t"))
    return None


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
                        ax, ay, az, gx, gy, gz, temp = parsed # get readings
                        now = time.perf_counter() - t0
                        
                        ###############################################
                        
                        # After getting the readings, we can filter into attributes
                        
                        floats = [ax, ay, az, gx, gy, gz]
                        
                        buffer.append(floats) # add to deque
                        if len(buffer) == SAMPLE_HZ:  # Full window
                            # wait for deque to fill before processing
                            win = np.array(buffer)
                            
                            # extract features to make a feature vector of each window
                            mean_vec = np.mean(win, axis=0)
                            min_vec = np.min(win, axis=0)
                            max_vec = np.max(win, axis=0)
                            range_vec = np.ptp(win, axis=0)
                            std_dev_vec = np.ptp(win, axis=0)
                            sma = calculate_sma(win, offsets=None)
                            rms_vec = calculate_rms(win)
                            
                            feature_vector = np.concatenate([
                            mean_vec, min_vec, max_vec, range_vec, std_dev_vec, rms_vec  # all shape (6,)
                            ])
                            
                            feature_vector = np.append(feature_vector, sma)
                            
                            # reshape
                            feature_vector = feature_vector.reshape(1, -1)
                            
                            feature_vector_scaled = scaler.transform(feature_vector)
                            gesture_pred = clf.predict(feature_vector_scaled)
                            if gesture_pred == 0:
                                print(f"Prediction:{gesture_pred}, up")
                            if gesture_pred == 1:
                                print(f"Prediction:{gesture_pred}, down")
                            if gesture_pred == 2:
                                print(f"Prediction:{gesture_pred}, left")
                            if gesture_pred == 3:
                                print(f"Prediction:{gesture_pred}, right")
                            if gesture_pred == 4:
                                print(f"Prediction:{gesture_pred}, forward")
                            if gesture_pred == 5:
                                print(f"Prediction:{gesture_pred}, back")
                            if gesture_pred == 6:
                                print(f"Prediction:{gesture_pred}, raise_hand")
                            if gesture_pred == 7:
                                print(f"Prediction:{gesture_pred}, ccw")
                        
                        ################################################
                        
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
        #print(self.t, self.ax, self.ay, self.az, self.gx, self.gy, self.gz, self.temp)
        with self.lock:
            return (
                list(self.t),
                list(self.ax), list(self.ay), list(self.az),
                list(self.gx), list(self.gy), list(self.gz),
                list(self.temp),
            )


def style_axes(ax, ylabel, ylim=None):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["text"], labelsize=8)
    ax.yaxis.label.set_color(C["text"])
    ax.xaxis.label.set_color(C["text"])
    ax.set_ylabel(ylabel, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(C["grid"])
    ax.grid(True, color=C["grid"], linewidth=0.5, linestyle="--")
    if ylim:
        ax.set_ylim(*ylim)


def main():
    parser = argparse.ArgumentParser(description="Real-time MPU6050 plotter")
    parser.add_argument("--port",   default=None,      help="Serial port (auto-detected if omitted)")
    parser.add_argument("--baud",   default=BAUD_RATE,  type=int, help=f"Baud rate (default {BAUD_RATE})")
    parser.add_argument("--window", default=WINDOW_SEC, type=float, help=f"Plot window in seconds (default {WINDOW_SEC})")
    args = parser.parse_args()

    port     = args.port or find_port()
    baud     = args.baud
    window   = args.window
    buf_size = int(window * SAMPLE_HZ * 2)   # 2× safety margin

    print(f"[INFO] Opening  {port}  at {baud} baud …")
    reader = SerialReader(port, baud, buf_size)
    reader.start()

    # ── Figure layout ─────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13, 8), facecolor=C["bg"])
    fig.canvas.manager.set_window_title("MPU6050 — Real-Time Stream")

    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45,
                           left=0.08, right=0.97, top=0.90, bottom=0.08)

    ax_acc  = fig.add_subplot(gs[0])
    ax_gyro = fig.add_subplot(gs[1])
    ax_temp = fig.add_subplot(gs[2])

    style_axes(ax_acc,  "Acceleration (g)")
    style_axes(ax_gyro, "Angular rate (°/s)")
    style_axes(ax_temp, "Temperature (°C)")

    # Title + status bar
    suptitle = fig.suptitle("MPU6050  Real-Time Stream  — 100 Hz",
                             color=C["title"], fontsize=13, fontweight="bold")
    status_txt = fig.text(0.5, 0.005, reader.status,
                          ha="center", fontsize=8, color="#888888")

    # Legend handles (empty lines animated later)
    la_x, = ax_acc.plot([], [], color=C["ax"], lw=1.2, label="Accel X")
    la_y, = ax_acc.plot([], [], color=C["ay"], lw=1.2, label="Accel Y")
    la_z, = ax_acc.plot([], [], color=C["az"], lw=1.2, label="Accel Z")

    lg_x, = ax_gyro.plot([], [], color=C["gx"], lw=1.2, label="Gyro X")
    lg_y, = ax_gyro.plot([], [], color=C["gy"], lw=1.2, label="Gyro Y")
    lg_z, = ax_gyro.plot([], [], color=C["gz"], lw=1.2, label="Gyro Z")

    lt,   = ax_temp.plot([], [], color=C["temp"], lw=1.5, label="Temperature")

    for a, lines in [(ax_acc,  [la_x, la_y, la_z]),
                     (ax_gyro, [lg_x, lg_y, lg_z]),
                     (ax_temp, [lt])]:
        leg = a.legend(handles=lines, loc="upper left", fontsize=7,
                       facecolor=C["panel"], edgecolor=C["grid"],
                       labelcolor=C["text"])

    ax_temp.set_xlabel("Time (s)", fontsize=9)

    # Live numeric readouts
    def readout(ax, x, y, text=""):
        return ax.text(1.001, y, text, transform=ax.transAxes,
                       color=C["text"], fontsize=7.5, va="center",
                       fontfamily="monospace")

    ro_ax = readout(ax_acc,  1, 0.83)
    ro_ay = readout(ax_acc,  1, 0.50)
    ro_az = readout(ax_acc,  1, 0.17)
    ro_gx = readout(ax_gyro, 1, 0.83)
    ro_gy = readout(ax_gyro, 1, 0.50)
    ro_gz = readout(ax_gyro, 1, 0.17)
    ro_t  = readout(ax_temp, 1, 0.50)

    def update(_):
        t, ax_, ay_, az_, gx_, gy_, gz_, tmp = reader.snapshot()

        if len(t) < 2:
            return

        t_now = t[-1]
        t_lo  = t_now - window

        # Slice to the visible window
        def trim(xs, ts):
            return [x for x, ti in zip(xs, ts) if ti >= t_lo]

        tv  = [ti for ti in t  if ti >= t_lo]
        axv = trim(ax_, t); ayv = trim(ay_, t); azv = trim(az_, t)
        gxv = trim(gx_, t); gyv = trim(gy_, t); gzv = trim(gz_, t)
        tv2 = tv  # shared time vector

        la_x.set_data(tv2, axv); la_y.set_data(tv2, ayv); la_z.set_data(tv2, azv)
        lg_x.set_data(tv2, gxv); lg_y.set_data(tv2, gyv); lg_z.set_data(tv2, gzv)

        tv3 = [ti for ti in t if ti >= t_lo]
        tmpv = trim(tmp, t)
        lt.set_data(tv3, tmpv)

        # X-axis range  (fixed [t_lo, t_now] window)
        for a in (ax_acc, ax_gyro, ax_temp):
            a.set_xlim(t_lo, t_now)

        # Y-axis auto-scale with a little padding
        def auto_ylim(ax, *series):
            vals = [v for s in series for v in s]
            if vals:
                lo, hi = min(vals), max(vals)
                pad = max((hi - lo) * 0.15, 0.05)
                ax.set_ylim(lo - pad, hi + pad)

        auto_ylim(ax_acc,  axv, ayv, azv)
        auto_ylim(ax_gyro, gxv, gyv, gzv)
        auto_ylim(ax_temp, tmpv)

        # Live readouts
        if ax_:
            ro_ax.set_text(f"AX {ax_[-1]:+7.3f}")
            ro_ay.set_text(f"AY {ay_[-1]:+7.3f}")
            ro_az.set_text(f"AZ {az_[-1]:+7.3f}")
            ro_gx.set_text(f"GX {gx_[-1]:+7.2f}")
            ro_gy.set_text(f"GY {gy_[-1]:+7.2f}")
            ro_gz.set_text(f"GZ {gz_[-1]:+7.2f}")
            ro_t.set_text(f"{tmp[-1]:.2f} °C")

        status_txt.set_text(reader.status)

    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

    plt.show()


if __name__ == "__main__":
    main()
