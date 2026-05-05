import serial
import time
import os
import sys

PORT = "COM4"
BAUD = 115200
RECORD_SECONDS = 4.0
OUTPUT_DIR = "gesture_data"

GESTURES = [
    "up",
    "down",
    "left",
    "right",
    "forward",
    "backward",
    "palm_up_wrist",
    "palm_down_wrist",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def record_to_file(ser, filename, duration):
    end_time = time.time() + duration
    with open(filename, "w", newline="") as f:
        f.write("# timestamp_sec,line\n")
        try:
            while time.time() < end_time:
                line = ser.readline().decode(errors="replace").strip()
                if not line:
                    continue
                ts = time.time()
                f.write(f"{ts:.6f},{line}\n")
        except KeyboardInterrupt:
            raise

def prompt_start(prompt_text):
    """
    Returns:
      - 'start' to begin capture (Enter)
      - 'cancel' to skip current capture ('c' or 'C')
      - 'skip_gesture' to skip entire gesture ('s' or 'S')
    Raises KeyboardInterrupt on Ctrl+C.
    """
    try:
        resp = input(prompt_text + " (Enter=start, 'c'=skip capture, 's'=skip gesture): ").strip()
    except KeyboardInterrupt:
        raise
    if resp.lower() == "s":
        return "skip_gesture"
    if resp.lower() == "c":
        return "cancel"
    return "start"

def main():
    print(f"Opening serial {PORT} @ {BAUD}...")
    try:
        with serial.Serial(PORT, BAUD, timeout=0.1) as ser:
            time.sleep(1.0)
            ser.reset_input_buffer()

            for gesture in GESTURES:
                skip_rest_of_gesture = False

                # initial capture saved as gesture_00.txt
                filename_00 = os.path.join(OUTPUT_DIR, f"{gesture}_00.txt")
                action = prompt_start(f"\nPrepare to perform '{gesture}' gesture (initial sample).")
                if action == "skip_gesture":
                    print(f"Skipped entire gesture '{gesture}'")
                    continue
                if action == "cancel":
                    print(f"Skipped {filename_00}")
                else:
                    print("Get ready...")
                    time.sleep(0.5)
                    print("Recording...")
                    try:
                        record_to_file(ser, filename_00, RECORD_SECONDS)
                        print(f"Saved {filename_00}")
                    except KeyboardInterrupt:
                        print("\nInterrupted. Exiting.")
                        return

                # repeat 20 times -> gesture_01 .. gesture_20
                for i in range(1, 21):
                    fname = os.path.join(OUTPUT_DIR, f"{gesture}_{i:02d}.txt")
                    action = prompt_start(f"Prepare to perform '{gesture}' gesture #{i:02d}.")
                    if action == "skip_gesture":
                        print(f"Skipped remaining captures for gesture '{gesture}'")
                        skip_rest_of_gesture = True
                        break
                    if action == "cancel":
                        print(f"Skipped {fname}")
                        continue
                    print("Get ready...")
                    time.sleep(0.5)
                    print("Recording...")
                    try:
                        record_to_file(ser, fname, RECORD_SECONDS)
                        print(f"Saved {fname}")
                    except KeyboardInterrupt:
                        print("\nInterrupted. Exiting.")
                        return

                if skip_rest_of_gesture:
                    continue

    except serial.SerialException as e:
        print(f"Serial error: {e}")
        sys.exit(1)

    print("\nAll captures complete.")

if __name__ == "__main__":
    main()
