"""
Useful methods that have no other place
"""
import re

# Regex that matches both raw ESP_LOGI lines and plain printed lines
LINE_RE = re.compile(
    r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)"
    r"\s*\|\s*"
    r"GX:(?P<gx>[-\d.]+)\s+GY:(?P<gy>[-\d.]+)\s+GZ:(?P<gz>[-\d.]+)"
    r"\s*\|\s*"
    r"T:(?P<t>[-\d.]+)"
)

def parse_line(line: str):
    """Return (ax, ay, az, gx, gy, gz, temp) floats or None."""
    m = LINE_RE.search(line)
    if m:
        return tuple(float(m.group(k)) for k in ("ax", "ay", "az", "gx", "gy", "gz", "t"))
    return None

def read_data_from_file(file_path):
    wave_data = []
    with open(file_path) as file:
        for line in file:
            # TODO: Comment out this line when reading files with raw esp32 outputs
            parsed = parse_line(line)
            if parsed == None:
                continue
            wave_data.append(parsed)
            # wave_data.append([float(num) for num in line.split(" ")])
    return wave_data