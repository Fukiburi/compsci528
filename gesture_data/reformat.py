import re
from pathlib import Path

def trim_and_fix(directory="./gesture_data"):
    for f in Path(directory).glob("*.txt"):
        lines = f.read_text().splitlines()
        rows = [re.split(r",\s*", line.strip()) for line in lines if line.strip()]
        rows = rows[-400:]
        output = "\n".join(" ".join(row) for row in rows)
        f.write_text(output)
        print(f"Trimmed {f.name} to {len(rows)} rows")

trim_and_fix()