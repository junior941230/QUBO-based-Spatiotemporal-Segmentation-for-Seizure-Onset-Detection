from pathlib import Path
from parser import parse_seizure_file

for p in sorted(Path("DESTINATION").glob("chb*/chb*-summary.txt")):
    try:
        parse_seizure_file(str(p))
        print(f"OK   {p.name}")
    except Exception as exc:
        print(f"FAIL {p.name}: {exc}")
