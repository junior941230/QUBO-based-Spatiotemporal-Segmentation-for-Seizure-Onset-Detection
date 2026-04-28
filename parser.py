import re
FILE_PATH = "DESTINATION/chb12/chb12-summary.txt"


def parse_seizure_file(summary_path):
    """
    Parse CHB-MIT summary.txt.
    Returns: dict { filename: [(start_sec, end_sec), ...] }
    """
    seizure_times = {}
    current_file = None

    # 抓 "xxx seconds" 裡的數字,避免 split 爆掉
    num_pattern = re.compile(r"(\d+)\s*seconds?", re.IGNORECASE)

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as fp:
        lines = fp.readlines()

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("File Name"):
            # "File Name: chb24_01.edf"
            parts = line.split(":", 1)
            if len(parts) == 2:
                current_file = parts[1].strip()
                seizure_times.setdefault(current_file, [])
            continue

        # 兼容 "Seizure Start Time" 與 "Seizure N Start Time"
        if "Seizure" in line and "Start Time" in line:
            match = num_pattern.search(line)
            if not match or current_file is None:
                continue
            start_sec = int(match.group(1))
            seizure_times.setdefault(
                current_file, []).append([start_sec, None])
            continue

        if "Seizure" in line and "End Time" in line:
            match = num_pattern.search(line)
            if not match or current_file is None:
                continue
            end_sec = int(match.group(1))
            # 補到最後一筆還沒填 end 的 tuple
            bucket = seizure_times.get(current_file, [])
            for item in reversed(bucket):
                if item[1] is None:
                    item[1] = end_sec
                    break
            continue

    # 清理沒配對成功的 entry,轉 tuple
    cleaned = {}
    for fname, items in seizure_times.items():
        pairs = [(s, e) for s, e in items if s is not None and e is not None]
        cleaned[fname] = pairs
    return cleaned


if __name__ == "__main__":

    # 印出結果
    print(parse_seizure_file(FILE_PATH))
