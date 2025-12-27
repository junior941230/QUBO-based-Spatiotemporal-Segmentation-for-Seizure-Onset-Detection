
FILE_PATH = "DESTINATION\chb01\chb01-summary.txt"


def parse_seizure_file(file_path):
    """
    解析發作時間檔案(txt)，回傳一個字典
    """
    seizure_mapping = {}
    current_file = None
    with open(file_path, "r") as file:
        content = file.read()
        for line in content.splitlines():
            if line.startswith("File Name:"):
                current_file = line.split(": ")[1]
                seizure_mapping[current_file] = []
            elif "Seizure" in line and "Start Time" in line:
                # 把字串切開，取出數字部分
                parts = line.split(": ")
                seconds = int(parts[1].split(" ")[0])  # 拿掉 ' seconds'

                # 暫存開始時間，因為要等讀到 End Time 才能湊成一對
                start_time = seconds

            # 3. 抓取發作結束時間
            elif "Seizure" in line and "End Time" in line:
                parts = line.split(": ")
                seconds = int(parts[1].split(" ")[0])
                end_time = seconds

                # 將 (開始, 結束) 存入當前檔案的紀錄中
                seizure_mapping[current_file].append((start_time, end_time))
    return seizure_mapping


if __name__ == "__main__":

    # 印出結果
    print(parse_seizure_file(FILE_PATH))
