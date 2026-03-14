from pipeline import preprocess_one_file, solve_qubo_seizure

if __name__ == "__main__":
    raw = preprocess_one_file('DESTINATION/chb01/chb01_03.edf', [(0, 10), (20, 30)])