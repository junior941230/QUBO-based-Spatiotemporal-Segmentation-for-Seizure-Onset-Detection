import mne
if __name__ == "__main__":
    montage = mne.channels.make_standard_montage('standard_1020')
    print(montage)