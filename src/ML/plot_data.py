import matplotlib.pylab as plt
import numpy as np

from src.ML.constants import SAMPLE_HZ, DATA_PATH

AXES_OPTIONS = {
        "ax_": 0,
        "ay_": 1,
        "az_": 2,

        "gx_": 3,
        "gy_": 4,
        "gz_": 5
    }

def retrieve_wave_data(target_file):
    wave_data = []
    with open(DATA_PATH + target_file + ".txt") as file:
        for line in file:
            wave_data.append([float(num) for num in line.split(" ")])
    return wave_data

def display_time_domain(target_file, sensor):
    plt.figure()
    raw_data = retrieve_wave_data(target_file)
    if sensor == "Gyroscope":
        legend = ["gx_", "gy_", "gz_"]
        wave_data = [line[3:] for line in raw_data]
    else:
        legend = ["ax_", "ay_", "az_"]
        wave_data = [line[:3]for line in raw_data]
    plt.plot(list(range(0, len(wave_data))), wave_data)
    plt.title(f"{target_file} {sensor} data")
    plt.xlabel("time (ms)")
    plt.ylabel("Amplitude")
    plt.legend(legend)
    plt.show()


def display_spectrogram(target_file, sensor, nfft=None):
    """
    Borrowed from lecture example
    MATLAB-like FFT magnitude plot (centered, both ± frequencies)
    """

    raw_data = retrieve_wave_data(target_file)
    plt.figure()

    if sensor == "Acceleration":
        wave_data = [line[:3] for line in raw_data]
        legend = ["ax_", "ay_", "az_"]
    else:
        wave_data = [line[3:] for line in raw_data]
        legend = ["gx_", "gy_", "gz_"]
    
    for index in range(3):

        x = np.asarray([line[index] for line in wave_data])

        if nfft is None:
            nfft = len(x)

        X = np.fft.fft(x, n=nfft)
        f = np.fft.fftfreq(nfft, d=1 / SAMPLE_HZ)

        # Center spectrum (like fftshift)
        Xs = np.fft.fftshift(X)
        fs_shift = np.fft.fftshift(f)

        mag = np.abs(Xs)

        plt.plot(fs_shift, mag)
        
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|X(f)|")
    plt.title(f"{target_file} {sensor} FFT magnitude")
    plt.grid(True)
    plt.legend(legend)
    plt.show()