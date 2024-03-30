import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

import utils


data_ekg1 = []
data_ekg_100 = []
data_ekg_noise = []

ekg1_time = None
ekg_100_time = None
ekg_noise_time = None

sampling_frequency_ekg1 = 1000
sampling_frequency_ekg_100 = 360
sampling_frequency_ekg_noise = 360


def main():
    exercise_2_5()
    return

    if len(sys.argv) < 2:
        print("Usage: python main.py <exercise_number>")
        return
    if (sys.argv[1] == "1"):
        files_to_choose = utils.list_files_in_directory("./resources")
        print("Zadanie 1:")
        print(*files_to_choose, sep=", ")
        print("Wybierz plik wpisując liczbę (1, 2 lub 3):")

        try:
            chosen_file = int(input())
            match chosen_file:
                case 1:
                    exercise_1_1()
                case 2:
                    exercise_1_2()
                case 3:
                    exercise_1_3()
        except Exception:
            print("Niepoprawny wybór")
            return


def exercise_1_1():
    global data_ekg1
    global ekg1_time
    global sampling_frequency_ekg1

    with open("./resources/ekg1.txt", "r+") as file_ekg_ex1:
        while True:
            line = file_ekg_ex1.readline()
            if not line:
                break
            values_row = line.lstrip().split()
            values_row = np.array(list(map(int, values_row)))

            data_ekg1.append(values_row)

    data_ekg1 = np.array(data_ekg1)
    ekg1_time = np.arange(len(data_ekg1)) / sampling_frequency_ekg1

    plt.figure(figsize=(20, 10))
    plt.legend(loc="upper left")

    for i in range(len(data_ekg1[0])):
        plt.plot(ekg1_time, data_ekg1[:, i])

    plt.legend(["Column " + str(i + 1) for i in range(len(data_ekg1[0]))])
    plt.title('ekg1.txt')
    plt.xlabel('Czas[s]')
    plt.ylabel('Wartość')
    plt.xlim(ekg1_time[0], ekg1_time[len(ekg1_time) - 1])

    plt.show()


def exercise_1_2():
    global data_ekg_100
    global ekg_100_time
    global sampling_frequency_ekg_100

    with open("./resources/ekg_100.txt", "r+") as file_ekg_100:
        while True:
            line = file_ekg_100.readline()
            if not line:
                break
            values_row = line.lstrip().split()
            values_row = np.array(list(map(float, values_row)))

            data_ekg_100.append(values_row)

    data_ekg_100 = np.array(data_ekg_100)
    ekg_100_time = np.arange(len(data_ekg_100)) / sampling_frequency_ekg_100

    plt.figure(figsize=(20, 10))
    plt.legend(loc="upper left")

    for i in range(len(data_ekg_100[0])):
        plt.plot(ekg_100_time, data_ekg_100[:, i])

    plt.legend(["Column " + str(i + 1) for i in range(len(data_ekg_100[0]))])
    plt.title('ekg_100.txt')
    plt.xlabel('Czas[s]')
    plt.ylabel('Wartość')
    plt.xlim(ekg_100_time[0], ekg_100_time[len(ekg_100_time) - 1])

    plt.show()


def exercise_1_3():
    global data_ekg_noise
    global ekg_noise_time
    global sampling_frequency_ekg_noise

    with open("./resources/ekg_noise.txt", "r+") as file_ekg_noise:
        while True:
            line = file_ekg_noise.readline()
            if not line:
                break
            values_row = line.lstrip().split()
            values_row = np.array(list(map(float, values_row)))

            data_ekg_noise.append(values_row)

    data_ekg_noise = np.array(data_ekg_noise)
    ekg_noise_time = np.arange(len(data_ekg_noise)) / \
        sampling_frequency_ekg_noise

    plt.figure(figsize=(20, 10))
    plt.legend(loc="upper left")

    for i in range(len(data_ekg_noise[0])):
        plt.plot(ekg_noise_time, data_ekg_noise[:, i])

    plt.legend(["Column " + str(i + 1) for i in range(len(data_ekg_noise[0]))])
    plt.title('ekg_noise.txt')
    plt.xlabel('Czas[s]')
    plt.ylabel('Wartość')
    plt.xlim(ekg_noise_time[0], ekg_noise_time[len(ekg_noise_time) - 1])

    plt.show()


def exercise_2_1and2():
    # Wygeneruj ciąg próbek odpowiadający fali sinusoidalnej o częstotliwości 50 Hz
    # i długości 65536.

    frequency = 50      # f = 50 Hz
    length = 65_536
    time = np.arange(length)
    sampling_rate = 2000      # fs = 2 kHz
    time_period = 1/sampling_rate       # t = 1/fs
    sinusoidal_signal = np.sin(
        2 * np.pi * frequency * time * time_period)     # sin(2 * pi * f * t)

    plt.figure(figsize=(20, 10))
    plt.plot(time, sinusoidal_signal)
    plt.title('Sygnał sinusoidalny')
    plt.ylabel('Amplituda')
    plt.xlabel('Numer próbki')
    plt.xlim(0, sampling_rate / 2)
    plt.axhline(y=0, color='black')
    plt.grid(True)
    plt.show()

    # Wyznacz dyskretną transformatę Fouriera tego sygnału i przedstaw jego widmo
    # amplitudowe na wykresie w zakresie częstotliwości [0, fs/2], gdzie fs oznacza
    # częstotliwość próbkowania.

    # Dyskretna Transformata Fouriera (DFT)
    dft_result = np.fft.fft(sinusoidal_signal)
    # Obliczenie częstotliwości
    freq = np.fft.fftfreq(length, 1/sampling_rate)
    # Indeksy częstotliwości do narysowania (0 do fs/2)
    positive_freq_indices = np.where(freq >= 0)
    # Widmo amplitudowe
    # Normalizacja przez długość sygnału
    amplitudes = np.abs(dft_result) / length
    # Podwojenie amplitud (z uwzględnieniem symetrii)
    amplitudes *= 2

    # Narysowanie widma amplitudowego
    plt.figure(figsize=(20, 10))
    plt.plot(freq[positive_freq_indices], amplitudes[positive_freq_indices])
    plt.title('Widmo Amplitudowe')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda')
    plt.xlim(0, sampling_rate / 2)  # Zakres częstotliwości [0, fs/2]
    plt.grid(True)
    plt.show()


def exercise_2_3():
    # Wygeneruj ciąg próbek mieszaniny dwóch fal sinusoidalnych (tzn. ich kombinacji
    # liniowej) o częstotliwościach 50 i 60 Hz. Wykonaj zadanie z punktu 2 dla tego
    # sygnału

    frequency_1 = 50
    frequency_2 = 60
    length = 65_536
    time = np.arange(length)
    sampling_rate = 2000      # fs = 2 kHz
    time_period = 1/sampling_rate       # t = 1/fs
    sinusoidal_signal_1 = np.sin(2 * np.pi * frequency_1 * time * time_period)
    sinusoidal_signal_2 = np.sin(2 * np.pi * frequency_2 * time * time_period)

    mixed_signal = sinusoidal_signal_1 + sinusoidal_signal_2

    plt.figure(figsize=(20, 10))
    plt.plot(time, mixed_signal)
    plt.title('Sygnał sinusoidalny')
    plt.ylabel('Amplituda')
    plt.xlabel('Numer próbki')
    plt.xlim(0, sampling_rate / 2)
    plt.axhline(y=0, color='black')
    plt.grid(True)
    plt.show()

    dft_result = np.fft.fft(mixed_signal)
    freq = np.fft.fftfreq(length, 1/sampling_rate)
    positive_freq_indices = np.where(freq >= 0)
    amplitudes = np.abs(dft_result) / length
    amplitudes *= 2

    # Narysowanie widma amplitudowego
    plt.figure(figsize=(20, 10))
    plt.plot(freq[positive_freq_indices], amplitudes[positive_freq_indices])
    plt.title('Widmo Amplitudowe')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda')
    plt.xlim(0, sampling_rate / 2)  # Zakres częstotliwości [0, fs/2]
    plt.grid(True)
    plt.show()


def exercise_2_4():
    frequency_1 = 50
    frequency_2 = 60
    length = 65_536
    time = np.arange(length)

    # Częstotliwość próbkowania 1000 Hz
    sampling_rate = 500
    time_period = 1/sampling_rate       # t = 1/fs
    sinusoidal_signal_1 = np.sin(2 * np.pi * frequency_1 * time * time_period)
    sinusoidal_signal_2 = np.sin(2 * np.pi * frequency_2 * time * time_period)

    mixed_signal = sinusoidal_signal_1 + sinusoidal_signal_2

    dft_result = np.fft.fft(mixed_signal)
    freq = np.fft.fftfreq(length, 1/sampling_rate)
    positive_freq_indices = np.where(freq >= 0)
    amplitudes = np.abs(dft_result) / length
    amplitudes *= 2

    # Narysowanie widma amplitudowego
    plt.figure(figsize=(20, 10))
    plt.plot(freq[positive_freq_indices], amplitudes[positive_freq_indices])
    plt.title('Widmo Amplitudowe')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda')
    plt.xlim(0, sampling_rate / 2)  # Zakres częstotliwości [0, fs/2]
    plt.grid(True)
    plt.show()

    # Częstotliwość próbkowania 10 kHz
    sampling_rate = 10000
    time_period = 1/sampling_rate
    sinusoidal_signal_1 = np.sin(2 * np.pi * frequency_1 * time * time_period)
    sinusoidal_signal_2 = np.sin(2 * np.pi * frequency_2 * time * time_period)

    mixed_signal = sinusoidal_signal_1 + sinusoidal_signal_2

    dft_result = np.fft.fft(mixed_signal)
    freq = np.fft.fftfreq(length, 1/sampling_rate)
    positive_freq_indices = np.where(freq >= 0)
    amplitudes = np.abs(dft_result) / length
    amplitudes *= 2

    plt.figure(figsize=(20, 10))
    plt.plot(freq[positive_freq_indices], amplitudes[positive_freq_indices])
    plt.title('Widmo Amplitudowe')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda')
    plt.xlim(0, sampling_rate / 2)  # Zakres częstotliwości [0, fs/2]
    plt.grid(True)
    plt.show()


def exercise_2_5():
    frequency = 50  # Hz

    length = 65536
    sampling_rate = 50000  # Hz

    # Czas trwania sygnału
    time = np.arange(length)

    # Wygenerowanie fali sinusoidalnej
    sinusoidal_signal = np.sin(2 * np.pi * frequency * time / sampling_rate)

    # Dyskretna Transformata Fouriera (DFT)
    dft_result = np.fft.fft(sinusoidal_signal)

    # Obliczenie częstotliwości
    freq = np.fft.fftfreq(length, 1/sampling_rate)

    # IDFT - Odwrotna Dyskretna Transformata Fouriera
    reconstructed_signal = np.fft.ifft(dft_result)

    # Plot original and reconstructed signals
    plt.figure(figsize=(20, 10))
    plt.plot(time, sinusoidal_signal, label='Oryginalny sygnał')
    plt.plot(time, reconstructed_signal,
             label='Odtworzony sygnał', linestyle='--')
    plt.title('Porównanie oryginalnego i odtworzonego sygnału')
    plt.xlabel('Numer próbki')
    plt.ylabel('Amplituda')
    plt.xlim(0, sampling_rate / 2)  # Zakres częstotliwości [0, fs/2]
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
