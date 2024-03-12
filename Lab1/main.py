import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


if __name__ == '__main__':
    main()
