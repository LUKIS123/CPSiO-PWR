import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utils


data_ekg1 = []
ekg1_time = None
sampling_frequency_ekg1 = 1000


def main():
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
    ekg1_time = np.arange(len(data_ekg1[0])) / sampling_frequency_ekg1

    ekg1_time = np.empty(len(data_ekg1))
    for i in range(len(data_ekg1)):
        ekg1_time[i] = i / sampling_frequency_ekg1

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


if __name__ == '__main__':
    main()
