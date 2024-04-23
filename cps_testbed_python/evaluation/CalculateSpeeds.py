import numpy as np
import pickle
import matplotlib.pyplot as plt

import pandas as pd


def main():
    save_file = "../../../experiment_measurements/ExperimentBigSwarm.pickle"
    with open(save_file, 'rb') as handle:
        data = pickle.load(handle)

    pos = data["logger_pos"]
    time_stamps = data["time"]

    # fft_data = np.fft.fft(pos[2000:, 2] - np.mean(pos[2000:, 2]))
    # f = np.array(range(len(fft_data))) / (0.01*len(fft_data))
    data = {}
    speeds_all = {}
    accs_all = {}
    for i in range(len(pos)):
        p = np.array(pos[i][:])
        t = np.array(time_stamps)
        dt = np.tile(np.array([t[1:] - t[:len(p) - 1]]).T, (1, 3))
        print(dt)
        speeds_all[i] = (p[1:, :] - p[:len(p) - 1, :]) / dt
        accs_all[i] = np.linalg.norm(speeds_all[i][1:, :] - speeds_all[i][:len(speeds_all[i]) - 1, :], axis=1) / dt[1:, 0]
        speeds_all[i] = np.linalg.norm(speeds_all[i], axis=1)


    speeds = []
    accs = []
    for i in range(len(pos)):
        plt.plot(time_stamps[1:], speeds_all[i])
        speeds.append(max(speeds_all[i]))
        accs.append(max(accs_all[i]))
    print(max(speeds))
    print(max(accs))
    plt.show()


if __name__ == "__main__":
    main()
