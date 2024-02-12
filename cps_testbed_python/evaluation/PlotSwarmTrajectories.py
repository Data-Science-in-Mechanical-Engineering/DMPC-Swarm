
import numpy as np
import pickle
import matplotlib.pyplot as plt

import pandas as pd


def main():
    offset = 0  #int(0.5 / 0.01)
    time_length = 40000 # int(10 / 0.01)
    id = 4
    save_file = "../../../experiment_measurements/ExperimentBigSwarmCrash.pickle"
    pos = None
    with open(save_file, 'rb') as handle:
        pos = pickle.load(handle)
    
    # fft_data = np.fft.fft(pos[2000:, 2] - np.mean(pos[2000:, 2]))
    # f = np.array(range(len(fft_data))) / (0.01*len(fft_data))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data = {}
    for i in range(len(pos)):
        print(i)
        p = np.array(pos[i][:])
        print(p)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.1)
        data[f"{i}d0"] = p[0::10, 0]
        data[f"{i}d1"] = p[0::10, 1]
        data[f"{i}d2"] = p[0::10, 2]

    min_dists = np.ones((len(pos[0]),)) * 1000000
    for t in range(len(pos[0])):
        for i in range(len(pos)):
            for j in range(len(pos)):
                if i != j:
                    pi = np.array(pos[i][t])
                    pj = np.array(pos[j][t])
                    d = np.linalg.norm(pi - pj)
                    if d < min_dists[t]:
                        min_dists[t] = d

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    data["mindists"] = min_dists[0::10]
    data["t"] = np.arange(len(data["mindists"])) * 0.1
    df = pd.DataFrame(data)
    df.to_csv("../../../experiment_measurements/BigSwarm.csv")
    eps = 1e-16
    ax.axes.set_xlim3d(left=-2-eps, right=2+eps)
    ax.axes.set_ylim3d(bottom=-2-eps, top=2+eps) 
    ax.axes.set_zlim3d(bottom=0, top=3)
    """ax = fig.add_subplot()
    ax.plot(p[:, 0])
    ax = fig.add_subplot()
    ax.plot(p[:, 1])
    ax = fig.add_subplot()
    ax.plot(p[:, 2]"""
    plt.show()

    plt.figure()
    plt.plot(min_dists)
    plt.show()


if __name__ == "__main__":
    main()
