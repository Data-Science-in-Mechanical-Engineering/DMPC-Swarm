import os
import pickle as p
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import pandas as pd

import pickle


if __name__ == "__main__":
    ignore_message_loss = True
    num_drones = 16
    name = "demo"
    targets = "../../../experiment_measurements/drone_trajectory_logger_testbed_experiment_demo.p"

    with open(targets, 'rb') as handle:
        targets = pickle.load(handle)

    trajectories = "../../../experiment_measurements/ExperimentDemo.pickle"
    pos = None
    with open(trajectories, 'rb') as handle:
        data = pickle.load(handle)

    pos = data["logger_pos"]
    time_stamps = data["time"]

    dists = []
    dists2 = []

    for drone_id in range(1, 17):
        current_target = targets[drone_id][list(targets[drone_id].keys())[0]][2]
        current_target_time = targets[drone_id][list(targets[drone_id].keys())[0]][0]
        current_target_idx = 0
        dists.append([])
        dists2.append([])
        for t in range(len(pos[drone_id-1])):
            while time_stamps[t] >= current_target_time:
                current_target_idx += 1
                current_target_time = targets[drone_id][list(targets[drone_id].keys())[current_target_idx]][0]
            current_target = targets[drone_id][list(targets[drone_id].keys())[current_target_idx]][2]
            dists[-1].append(np.linalg.norm(pos[drone_id-1][t] - current_target))
            dists2[-1].append(np.linalg.norm(targets[drone_id][list(targets[drone_id].keys())[current_target_idx]][1][0:3] - current_target))
    time_stamps = np.array(time_stamps) - time_stamps[0]
    for d in dists:
        plt.plot(time_stamps, d)

    for d in dists2:
        plt.plot(time_stamps+0.5, d)

    plt.show()

    """path = os.path.dirname(os.path.abspath(__file__)) + f"/../../../{folder_name}/"

    plot_states = False
    files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith(f"simulation_result-{num_drones}_drones_simnr_")]
    num_crashed = 0
    num_optimizer_runs = 0
    num_succ_optimizer_runs = 0
    num_targets_reached = 0
    for f in files:
        result = p.load(open(f, "rb"))
        for i in range(num_drones):
            d = []
            for j in range(len(result["target_pos"])):
                #print(result["target_pos"][j])
                d.append(np.linalg.norm(result["target_pos"][j][i+1] - result[f"state_{i}"][j][0:3]))
            plt.plot(d)
        plt.show()"""