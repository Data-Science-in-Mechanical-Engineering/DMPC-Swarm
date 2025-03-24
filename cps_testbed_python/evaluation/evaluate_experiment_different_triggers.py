import copy
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
    num_cus = 2
    name = ("DT")
    targets = f"../../experiment_measurements/drone_trajectory_logger_testbed_experiment_demo_{name}.p"

    with open(targets, 'rb') as handle:
        targets = pickle.load(handle)


    trajectories = f"../../experiment_measurements/Experiment{name}.pickle"
    pos = None
    with open(trajectories, 'rb') as handle:
        data = pickle.load(handle)

    pos = data["logger_pos"]
    time_stamps = data["time"]
    # print(time_stamps)
    # print(len(targets))
    # print(targets.keys())

    dists = []
    dists2 = []
    times = []
    offset = None
    for drone_id in range(1, 17):
        current_target_idx = 1
        while drone_id not in targets[list(targets.keys())[current_target_idx]][1].keys():
            current_target_idx += 1

        current_target = targets[list(targets.keys())[current_target_idx]][1][drone_id]
        current_target_time = targets[list(targets.keys())[current_target_idx]][0]
        dists.append([])
        dists2.append([])
        times.append([])
        print("---------------------")
        for t in range(len(pos[drone_id-1])):
            while time_stamps[t] >= current_target_time:
                current_target_idx += 1
                if current_target_idx < len(targets):
                    current_target_time = targets[list(targets.keys())[current_target_idx]][0]
                else:
                    current_target_idx -= 1
                    break
            old_current_target = copy.deepcopy(current_target)
            current_target = targets[list(targets.keys())[current_target_idx]][1][drone_id]
            if np.linalg.norm(current_target - old_current_target) > 0.1 and offset is None:
                offset = time_stamps[t]

            dists[-1].append(np.linalg.norm(pos[drone_id-1][t] - current_target))
            times[-1].append(time_stamps[t])

    dists = np.array(dists)

    times = np.array(times) - offset

    dists = dists[:, times[0]>=0]
    times = times[0, times[0]>=0]

    print(dists.shape)
    for i in range(len(dists)):
        plt.plot(times, dists[i])

    df = pd.DataFrame({"t": times[0::10], "dmax": np.max(dists, axis=0)[0::10], "dmin": np.min(dists, axis=0)[0::10], "mean": np.mean(dists, axis=0)[0::10]})
    df.to_csv(
        f"/home/alex/Documents/009_Paper/papers-dsme-nes/dmpc/plot_data/HardwareExperimentFigures_{name}.csv")

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