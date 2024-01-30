import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    name_run = "testbed_experiment_dyna"
    csv_path = "/users/mf724021/Documents/009_Paper/robot_swarm_science_robotics/Images"
    starting_time = 100
    with open(f'../../../experiment_measurements/drone_trajectory_logger_{name_run}.p', 'rb') as handle:
        drone_trajectory_logger = pickle.load(handle)

    times = np.array([t for t in drone_trajectory_logger[1]])
    times = times[times > starting_time]

    times = np.sort(times)

    max_dists = -np.ones((len(times),))
    min_inter_dists = np.ones((len(times),)) * 100000
    for i, t in enumerate(times):
        for drone_id in drone_trajectory_logger:
            if t in drone_trajectory_logger[drone_id]:
                d = np.linalg.norm(drone_trajectory_logger[drone_id][t][0] - drone_trajectory_logger[drone_id][t][1])
                if max_dists[i] < d:
                    max_dists[i] = d

                for other_drone_id in drone_trajectory_logger:
                    if other_drone_id != drone_id and t in drone_trajectory_logger[other_drone_id]:
                        d = np.linalg.norm(
                            drone_trajectory_logger[drone_id][t][0] - drone_trajectory_logger[other_drone_id][t][0])
                        if min_inter_dists[i] > d:
                            min_inter_dists[i] = d


    df = pd.DataFrame({"t": (times-starting_time)*0.2, "d": max_dists, "m": min_inter_dists})
    # df.to_csv(f"{csv_path}/MessageLossHardwareExperiments_{name_run}.csv", sep=",")

    plt.plot(max_dists) #times, max_dists)
    plt.show()

    plt.plot(times, min_inter_dists)
    plt.show()
