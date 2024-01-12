import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    name_run = "testbed_experiment"
    starting_time = 100
    with open(f'../../../experiment_measurements/drone_trajectory_logger_{name_run}.p', 'rb') as handle:
        drone_trajectory_logger = pickle.load(handle)

    times = np.array([t for t in drone_trajectory_logger[1]])
    times = times[times > starting_time]

    times = np.sort(times)

    max_dists = -np.ones((len(times),))
    for i, t in enumerate(times):
        for drone_id in drone_trajectory_logger:
            if t in drone_trajectory_logger[drone_id]:
                d = np.linalg.norm(drone_trajectory_logger[drone_id][t][0] - drone_trajectory_logger[drone_id][t][1])
                if max_dists[i] < d:
                    max_dists[i] = d

    plt.plot(times, max_dists)
    plt.show()
