import os
import pickle as p
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    ignore_message_loss = True
    num_drones = 16
    folder_name = "simulation_results/dmpc/demo_visitors"   # hpc_runs/DMPC_MESSAGE_LOSS"
    path = os.path.dirname(os.path.abspath(__file__)) + f"/../../../{folder_name}/"

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
        plt.show()