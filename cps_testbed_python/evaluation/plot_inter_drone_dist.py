import os
import numpy as np
import pickle as p

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # plot_comparison(16, 0.01, ignore_message_loss=False, quant=False)
    colors = ["b", "r", "g"]

    message_loss_prob = 0.01
    num_cus = 2
    simulate_quantization = False
    num_drones = 16
    folder_name = "hpc_runs/None"

    for i, ignore_message_loss in enumerate([False, True]):


        path = os.path.dirname(os.path.abspath(__file__)) + f"/../../../{folder_name}/" \
               + f"dmpc_simulation_results_iml{ignore_message_loss}_{int(100 * message_loss_prob + 1e-7)}_{num_cus}cus_{'quant' if simulate_quantization else ''}"

        plot_states = False
        files = [os.path.join(path, f) for f in os.listdir(path) if
                 f.startswith(f"simulation_result-{num_drones}_drones_simnr_")]
        min_dists = []
        for f in files:
            result = p.load(open(f, "rb"))

            if "min_inter_drone_dist" in result:
                min_dists.append(result["min_inter_drone_dist"][0])

        colors = ["b", "r", "g"]
        box1 = plt.boxplot([min_dists], whis=(1, 99))
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color=colors[i])

    plt.show()
