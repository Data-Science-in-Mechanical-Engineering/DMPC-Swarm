import os
import numpy as np
import pickle as p

import matplotlib.pyplot as plt

import pandas as pd

if __name__ == "__main__":
    # plot_comparison(16, 0.01, ignore_message_loss=False, quant=False)
    colors = ["b", "r", "g"]

    message_loss_prob = 0.01
    num_cus = 2
    simulate_quantization = False
    num_drones = 16
    folder_name = "hpc_runs/COMPARISON_DMPC_MLR_DMPC"

    colors = ["b", "r", "g"]

    message_loss_probs = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    for i, ignore_message_loss in enumerate([False, True]):
        min_dists = []
        success_rates = []
        for message_loss_prob in message_loss_probs:


            path = os.path.dirname(os.path.abspath(__file__)) + f"/../../../{folder_name}/" \
                   + f"dmpc_simulation_results_iml{ignore_message_loss}_{int(100 * message_loss_prob + 1e-7)}_{num_cus}cus_{'quant' if simulate_quantization else ''}"

            plot_states = False
            files = [os.path.join(path, f) for f in os.listdir(path) if
                     f.startswith(f"simulation_result-{num_drones}_drones_simnr_")]
            min_dists_temp = []
            success_rates_temp = 0
            for f in files:
                result = p.load(open(f, "rb"))

                min_dists_temp.append(result["min_inter_drone_dist"][0])
                success_rates_temp += result["num_targets_reached"][0]

            min_dists.append(min_dists_temp)
            success_rates.append(success_rates_temp / (len(files) * num_drones))

        box1 = plt.boxplot(min_dists, whis=(1, 99))
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color=colors[i])

        min_dists = np.array(min_dists)
        boxplot_data = {}
        boxplot_data["index"] = np.array(message_loss_probs) * 100
        boxplot_data["median"] = np.median(min_dists, axis=1)
        boxplot_data["box_top"] = np.percentile(min_dists, q=75, axis=1)
        boxplot_data["box_bottom"] = np.percentile(min_dists, q=25, axis=1)
        boxplot_data["whisker_top"] = np.percentile(min_dists, q=100, axis=1)
        boxplot_data["whisker_bottom"] = np.percentile(min_dists, q=0, axis=1)
        boxplot_data["success_rates"] = np.array(success_rates)*100

        df = pd.DataFrame(data=boxplot_data)
        df.to_csv(
            f"/home/alex/Documents/009_Paper/robot_swarm_science_robotics/plot_data/Inter_UAV_dist_{num_drones}_{ignore_message_loss}.csv",
            sep=" ", header=False, index=False)

    plt.show()
