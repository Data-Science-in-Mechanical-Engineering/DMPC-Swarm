import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as p

def plot_comparison(num_drones, message_loss, ignore_message_loss, quant):
    colors = ["b", "r", "g", "k"]
    for i, name_trigger in enumerate(["COMPARISON_DMPC_MLR_DMPC"]):   # "DMPC_RR", "DMPC_HT", "DMPC_DT",
        target_reached_times = []
        cu_numbers = [1, 2, 3, 4] + [i for i in range(5, 11, 2)]
        for num_cus in cu_numbers:
            path = f"../../../hpc_runs/{name_trigger}/" \
                   f"dmpc_simulation_results_iml{ignore_message_loss}_{int(round(100 * message_loss))}_{num_cus}cus_{'quant' if quant else ''}"
            print(f"../../../hpc_runs/{name_trigger}/" \
                   f"dmpc_simulation_results_iml{ignore_message_loss}_{int(round(100 * message_loss))}_{num_cus}cus_{'quant' if quant else ''}"
           )
            # f"dmpc/dmpc_simulation_results_iml{ignore_message_loss}_{int(round(100 * message_loss))}_{num_cus}cus_{'' if not quant else 'quant'}"
            # print(path)
            files = [os.path.join(path, f) for f in os.listdir(path) if
                     f.startswith(f"simulation_result-{num_drones}_drones_simnr_")]

            target_reached_time = []
            # print(files)
            for f in files:
                result = p.load(open(f, "rb"))
                if result["crashed"][0]:
                    continue
                target_reached_time.append(result["target_reached_time"][0])
            print(len(target_reached_time))
            target_reached_times.append(target_reached_time)

        # print(target_reached_times)
        target_reached_times_np = np.array(target_reached_times)

        boxplot_data = {}
        boxplot_data["index"] = np.array(cu_numbers)
        boxplot_data["median"] = np.median(target_reached_times_np, axis=1)
        boxplot_data["box_top"] = np.percentile(target_reached_times_np, q=75, axis=1)
        boxplot_data["box_bottom"] = np.percentile(target_reached_times_np, q=25, axis=1)
        boxplot_data["whisker_top"] = np.percentile(target_reached_times_np, q=100, axis=1)
        boxplot_data["whisker_bottom"] = np.percentile(target_reached_times_np, q=0, axis=1)

        df = pd.DataFrame(data=boxplot_data)
        df.to_csv(
            f"../../../experiment_measurements/ArrivalTime_UAVs{num_drones}_{name_trigger}.csv",
            sep=" ", header=False, index=False)


        box1 = plt.boxplot(target_reached_times)
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box1[item], color=colors[i])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_comparison(num_drones=10, message_loss=0.01, ignore_message_loss=False, quant=False)