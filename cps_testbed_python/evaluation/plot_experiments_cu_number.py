import os
import matplotlib.pyplot as plt
import pickle as p

import numpy as np
import pandas as pd

if __name__ == "__main__":
    ignore_message_loss = False
    simulate_quantization = False
    num_drones = 10
    num_cu_data = {}
    trigger = "HT"

    probs = [int(1*i) for i in range(11)]
    csv_path = "/home/alex/Documents/009_Paper/papers-dsme-nes/robot_swarm_science_robotics/plot_data"

    folder_name = "/data/hpc_runs/dmpc/COMPARISON_TRIGGERS"

    target_reached_times_per_cu = {}

    for num_cus in [2, 3, 5, 7, 9, 11]:
        num_cu_data[num_cus] = []
        target_reached_times_per_prob = []
        for message_loss_prob in probs:
            path = f"{folder_name}/" \
                   + f"dmpc_simulation_results_iml{ignore_message_loss}_{message_loss_prob}_{num_cus}cus_{'quant' if simulate_quantization else ''}_{trigger}"

            plot_states = False
            files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith(f"simulation_result-{num_drones}_drones_simnr_")]
            num_crashed = 0
            num_optimizer_runs = 0
            num_succ_optimizer_runs = 0
            num_targets_reached = 0

            target_reached_times = []
            for f in files:
                result = p.load(open(f, "rb"))
                if result["crashed"][0]:
                    num_crashed += 1

                if result["num_targets_reached"][0] != num_drones:
                    print(result["num_targets_reached"][0])
                    #fig = plt.figure()
                    #ax = plt.axes(projection='3d')
                    """for j in range(num_drones):
                        states = result[f"state_{j}"]
                        xline = []
                        yline = []
                        zline = []
                        for s in states:
                            xline.append(s[0])
                            yline.append(s[1])
                            zline.append(s[2])
                        #ax.plot3D(xline, yline, zline)"""
                else:
                    target_reached_times.append(result["target_reached_time"][0])


                num_optimizer_runs += result["num_optimizer_runs"][0]
                num_succ_optimizer_runs += result["num_succ_optimizer_runs"][0]
                num_targets_reached += result["num_targets_reached"][0]

            if len(target_reached_times) != 0:
                target_reached_times_per_prob.append(np.percentile(np.array(target_reached_times), q=99))
            else:
                target_reached_times_per_prob.append(200)

            num_cu_data[num_cus].append(num_targets_reached/(num_drones*len(files))*100)

        target_reached_times_per_cu[num_cus] = target_reached_times_per_prob

    for num_cus in num_cu_data:
        plt.plot(probs, num_cu_data[num_cus], label=f"{num_cus} CUs")
        df = pd.DataFrame({"p": probs, "s": num_cu_data[num_cus], "t": target_reached_times_per_cu[num_cus]})
        df.to_csv(f"{csv_path}/SuccRateNumDrones{num_drones}CUs{num_cus}.csv", sep=",")

    plt.xlabel("Message loss probability (%)")
    plt.ylabel("Success rate")
    plt.legend()
    plt.show()
