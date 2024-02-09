import os
import matplotlib.pyplot as plt
import pickle as p
import pandas as pd

if __name__ == "__main__":
    ignore_message_loss = False
    simulate_quantization = False
    num_drones = 16
    num_cu_data = {}

    probs = [int(1*i) for i in range(10)]
    csv_path = "/home/alex/Documents/009_Paper/robot_swarm_science_robotics/Images"

    folder_name = "hpc_runs/DMPC_MESSAGE_LOSS"

    for num_cus in [2, 3, 5, 7, 9, 11]:
        num_cu_data[num_cus] = []
        for message_loss_prob in probs:
            path = os.path.dirname(os.path.abspath(__file__)) + f"/../../../{folder_name}/" \
                   + f"dmpc_simulation_results_iml{ignore_message_loss}_{message_loss_prob}_{num_cus}cus_{'quant' if simulate_quantization else ''}"

            plot_states = False
            files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith(f"simulation_result-{num_drones}_drones_simnr_")]
            num_crashed = 0
            num_optimizer_runs = 0
            num_succ_optimizer_runs = 0
            num_targets_reached = 0
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


                num_optimizer_runs += result["num_optimizer_runs"][0]
                num_succ_optimizer_runs += result["num_succ_optimizer_runs"][0]
                num_targets_reached += result["num_targets_reached"][0]

            num_cu_data[num_cus].append(num_targets_reached/(num_drones*len(files))*100)

    for num_cus in num_cu_data:
        plt.plot(probs, num_cu_data[num_cus], label=f"{num_cus} CUs")
        df = pd.DataFrame({"p": probs, "s": num_cu_data[num_cus]})
        df.to_csv(f"{csv_path}/SuccRateNumDrones{num_drones}CUs{num_cus}.csv", sep=",")

    plt.xlabel("Message loss probability (%)")
    plt.ylabel("Success rate")
    plt.legend()
    plt.show()
