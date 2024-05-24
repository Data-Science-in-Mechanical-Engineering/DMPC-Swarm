import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    name_run = "testbed_experiment_dyna"
    csv_path = "/home/alex/Documents/009_Paper/robot_swarm_science_robotics/Images"
    starting_time = 100
    with open(f'../../../experiment_measurements/num_trigger_times_{name_run}_20_10_0_0_0.p', 'rb') as handle:
        trigger_cu1 = pickle.load(handle)

    with open(f'../../../experiment_measurements/num_trigger_times_{name_run}_21_10_0_0_0.p', 'rb') as handle:
        trigger_cu2 = pickle.load(handle)

    plt.plot(trigger_cu1["selected_UAVs"]["round"], trigger_cu1["selected_UAVs"]["selected"])
    plt.plot(trigger_cu2["selected_UAVs"]["round"], trigger_cu2["selected_UAVs"]["selected"])
    plt.show()

    drone_messsage_load = 7 * (2+2*9+1+2*3+2+1+1)
    cu_message_load = 2+2*45+2*9+2+1+1 + 15

    network_load = np.ones(len(trigger_cu1["selected_UAVs"]["round"])) * drone_messsage_load + cu_message_load

    for i in range(len(trigger_cu1["selected_UAVs"]["round"])):
        if trigger_cu1["selected_UAVs"]["round"][i] in trigger_cu2["selected_UAVs"]["round"]:
            network_load[i] += cu_message_load

    plt.plot(trigger_cu1["selected_UAVs"]["round"], network_load)
    plt.show()

    df = pd.DataFrame({"t": (np.array(trigger_cu1["selected_UAVs"]["round"]) - starting_time)[400:], "s": trigger_cu1["selected_UAVs"]["selected"][400:]})
    df.to_csv(f"{csv_path}/DynaNumCusTrigger1.csv", sep=",")

    df = pd.DataFrame(
        {"t": np.array(trigger_cu2["selected_UAVs"]["round"]) - starting_time, "s": trigger_cu2["selected_UAVs"]["selected"]})
    df.to_csv(f"{csv_path}/DynaNumCusTrigger2.csv", sep=",")


