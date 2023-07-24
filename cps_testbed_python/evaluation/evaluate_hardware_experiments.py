import pickle
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


def evaluate_num_trigger_times(cu_ids, alpha_1, alpha_2, alpha_3, alpha_4, simulated):
	num_trigger_times = None

	for cu_id in cu_ids:
		sim = "" if not simulated else "_sim"
		path = f'../../../experiment_measurements/num_trigger_times{sim}{cu_id}_{int(alpha_1)}_{int(alpha_2)}_{int(alpha_3)}_{int(alpha_4)}.p'
		with open(path, "rb") as file:
			data = pickle.load(file)["num_trigger_times"]
		if num_trigger_times is None:
			num_trigger_times = data
		else:
			num_trigger_times = {agent_id: num_trigger_times[agent_id] + data[agent_id] for agent_id in data}

	print(num_trigger_times)

	csv_data = {"time": None}
	for cu_id in cu_ids:
		sim = "" if not simulated else "_sim"
		path = f'../../../experiment_measurements/num_trigger_times{sim}{cu_id}_{int(alpha_1)}_{int(alpha_2)}_{int(alpha_3)}_{int(alpha_4)}.p'
		with open(path, "rb") as file:
			data = pickle.load(file)["selected_UAVs"]
		x = [i for i in range(len(data))]
		plt.scatter(x, data, marker="x")
		csv_data["time"] = np.array(x) * 0.2
		csv_data[str(cu_id)] = data

	plt.show()

	df = pd.DataFrame(data=csv_data)
	df.to_csv(f"../../../experiment_measurements/num_trigger_times{sim}_{int(alpha_1)}_{int(alpha_2)}_{int(alpha_3)}_{int(alpha_4)}.csv",
			  sep=" ", header=False)


def evaluate_trajectory(path, name):
	with open(path, "rb") as file:
		data = pickle.load(file)

	csv_data = {}
	for i in range(len(data)):
		p = np.array(data[i])
		x = np.array([i for i in range(len(p))])
		csv_data[f"{str(i)}_1"] = p[::10, 0]
		csv_data[f"{str(i)}_2"] = p[::10, 1]
		csv_data[f"{str(i)}_3"] = p[::10, 2]
		plt.plot(x, p[:, 0])
	plt.show()

	df = pd.DataFrame(data=csv_data)
	df.to_csv(
		f"../../../experiment_measurements/{name}.csv",
		sep=" ", header=False)


def evaluate_trajectory_simulation(path, name, num_drones):
	with open(path, "rb") as file:
		result = pickle.load(file)
	csv_data = {}
	for i in range(num_drones):
		states = np.array(result[f"state_{i}"])
		csv_data[f"{str(i)}_1"] = states[1800::10, 0]
		csv_data[f"{str(i)}_2"] = states[1800::10, 1]
		csv_data[f"{str(i)}_3"] = states[1800::10, 2]
		x = np.array([i for i in range(len(csv_data[f"{str(i)}_1"]))])
		plt.plot(csv_data[f"{str(i)}_1"], csv_data[f"{str(i)}_2"])
	plt.show()

	df = pd.DataFrame(data=csv_data)
	df.to_csv(
		f"../../../experiment_measurements/{name}.csv",
		sep=" ", header=False)


if __name__ == "__main__":
	evaluate_num_trigger_times(cu_ids=[40, 41], alpha_1=1*1, alpha_2=10*0, alpha_3=1*0, alpha_4=1, simulated=True)

	path = "../../../experiment_measurements/ExperimentCircleBad.pickle"
	#evaluate_trajectory(path)

	path = "../../../batch_simulation_results/dmpc/dmpc_simulation_results_not_ignore_message_loss_demo1/simulation_result-6_drones_simnr_1.pkl"
	evaluate_trajectory_simulation(path, "CircleET", 6)
