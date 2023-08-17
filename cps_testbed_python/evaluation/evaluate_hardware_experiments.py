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
			data = pickle.load(file)["selected_UAVs"]["selected"]
			data = data[300:390]
		x = [i for i in range(len(data))]
		print(len(data))
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

		csv_data[f"{str(i)}_1"] = p[3000:5000:10, 0]
		csv_data[f"{str(i)}_2"] = p[3000:5000:10, 1]
		csv_data[f"{str(i)}_3"] = p[3000:5000:10, 2]
		x = np.array([i for i in range(len(csv_data[f"{str(i)}_1"]))])
		plt.plot(x, csv_data[f"{str(i)}_1"])
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
		csv_data[f"{str(i)}_1"] = states[3000::10, 0]
		csv_data[f"{str(i)}_2"] = states[3000::10, 1]
		csv_data[f"{str(i)}_3"] = states[3000::10, 2]
		x = np.array([i for i in range(len(csv_data[f"{str(i)}_1"]))])
		plt.plot(csv_data[f"{str(i)}_1"], csv_data[f"{str(i)}_2"])
	plt.show()

	df = pd.DataFrame(data=csv_data)
	df.to_csv(
		f"../../../experiment_measurements/{name}.csv",
		sep=" ", header=False)


def compare_simulation_real_world1(path_simulation, path_real_world, path_real_world_setpoints, num_drones, name):
	with open(path_simulation, "rb") as file:
		result = pickle.load(file)
	csv_data = {}
	for i in range(num_drones):
		states = np.array(result[f"state_{i}"])
		csv_data[f"{str(i)}_1"] = states[:1000:10, 0]
		csv_data[f"{str(i)}_2"] = states[:1000:10, 1]
		csv_data[f"{str(i)}_3"] = states[:1000:10, 2]
		x = np.array([i for i in range(len(csv_data[f"{str(i)}_1"]))])
		plt.plot(csv_data[f"{str(i)}_1"], csv_data[f"{str(i)}_2"])

	df = pd.DataFrame(data=csv_data)
	df.to_csv(
		f"../../../experiment_measurements/{name}Simulated.csv",
		sep=" ", header=False)

	csv_data = {}
	for i in range(num_drones):
		states = np.array(result[f"state_set{i}"])
		csv_data[f"{str(i)}set_1"] = states[::10, 0]
		csv_data[f"{str(i)}set_2"] = states[::10, 1]
		csv_data[f"{str(i)}set_3"] = states[::10, 2]
		x = np.array([i for i in range(len(csv_data[f"{str(i)}set_1"]))])
		plt.plot(csv_data[f"{str(i)}set_1"][-200:], csv_data[f"{str(i)}set_2"][-200:], "--")

	df = pd.DataFrame(data=csv_data)
	df.to_csv(
		f"../../../experiment_measurements/{name}SimulatedSetpoints.csv",
		sep=" ", header=False)

	with open(path_real_world, "rb") as file:
		data = pickle.load(file)

	#plt.figure()
	csv_data = {}
	for i in range(len(data)):
		p = np.array(data[i])

		csv_data[f"{str(i)}_1"] = p[3000::10, 0]
		csv_data[f"{str(i)}_2"] = p[3000::10, 1]
		csv_data[f"{str(i)}_3"] = p[3000::10, 2]
		x = np.array([i for i in range(len(csv_data[f"{str(i)}_1"]))])
		plt.plot(csv_data[f"{str(i)}_1"], csv_data[f"{str(i)}_2"])

	df = pd.DataFrame(data=csv_data)
	df.to_csv(
		f"../../../experiment_measurements/{name}Real.csv",
		sep=" ", header=False)

	with open(path_real_world_setpoints, "rb") as file:
		data = pickle.load(file)

	csv_data = {}
	for i in range(num_drones):
		p = np.array(data[i])

		csv_data[f"{str(i)}set_1"] = []
		csv_data[f"{str(i)}set_2"] = []
		csv_data[f"{str(i)}set_3"] = []
		for j in range(len(data)):
			csv_data[f"{str(i)}set_1"].append(data[j][i+1][0])
			csv_data[f"{str(i)}set_2"].append(data[j][i+1][1])
			csv_data[f"{str(i)}set_3"].append(data[j][i+1][2])
		x = np.array([i for i in range(len(csv_data[f"{str(i)}set_1"]))])
		csv_data[f"{str(i)}set_1"] = np.array(csv_data[f"{str(i)}set_1"])
		csv_data[f"{str(i)}set_2"] = np.array(csv_data[f"{str(i)}set_2"])
		csv_data[f"{str(i)}set_3"] = np.array(csv_data[f"{str(i)}set_3"])
		plt.plot(csv_data[f"{str(i)}set_1"][200:300], csv_data[f"{str(i)}set_2"][200:300], ":")
	plt.show()

	df = pd.DataFrame(data=csv_data)
	df.to_csv(
		f"../../../experiment_measurements/{name}RealSetpoints.csv",
		sep=" ", header=False)


def compare_simulation_real_world(path_simulation, path_real_world, path_real_world_setpoints, num_drones, name):
	with open(path_simulation, "rb") as file:
		result = pickle.load(file)
	csv_data = {}
	for i in range(num_drones):
		states = np.array(result[f"state_{i}"])
		csv_data[f"{str(i)}_1"] = states[:1000:10, 0]
		csv_data[f"{str(i)}_2"] = states[:1000:10, 1]
		csv_data[f"{str(i)}_3"] = states[:1000:10, 2]
		x = np.array([i for i in range(len(csv_data[f"{str(i)}_1"]))])
		#plt.plot(csv_data[f"{str(i)}_1"], csv_data[f"{str(i)}_2"])

	df = pd.DataFrame(data=csv_data)
	df.to_csv(
		f"../../../experiment_measurements/{name}Simulated.csv",
		sep=" ", header=False)

	csv_data = {}
	for i in range(num_drones):
		states = np.array(result[f"state_set{i}"])
		csv_data[f"{str(i)}set_1"] = states[::, 0]
		csv_data[f"{str(i)}set_2"] = states[::, 1]
		csv_data[f"{str(i)}set_3"] = states[::, 2]
		x = np.array([i for i in range(len(csv_data[f"{str(i)}set_1"]))])
		#plt.plot(x+1, csv_data[f"{str(i)}set_1"][-200:], "--") # , csv_data[f"{str(i)}set_2"][-200:], "--")
		plt.plot(csv_data[f"{str(i)}set_1"][:], csv_data[f"{str(i)}set_2"][:], "--")

	df = pd.DataFrame(data=csv_data)
	df.to_csv(
		f"../../../experiment_measurements/{name}SimulatedSetpoints.csv",
		sep=" ", header=False)

	with open(path_real_world, "rb") as file:
		data = pickle.load(file)

	#plt.figure()
	csv_data = {}
	for i in range(len(data)):
		p = np.array(data[i])

		csv_data[f"{str(i)}_1"] = p[0:1000:10, 0]
		csv_data[f"{str(i)}_2"] = p[0:1000:10, 1]
		csv_data[f"{str(i)}_3"] = p[0:1000:10, 2]
		x = np.array([i for i in range(len(csv_data[f"{str(i)}_1"]))])
		#plt.plot(csv_data[f"{str(i)}_1"], csv_data[f"{str(i)}_2"])

	df = pd.DataFrame(data=csv_data)
	df.to_csv(
		f"../../../experiment_measurements/{name}Real.csv",
		sep=" ", header=False)

	with open(path_real_world_setpoints, "rb") as file:
		data = pickle.load(file)

	csv_data = {}
	for i in range(num_drones):
		p = np.array(data[i])

		csv_data[f"{str(i)}set_1"] = []
		csv_data[f"{str(i)}set_2"] = []
		csv_data[f"{str(i)}set_3"] = []
		for j in range(len(data)):
			csv_data[f"{str(i)}set_1"].append(data[j][i+1][0])
			csv_data[f"{str(i)}set_2"].append(data[j][i+1][1])
			csv_data[f"{str(i)}set_3"].append(data[j][i+1][2])
		x = np.array([i for i in range(len(csv_data[f"{str(i)}set_1"]))])
		csv_data[f"{str(i)}set_1"] = np.array(csv_data[f"{str(i)}set_1"])
		csv_data[f"{str(i)}set_2"] = np.array(csv_data[f"{str(i)}set_2"])
		csv_data[f"{str(i)}set_3"] = np.array(csv_data[f"{str(i)}set_3"])
		# plt.plot(x[0:100], csv_data[f"{str(i)}set_1"][0:100])  #, csv_data[f"{str(i)}set_2"][0:100], ":")
		plt.plot(csv_data[f"{str(i)}set_1"][0:100], csv_data[f"{str(i)}set_2"][0:100], ":")

	plt.show()

	df = pd.DataFrame(data=csv_data)
	df.to_csv(
		f"../../../experiment_measurements/{name}RealSetpoints.csv",
		sep=" ", header=False)


if __name__ == "__main__":
	path_simulation = "../../../batch_simulation_results/dmpc/dmpc_simulation_results_not_ignore_message_loss_demo1/simulation_result-6_drones_simnr_1.pkl"
	path_real_world = "../../../experiment_measurements/FinalResults/ExperimentCompare.pickle"
	path_real_world_setpoints = "../../../experiment_measurements/CU20setpoints450.p"
	compare_simulation_real_world(path_simulation, path_real_world, path_real_world_setpoints, 6, "Demo2")
	exit(0)

	evaluate_num_trigger_times(cu_ids=[20, 21], alpha_1=10, alpha_2=0, alpha_3=0, alpha_4=10, simulated=False)

	path = "../../../experiment_measurements/FinalResults/ExperimentDemoGood.pickle"
	evaluate_trajectory(path, "DemoGood")

	#path = "../../../batch_simulation_results/dmpc/dmpc_simulation_results_not_ignore_message_loss_demo1/simulation_result-6_drones_simnr_1.pkl"
	#evaluate_trajectory_simulation(path, "CircleET", 6)
