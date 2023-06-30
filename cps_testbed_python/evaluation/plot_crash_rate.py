import os
import pickle as p
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def load_data(path, filenames, nums_cus=[1, 3, 5, 7, 9, 11, 13, 15], nums_drones=[10, 15]):
	num_crashed = {}
	num_optimizer_runs = {}
	num_succ_optimizer_runs = {}
	num_targets_reached = {}
	num_experiments = {}
	index = 0
	for filename in filenames:
		for num_drones in nums_drones:
			files = [os.path.join(path, filename, f) for f in os.listdir(os.path.join(path, filename)) if f.startswith(f"simulation_result-{num_drones}_drones_simnr_")]
			print(files)
			num_crashed[num_drones*1000 + nums_cus[index]] = 0
			num_optimizer_runs[num_drones*1000 + nums_cus[index]] = 0
			num_succ_optimizer_runs[num_drones*1000 + nums_cus[index]] = 0
			num_targets_reached[num_drones*1000 + nums_cus[index]] = 0
			num_experiments[num_drones*1000 + nums_cus[index]] = 0
			for f in files:
				result = p.load(open(f, "rb"))
				if result["crashed"][0]:
					num_crashed[num_drones*1000 + nums_cus[index]] += 1

				num_optimizer_runs[num_drones*1000 + nums_cus[index]] += result["num_optimizer_runs"][0]
				num_succ_optimizer_runs[num_drones*1000 + nums_cus[index]] += result["num_succ_optimizer_runs"][0]
				num_targets_reached[num_drones*1000 + nums_cus[index]] += result["num_targets_reached"][0]
				num_experiments[num_drones * 1000 + nums_cus[index]] += 1
		index += 1
	return num_optimizer_runs, num_succ_optimizer_runs, num_targets_reached, num_experiments, num_crashed


def plot_crash_rate(ax, num_crashed, num_experiments, nums_drones, nums_cus, marker):
	crash_rate = {num_drones: [] for num_drones in nums_drones}
	x_data = {num_drones: [] for num_drones in nums_drones}
	index = 0
	for num_drones in nums_drones:
		for num_cus in nums_cus:
			if num_cus <= num_drones:
				crash_rate[num_drones].append(num_crashed[num_drones * 1000 + num_cus] / (
							num_experiments[num_drones * 1000 + num_cus]+1e-9) * 100)
				x_data[num_drones].append(num_cus)

		ax.scatter(x_data[num_drones], crash_rate[num_drones], color=colors[index], marker=marker)
		index += 1

def plot_arrival_rate(ax, num_targets_reached, num_experiments, nums_drones, nums_cus, marker):
	crash_rate = {num_drones: [] for num_drones in nums_drones}
	x_data = {num_drones: [] for num_drones in nums_drones}
	index = 0
	for num_drones in nums_drones:
		for num_cus in nums_cus:
			if num_cus <= num_drones:
				crash_rate[num_drones].append(num_targets_reached[num_drones * 1000 + num_cus] / (
							num_experiments[num_drones * 1000 + num_cus] * num_drones+1e-9) * 100)
				x_data[num_drones].append(num_cus)

		ax.scatter(x_data[num_drones], crash_rate[num_drones], color=colors[index], marker=marker)
		index += 1


if __name__ == "__main__":
	nums_cus = [1, 3, 5, 7, 9, 11, 13]
	nums_drones = [10, 15]
	filenames_mlrdmpc = [f"dmpc_simulation_results_not_ignore_message_loss_001_{num_cus}_cus3" for num_cus in nums_cus]
	filenames_dmpc = [f"dmpc_simulation_results_ignore_message_loss_005_{num_cus}_cus3" for num_cus in nums_cus]
	xlabels = [f"M = {num_cus}" for num_cus in nums_cus]

	path = "C:\\Users\\mf724021\\Documents\\003_Testbed\\007_Code\\CPSTestbed\\simulation\\CPSTestbed\\" \
		   "batch_simulation_results\\dmpc"
	#path = str(Path('~').expanduser()) + "/Documents/AntiCollision/cpstestbed/simulation/CPSTestbed/batch_simulation_results/" \
	#									  "dmpc"

	num_optimizer_runs_mlrdmpc, num_succ_optimizer_runs_mlrdmpc, num_targets_reached_mlrdmpc, num_experiments_mlrdmpc, num_crashed_mlrdmpc = load_data(path, filenames_mlrdmpc, nums_cus=nums_cus, nums_drones=nums_drones)
	num_optimizer_runs_dmpc, num_succ_optimizer_runs_dmpc, num_targets_reached_dmpc, num_experiments_dmpc, num_crashed_dmpc = load_data(path, filenames_dmpc, nums_cus=nums_cus, nums_drones=nums_drones)

	colors = ['blue', 'red', 'green']
	fig = plt.figure()

	# Creating axes instance
	ax = plt.axes()

	plot_crash_rate(ax, num_crashed=num_crashed_mlrdmpc, num_experiments= num_experiments_mlrdmpc, nums_drones=nums_drones, nums_cus=nums_cus, marker="o")
	plot_crash_rate(ax, num_crashed=num_crashed_dmpc, num_experiments=num_experiments_dmpc, nums_drones=nums_drones, nums_cus=nums_cus, marker="x")

	plt.ylabel("Crash rate (%)")
	plt.xlabel("Number CUs")

	legend_elements = [Patch(facecolor=colors[i], edgecolor='k',
							 label=f"Number drones = {nums_drones[i]}") for i in range(len(nums_drones))] + \
					  [Line2D([0], [0], marker='o', color='black', label='MLR-DMPC', linestyle='None',
							  markerfacecolor='black')] + [Line2D([0], [0], marker='x', color='black', label='DMPC', linestyle='None',
							  markerfacecolor='black')]

	ax.legend(handles=legend_elements, loc='right')
	ax.set_ylim(-0.1, 20)
	#plt.xticks(nums_cus, xlabels)

	fig = plt.figure()

	# Creating axes instance
	ax = plt.axes()

	plot_arrival_rate(ax, num_targets_reached=num_targets_reached_mlrdmpc, num_experiments=num_experiments_mlrdmpc,
					nums_drones=nums_drones, nums_cus=nums_cus, marker="o")
	plot_arrival_rate(ax, num_targets_reached=num_targets_reached_dmpc, num_experiments=num_experiments_dmpc, nums_drones=nums_drones,
					nums_cus=nums_cus, marker="x")

	plt.ylabel("Arrival rate (%)")
	plt.xlabel("Number CUs")

	legend_elements = [Patch(facecolor=colors[i], edgecolor='k',
							 label=f"Number drones = {nums_drones[i]}") for i in range(len(nums_drones))] + \
					  [Line2D([0], [0], marker='o', color='black', label='MLR-DMPC', linestyle='None',
							  markerfacecolor='black')] + [Line2D([0], [0], marker='x', color='black', label='DMPC', linestyle='None',
							  markerfacecolor='black')]

	ax.legend(handles=legend_elements, loc='right')
	ax.set_ylim(-1, 101)

	#plt.xticks(nums_cus, xlabels)
	# show plot
	plt.show()

