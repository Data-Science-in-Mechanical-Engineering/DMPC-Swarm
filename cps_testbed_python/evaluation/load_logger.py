import os
import pickle as p
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import pandas as pd


def print_success_statistics(num_drones, message_loss, num_cus, quant):
	path = str(Path('~').expanduser()) + f"/Documents/batch_simulation_results/" \
		f"dmpc/dmpc_simulation_results_ignore_message_loss_{int(round(100*message_loss))}_{num_cus}cus_{'' if not quant else 'quant'}"

	plot_states = False
	files = [os.path.join(path, f) for f in os.listdir(path) if
			 f.startswith(f"simulation_result-{num_drones}_drones_simnr_")]
	num_crashed = 0
	num_optimizer_runs = 0
	num_succ_optimizer_runs = 0
	num_targets_reached = 0
	for f in files:
		result = p.load(open(f, "rb"))
		if result["crashed"][0]:
			num_crashed += 1
			fig = plt.figure()
			ax = plt.axes(projection='3d')
			p1 = None
			p2 = None
			for j in range(num_drones):
				for k in range(num_drones):
					if j == k:
						continue
					if np.linalg.norm(np.diag([1, 1, 1 / 4]) @ (
							result[f"state_{j}"][-1][0:3] - result[f"state_{k}"][-1][0:3])) < 0.3:
						p1 = result[f"state_{j}"][-1][0:3]
						p2 = result[f"state_{k}"][-1][0:3]
						print(result[f"state_{j}"][-1][0:3])
						print(result[f"state_{k}"][-1][0:3])
						print(np.linalg.norm(
							np.diag([1, 1, 1 / 4]) @ (result[f"state_{j}"][-1][0:3] - result[f"state_{k}"][-1][0:3])))

			for j in range(num_drones):
				states = result[f"state_{j}"]
				xline = []
				yline = []
				zline = []
				for s in states:
					xline.append(s[0])
					yline.append(s[1])
					zline.append(s[2])
				ax.plot3D(xline, yline, zline)
			print(p1)
			print(p2)
			ax.scatter([p1[0]], [p1[1]], [p1[2]])
			ax.scatter([p2[0]], [p2[1]], [p2[2]])

		if result["num_targets_reached"][0] != num_drones:
			print(result["num_targets_reached"][0])
			# fig = plt.figure()
			# ax = plt.axes(projection='3d')
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
	plt.show()
	print(f"Crashed percentage: {num_crashed / len(files) * 100}% of {len(files)} experiments")
	print(f"num_optimizer_runs: {num_optimizer_runs}")
	print(f"Infeasibility percentage: {(num_optimizer_runs - num_succ_optimizer_runs) / num_optimizer_runs * 100}%")
	print(f"num_targets_reached percentage {num_targets_reached / (num_drones * len(files)) * 100}%")


def plot_comparison(num_drones, message_loss, ignore_message_loss, quant):

	target_reached_times = []

	for num_cus in range(1, num_drones+1, 2):
		path = "../../../batch_simulation_results/" \
				f"dmpc/dmpc_simulation_results_ignore_message_loss_{int(round(100 * message_loss))}_{num_cus}cus"
		# f"dmpc/dmpc_simulation_results_iml{ignore_message_loss}_{int(round(100 * message_loss))}_{num_cus}cus_{'' if not quant else 'quant'}"
		print(path)
		files = [os.path.join(path, f) for f in os.listdir(path) if
				 f.startswith(f"simulation_result-{num_drones}_drones_simnr_")]

		target_reached_time = []
		print(files)
		for f in files:
			result = p.load(open(f, "rb"))
			if result["crashed"][0]:
				continue
			target_reached_time.append(result["target_reached_time"][0])

		target_reached_times.append(target_reached_time)

	target_reached_times_np = np.array(target_reached_times)
	print(np.max(target_reached_times_np, axis=1))

	boxplot_data = {}
	boxplot_data["index"] = np.array([i for i in range(1, num_drones+1, 2)])
	boxplot_data["median"] = np.median(target_reached_times_np, axis=1)
	boxplot_data["box_top"] = np.percentile(target_reached_times_np, q=75, axis=1)
	boxplot_data["box_bottom"] = np.percentile(target_reached_times_np, q=25, axis=1)
	boxplot_data["whisker_top"] = np.percentile(target_reached_times_np, q=95, axis=1)
	boxplot_data["whisker_bottom"] = np.percentile(target_reached_times_np, q=5, axis=1)

	df = pd.DataFrame(data=boxplot_data)
	df.to_csv(
		f"../../../experiment_measurements/UAVs{num_drones}Boxplot.csv",
		sep=" ", header=False, index=False)

	plt.figure()
	plt.boxplot(target_reached_times)
	plt.show()


if __name__ == "__main__":
	# plot_comparison(10, 0, ignore_message_loss=False, quant=False)

	ignore_message_loss = False
	message_loss_prob = 0.01
	num_cus = 3
	simulate_quantization = True
	path = os.path.dirname(os.path.abspath(__file__)) + "/../../../batch_simulation_results/dmpc/" \
		   + f"dmpc_simulation_results_iml{ignore_message_loss}_{int(100 * message_loss_prob + 1e-7)}_{num_cus}cus_{'quant' if simulate_quantization else ''}"

	plot_states = False
	num_drones = 15
	files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith(f"simulation_result-{num_drones}_drones_simnr_")]
	num_crashed = 0
	num_optimizer_runs = 0
	num_succ_optimizer_runs = 0
	num_targets_reached = 0
	for f in files:
		result = p.load(open(f, "rb"))
		if result["crashed"][0]:
			num_crashed += 1
			"""fig = plt.figure()
			ax = plt.axes(projection='3d')
			p1 = None
			p2 = None
			for j in range(num_drones):
				for k in range(num_drones):
					if j == k:
						continue
					if np.linalg.norm(np.diag([1, 1, 1/4])@(result[f"state_{j}"][-1][0:3] - result[f"state_{k}"][-1][0:3])) < 0.3:
						p1 = result[f"state_{j}"][-1][0:3]
						p2 = result[f"state_{k}"][-1][0:3]
						print(result[f"state_{j}"][-1][0:3])
						print(result[f"state_{k}"][-1][0:3])
						print(np.linalg.norm(np.diag([1, 1, 1/4])@(result[f"state_{j}"][-1][0:3] - result[f"state_{k}"][-1][0:3])))

			for j in range(num_drones):
				states = result[f"state_{j}"]
				xline = []
				yline = []
				zline = []
				for s in states:
					xline.append(s[0])
					yline.append(s[1])
					zline.append(s[2])
				ax.plot3D(xline, yline, zline)
			print(p1)
			print(p2)
			ax.scatter([p1[0]], [p1[1]], [p1[2]])
			ax.scatter([p2[0]], [p2[1]], [p2[2]])"""

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
	plt.show()
	print(f"num_drones {num_drones}")
	print(f"message_loss_prob {message_loss_prob*100}%")
	print(f"Crashed percentage: {num_crashed / len(files) * 100}% of {len(files)} experiments")
	print(f"num_optimizer_runs: {num_optimizer_runs}")
	print(f"Infeasibility percentage: {(num_optimizer_runs - num_succ_optimizer_runs) / num_optimizer_runs * 100}%")
	print(f"num_targets_reached percentage {num_targets_reached/(num_drones*len(files))*100}%")


