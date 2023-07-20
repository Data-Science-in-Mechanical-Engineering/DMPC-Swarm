import os
import pickle as p
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
if __name__ == "__main__":
	path = "C:\\Users\\mf724021\\Documents\\003_Testbed\\007_Code\\batch_simulation_results\\dmpc\\dmpc_simulation_results_ignore_message_loss_000_2cus"
		   #"batch_simulation_results\\dmpc\\dmpc_simulation_results_not_ignore_message_loss_005\\"
	path = str(Path('~').expanduser()) + "/Documents/batch_simulation_results/" \
										  "dmpc/dmpc_simulation_results_ignore_message_loss_10_2cus"
	plot_states = False
	num_drones = 10
	files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith(f"simulation_result-{num_drones}_drones_simnr_")]
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
			ax.scatter([p2[0]], [p2[1]], [p2[2]])

		if result["num_targets_reached"][0] != num_drones:
			print(result["num_targets_reached"][0])
			#fig = plt.figure()
			#ax = plt.axes(projection='3d')
			for j in range(num_drones):
				states = result[f"state_{j}"]
				xline = []
				yline = []
				zline = []
				for s in states:
					xline.append(s[0])
					yline.append(s[1])
					zline.append(s[2])
				#ax.plot3D(xline, yline, zline)


		num_optimizer_runs += result["num_optimizer_runs"][0]
		num_succ_optimizer_runs += result["num_succ_optimizer_runs"][0]
		num_targets_reached += result["num_targets_reached"][0]
	plt.show()
	print(f"Crashed percentage: {num_crashed / len(files) * 100}% of {len(files)} experiments")
	print(f"num_optimizer_runs: {num_optimizer_runs}")
	print(f"Infeasibility percentage: {(num_optimizer_runs - num_succ_optimizer_runs) / num_optimizer_runs * 100}%")
	print(f"num_targets_reached percentage {num_targets_reached/(num_drones*len(files))*100}%")


