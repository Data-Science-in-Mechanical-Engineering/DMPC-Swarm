import os
import pickle as p
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
if __name__ == "__main__":
	path = "C:\\Users\\mf724021\\Documents\\003_Testbed\\007_Code\\CPSTestbed\\simulation\\CPSTestbed\\" \
		   "batch_simulation_results\\dmpc\\dmpc_simulation_results_not_ignore_message_loss_demo1"
	path_simulation = path + "\\simulation_result-6_drones_simnr_1.pkl"
	path_real_world = path + "\\Experiment100-25.pickle"
		   #"batch_simulation_results\\dmpc\\dmpc_simulation_results_not_ignore_message_loss_005\\"
	#path = str(Path('~').expanduser()) + "/Documents/AntiCollision/cpstestbed/simulation/CPSTestbed/batch_simulation_results/" \
#										  "dmpc/dmpc_simulation_results_not_ignore_message_loss_no_hlp_001"
	plot_states = False
	num_drones = 6
	colors = ['red', 'blue', 'green', 'black', 'magenta', 'purple']
	INIT_XYZS = np.array([
		[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [-1.0, -1.0, 1.0],
		[1.0, -1.0, 1.0]])
	INIT_TARGETS = np.array([[1.0, 1.0, 1.0], [-1.0, 1.05, 1.0], [1.0, 0.0, 1.0], [-1.0, 0.05, 1.0], [1.0, -1.0, 1.0],
							 [-1.0, -1.05, 1.0]
							 ])
	result_simulation = p.load(open(path_simulation, "rb"))
	fig = plt.figure()
	ax = plt.axes()
	"""projection='2d'"""

	for j in range(num_drones):
		states = result_simulation[f"state_set{j}"]
		xline = []
		yline = []
		zline = []
		for s in states:
			a = 2
			s_ = s - np.array([a, a, 0, 0, 0, 0, 0, 0, 0])
			xline.append(s_[0])
			yline.append(s_[1])
			#zline.append(s_[2])
		#ax.plot3D(xline, yline, zline, color=colors[j])
		ax.plot(xline, yline, color=colors[j])

	for j in range(num_drones):
		states = result_simulation[f"state_{j}"]
		xline = []
		yline = []
		zline = []
		for s in states:
			a = 2
			s_ = s - np.array([a, a, 0, 0, 0, 0, 0, 0, 0])
			xline.append(s_[0])
			yline.append(s_[1])
			#zline.append(s_[2])
		ax.plot(xline, yline, ":", color=colors[j])
		ax.scatter(INIT_XYZS[j][0], INIT_XYZS[j][1], color=colors[j], s=40)
		ax.scatter(INIT_TARGETS[j][0], INIT_TARGETS[j][1], color=colors[j], marker="x", s=100)

	result_real_world = p.load(open(path_real_world, "rb"))
	for j in range(num_drones):
		states = result_real_world[j]
		xline = []
		yline = []
		zline = []
		for s in states[500:1500]:
			xline.append(s[0])
			yline.append(s[1])
			#zline.append(s[2])
			#print(s[2])
		ax.plot(xline, yline, '--', color=colors[j])

	linestyles = ["-", ":", "--"]
	legend_elements = [Line2D([0], [0], linestyle=linestyles[i],  color='black') for i in range(3)] + [Line2D([0], [0], marker='o', color='black', label='Scatter', linestyle='None',
                          markerfacecolor='black')] + [Line2D([0], [0], marker='x', color='black', label='Scatter', linestyle='None',
                          markerfacecolor='black')]

	ax.legend(legend_elements, ["Setpoints", "Simulation", "Measurement", "Start", "Target"], loc='best')
	ax.set_aspect(1)
	ax.set_xlim(-1.3, 2.5)
	plt.xlabel("x (m)")
	plt.ylabel("y (m)")

	fig = plt.figure()
	ax = plt.axes()
	for j in range(num_drones):
		states = result_real_world[j]
		xline = []
		yline = []
		zline = []
		for s in states[1100:1500]:
			xline.append(s[0])
			yline.append(s[1])
			zline.append(s[2])
		ax.plot(zline, color=colors[j])
		print(f"[{xline[0]}, {yline[0]}, 1],")

	plt.show()


