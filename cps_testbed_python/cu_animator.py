import threading
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import numpy as np


def animate(data_pipeline):
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	print("Hallooooooooooooooooooooooooooooooooooooooo")
	plt.show()
	while True:
		print("Hallo")
		trajs = data_pipeline.get_trajectories()
		if trajs is not None:
			ax1.clear()
			for t in trajs:
				ax1.plot(trajs[1], trajs[2])

			ax1.set_xlim(left=-4, right=4)
			ax1.set_ylim(left=-4, right=4)


class DataPipeline:
	def __init__(self):
		self.__data = None
		self.__trajectories = None

	def set_trajectories(self, trajectories, target_positions, intermediate_target_positions):
		with open(
				f'../../experiment_measurements/animations.p',
				'wb') as handle:
			pickle.dump({"trajectories": trajectories, "target_positions": target_positions,
						 "intermediate_target_positions": intermediate_target_positions}, handle)

	def get_trajectories(self):
		trajectories = None
		while trajectories is None:
			try:
				with open(
						f'../../experiment_measurements/animations.p',
						'rb') as handle:
					trajectories = pickle.load(handle)
			except Exception as e:
				print(e)
		return trajectories


if __name__ == "__main__":
	dp = DataPipeline()

	fig, ax = plt.subplots()
	xdata, ydata = [], []
	ln, = ax.plot([], [], 'ro')


	def update(i):
		data = dp.get_trajectories()
		colors = ["b", "k", "g", "m", "r", "c"]

		ax.clear()

		trajectories = data["trajectories"]
		for i, key in enumerate(trajectories):
			t = trajectories[key]
			ax.plot(t[:, 0], t[:, 1], color=colors[i%len(colors)])
			ax.scatter([t[0, 0]], [t[0, 1]], color=colors[i%len(colors)])

		target_positions = data["target_positions"]
		if target_positions is not None:
			for i, t in enumerate(target_positions):
				if target_positions[t] is not None:
					ax.scatter(target_positions[t][0], target_positions[t][1], s=10**2, color=colors[i % len(colors)], marker="X")

		target_positions = data["intermediate_target_positions"]
		if target_positions is not None:
			for i, t in enumerate(target_positions):
				if target_positions[t] is not None:
					ax.scatter(target_positions[t][0], target_positions[t][1], s=10 ** 2, color=colors[i % len(colors)],
							   marker="*")

		ax.axis('equal')
		ax.set_xlim(left=-4, right=4)
		ax.set_ylim(bottom=-4, top=4)

	ani = FuncAnimation(fig, update, interval=200)
	plt.show()
