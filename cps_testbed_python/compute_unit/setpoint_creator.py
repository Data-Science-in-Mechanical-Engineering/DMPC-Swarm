import numpy as np
from abc import ABC, abstractmethod
import copy
import math

CIRCLE = 0
DEMO = 1


class SetpointCreator:
	def __init__(self, drones, testbeds, target_reached_dist=5e-2, max_setpoint_age=100, demo_setpoints=CIRCLE):
		"""

		Args:
			drones (Dict): Dictionary containing the id as keys and the type of the testbed as content. The id are all drones. If they are started or not
			testbeds (Dict): Dictionary containing the type of the testbed as keys and the the min, max and offset as tuple.
			target_reached_dist (float): distance after which a drone is considered to have reached its target.
			max_setpoint_age (float): maximum age of the setpoints, after which the creator proposes a new one.
		"""
		self.__drones = drones
		self.__testbeds = testbeds
		self.__target_reached_dist = target_reached_dist
		self.__max_setpoint_age = max_setpoint_age
		self.__demo_setpoints = demo_setpoints

		self.__round = 0

		self.__current_setpoints_reached = {drone_id: False for drone_id in drones}

		self.__current_setpoints = {drone_id: self.generate_new_setpoint(drones[drone_id]) for drone_id in drones}
		self.__current_setpoints_age = {drone_id: 0 for drone_id in drones}

	@property
	def drones(self):
		return self.__drones

	@property
	def testbeds(self):
		return self.__testbeds

	def check_current_setpoints_reached(self, drones_states):
		"""

		Returns:

		"""
		for drone_id in self.__current_setpoints:
			# self.__current_setpoints_reached[drone_id] = False
			# check if the crone is in the testbed and we know where it is
			if self.__current_setpoints_age[drone_id] > self.__max_setpoint_age:
				self.__current_setpoints_reached[drone_id] = True
			"""if drone_id in drones_states and self.__current_setpoints[drone_id] is not None:
				if drones_states[drone_id] is not None:
					# if np.linalg.norm(drones_states[drone_id][0:3] - self.__current_setpoints[drone_id]) < self.__target_reached_dist \
					if self.__current_setpoints_age[drone_id] > self.__max_setpoint_age:
						self.__current_setpoints_reached[drone_id] = True"""

	def next_setpoints(self, drones_states, round_nmbr):
		"""

		Args:
			drones_states (Dict): Dictionary containing the states of the drones.

		Returns:

		"""
		self.__round = round_nmbr
		self.check_current_setpoints_reached(drones_states)
		old_setpoints = copy.deepcopy(self.__current_setpoints)

		# for the drones that have reached their setpoint calculate a new one
		for drone_id in self.__current_setpoints_reached:
			if self.__current_setpoints_reached[drone_id]:
				self.__current_setpoints[drone_id] = self.generate_new_setpoint(self.drones[drone_id])
				self.__current_setpoints_age[drone_id] = 0
				self.__current_setpoints_reached[drone_id] = False

			if self.__drones[drone_id] == "Mobile" or True:
				self.__current_setpoints[drone_id] = self.generate_new_circle_setpoint(self.drones[drone_id], drone_id)
			self.__current_setpoints_age[drone_id] += 1

		setpoints_changed = False
		for k in self.__current_setpoints:
			if np.linalg.norm(self.__current_setpoints[k] - old_setpoints[k]) > 1e-5:
				setpoints_changed = True

		return self.__current_setpoints, setpoints_changed

	def generate_new_circle_setpoint(self, name_testbed, drone_id):
		if self.__demo_setpoints == CIRCLE:
			angle_offset = 0
			if self.__round >= 180:
				angle_offset = math.pi

			min_pos = np.array(self.__testbeds[name_testbed][0])
			max_pos = np.array(self.__testbeds[name_testbed][1])
			offset = np.array(self.__testbeds[name_testbed][2])
			dpos = (max_pos - min_pos) * 0.8
			mean = (min_pos + max_pos) / 2 + offset
			angle = 2 * math.pi * drone_id / 6 + angle_offset
			dpos = [1.5, 1.5, 1.5]
			return np.array([dpos[0] * math.cos(angle), dpos[1] * math.sin(angle), 1])
		elif self.__demo_setpoints == DEMO:
			if self.__round%300 >= 150:
				targets = {1: [0, -1.3, 1], 2: [0, 1.3, 1], 3: [-1, 0, 1], 4: [-0.3, 0, 1], 5: [0.3, 0, 1], 6: [1.0, 0, 1]}
				return np.array(targets[drone_id])
			else:
				targets = {1: [0, 1.3, 1], 2: [0, -1.3, 1], 3: [-1, 0, 1], 4: [-0.3, 0, 1], 5: [0.3, 0, 1],
						   6: [1.0, 0, 1]}
				return np.array(targets[drone_id])


		mult = 1
		if name_testbed == "Vicon" and not (drone_id == 1 or drone_id == 1000):
			a = 0.5
			const_targets = {1: [-1 - a, 1 + a, 1], 2: [0, 1 + a, 1], 3: [1 + a, 1 + a, 1],
							 5: [-1 - a, 0, 1], 6: [0, 0, 1], 7: [1 + a, 0, 1],
							 8: [-1 - a, -1 - a, 1], 9: [0, -1 - a, 1], 10: [1 + a, -1 - a, 1]}
			return np.array(const_targets[drone_id])

		"""if (self.__round+50) % 100 <= 50:
			if drone_id == 1:
				return np.array([1.1, -1, 1])
			else:
				return np.array([-0.9, 1, 1])
		else:
			if drone_id == 1:
				return np.array([-1.1, 1, 1])
			else:
				return np.array([0.9, -1, 1])"""

		min_pos = np.array(self.__testbeds[name_testbed][0])
		max_pos = np.array(self.__testbeds[name_testbed][1])
		offset = np.array(self.__testbeds[name_testbed][2])
		dpos = (max_pos - min_pos) * 0.8
		mean = (min_pos + max_pos) / 2 + offset
		angle = 2*math.pi * (self.__round * mult) / 30 + 2*math.pi*drone_id / 6 + math.pi
		dpos = [0.2, 0.2, 0.2]
		return np.array([dpos[0]*math.cos(angle), dpos[1]*math.sin(angle), 0]) + mean

	def generate_new_setpoint(self, name_testbed):
		"""

		Args:
			name_testbed (str): name of the testbed

		Returns:
			setpoint (Array): a random setpoint in the testbed (in the local coordinate system of the testbed)
		"""
		min_pos = np.array(self.__testbeds[name_testbed][0])
		max_pos = np.array(self.__testbeds[name_testbed][1])
		#TODO, currently in the global coordinate system change in the future (we need this to send them)
		return np.random.rand(3) * (max_pos - min_pos - 0.2) + min_pos + 0.1 + np.array(self.__testbeds[name_testbed][2])

	def add_drone(self, drone_id, state):
		"""

		Args:
			drone_id:  id of the drone to be added
			state:  state of the drone

		Returns:

		"""
		assert drone_id in self.__drones, f"Drone id {drone_id} not in the registered drones."

		self.__current_setpoints[drone_id] = self.generate_new_setpoint(self.__drones[drone_id])


