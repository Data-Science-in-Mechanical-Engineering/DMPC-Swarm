import numpy as np
from abc import ABC, abstractmethod
import copy
import math


class SetpointCreator:
	def __init__(self, drones, testbeds, target_reached_dist=5e-2, max_setpoint_age=100):
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

		self.__round = 0

		self.__current_setpoints_reached = {drone_id: False for drone_id in drones}

		self.__current_setpoints = {drone_id: self.generate_new_setpoint(drones[drone_id]) for drone_id in drones}
		print(self.__current_setpoints)
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

		# for the drones that have reached their setpoint calculate a new one
		for drone_id in self.__current_setpoints_reached:
			if self.__current_setpoints_reached[drone_id]:
				self.__current_setpoints[drone_id] = self.generate_new_setpoint(self.drones[drone_id])
				self.__current_setpoints_age[drone_id] = 0
				self.__current_setpoints_reached[drone_id] = False

			if self.__drones[drone_id] == "Mobile" or True:
				self.__current_setpoints[drone_id] = self.generate_new_circle_setpoint(self.drones[drone_id], drone_id)
			self.__current_setpoints_age[drone_id] += 1
			print(f"{drone_id}: {self.__current_setpoints_age[drone_id]}, {self.__current_setpoints_reached[drone_id]}")

		return self.__current_setpoints

	def generate_new_circle_setpoint(self, name_testbed, drone_id):
		min_pos = np.array(self.__testbeds[name_testbed][0])
		max_pos = np.array(self.__testbeds[name_testbed][1])
		dpos = (max_pos - min_pos) * 0.8
		mean = (min_pos + max_pos) / 2
		r = 0.5
		angle = 2*math.pi * (self.__round + drone_id*5) / 40
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


