import numpy as np
from abc import ABC, abstractmethod
import copy
import math

CIRCLE = 0
DEMO = 1
CIRCLE_DYNAMIC = 2
MESSAGE_LOSS_CRASH = 3
DEMO_CIRCLE = 4
CIRCLE_COMPARE = 5
RANDOM = 6


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

		self.__old_setpoints = copy.deepcopy(self.__current_setpoints)

		self.__random_setpoints = {}
		self.__random_setpoints_calculated = {}

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
		if round_nmbr != self.__round:
			self.__old_setpoints = copy.deepcopy(self.__current_setpoints)
		self.__round = round_nmbr
		self.check_current_setpoints_reached(drones_states)

		for drone_id in self.__drones:
			if self.__demo_setpoints == CIRCLE:
				self.__current_setpoints[drone_id] = self.generate_new_circle_setpoint(drone_id)
			elif self.__demo_setpoints == CIRCLE_COMPARE:
				self.__current_setpoints[drone_id] = self.generate_new_circle_compare_setpoint(drone_id)
			elif self.__demo_setpoints == DEMO:
				self.__current_setpoints[drone_id] = self.generate_new_demo_setpoint(drone_id)
			elif self.__demo_setpoints == CIRCLE_DYNAMIC:
				self.__current_setpoints[drone_id] = self.generate_new_dynamic_circle_setpoint(drone_id)
			elif self.__demo_setpoints == MESSAGE_LOSS_CRASH:
				self.__current_setpoints[drone_id] = self.generate_new_message_loss_crash_setpoint(drone_id)
			elif self.__demo_setpoints == DEMO_CIRCLE:
				self.__current_setpoints[drone_id] = self.generate_new_demo_circle_setpoint(drone_id)
			elif self.__demo_setpoints == RANDOM:
				self.__current_setpoints[drone_id] = self.generate_new_random_setpoint(drone_id)

		setpoints_changed = False
		for k in self.__current_setpoints:
			if np.linalg.norm(self.__current_setpoints[k] - self.__old_setpoints[k]) > 1e-5:
				setpoints_changed = True

		return self.__current_setpoints, setpoints_changed

	def generate_new_circle_setpoint(self, drone_id):
		if drone_id == 10:
			drone_id = 2
		name_testbed = self.__drones[drone_id]
		angle_offset = 0
		if self.__round%200 >= 100:
			angle_offset = math.pi

		offset = np.array(self.__testbeds[name_testbed][2])
		angle = 2 * math.pi * drone_id / 6 + angle_offset
		dpos = [1.5, 1.5, 1.5]
		return np.array([dpos[0] * math.cos(angle), dpos[1] * math.sin(angle), 0.8]) + offset

	def generate_new_circle_compare_setpoint(self, drone_id):
		if drone_id > 6:
			return np.array([0.0, 0.0, 0.0])
		if self.__round >= 200:
			return self.generate_new_circle_setpoint(drone_id)
		if self.__round <= 100:
			return self.generate_new_demo_setpoint(drone_id)
		targets = {1: [-1, 1, 1], 2: [0, 1, 1], 3: [1, 1, 1],
				   4: [-1, -1, 1], 5: [0, -1, 1], 6: [1, -1, 1]}
		return np.array(targets[drone_id], dtype=np.float32)

	def generate_new_demo_setpoint(self, drone_id):
		if drone_id < 7:
			if self.__round % 300 <= 150:
				targets = {1: [-1.3, 0, 1], 2: [1.3, 0, 1], 3: [0, -1.0, 1.0], 4: [0, -0.21, 1.0], 5: [0, 0.21, 1], 6: [0, 1.0, 1],
						   # 7: [0.23, -1.3, 1], 8: [0.23, 1.3, 1]
						   }
				return np.array(targets[drone_id])
			else:
				targets = {1: [1.3, 0, 1], 2: [-1.3, 0, 1], 3: [0, -1.0, 1.0], 4: [0, -0.21, 1.0], 5: [0, 0.21, 1], 6: [0, 1.0, 1],
						   # 7: [0.23, 1.3, 1], 8: [0.23, -1.3, 1]
						   }
				return np.array(targets[drone_id])
		else:
			return self.generate_new_circle_setpoint(drone_id)

	def generate_new_dynamic_circle_setpoint(self, di):

		drone_id = di if di != 10 else 2
		name_testbed = self.__drones[drone_id]
		min_pos = np.array(self.__testbeds[name_testbed][0])
		max_pos = np.array(self.__testbeds[name_testbed][1])
		offset = np.array(self.__testbeds[name_testbed][2])
		dpos = (max_pos - min_pos) / 2 * 0.8
		dpos = [1.4, 1.4] if drone_id != 7 else [0.8, 0.8]
		mean = (min_pos + max_pos) / 2 + offset
		if name_testbed == "Vicon":
<<<<<<< HEAD

			angle = 2 * math.pi * (self.__round) / 70.0 + 2 * math.pi * drone_id / 10 + math.pi
			mean[2] = 1.0 + drone_id*0.1 if drone_id != 13 else 1.0
			mean[1] += 0.2
			mean[0] += 0.2
=======
			angle = 2 * math.pi * (self.__round) / 65.0 + 2 * math.pi * drone_id / 6 + math.pi
			mean[2] = 0.7 #+ drone_id*0.1 if drone_id != 13 else 1.0
			mean[1] -= 0.0
			mean[0] -= 0.0
>>>>>>> bd4635722fb6db6d4a0132d00df1beadace0374c
		else:
			angle = 2 * math.pi * (self.__round) / 65.0 + 2 * math.pi * drone_id / 2 + math.pi
			mean[2] = 0.6
		if self.__round >= 400 and drone_id == 7:
			return np.array([1.7, 1.0, 1.0])
		return np.array([dpos[0] * math.cos(angle), dpos[1] * math.sin(angle), 0]) + mean

	def generate_new_demo_circle_setpoint(self, drone_id):
		if (drone_id != 3 and drone_id != 1) or self.__round <= 400:
			name_testbed = self.__drones[drone_id]
			min_pos = np.array(self.__testbeds[name_testbed][0])
			max_pos = np.array(self.__testbeds[name_testbed][1])
			offset = np.array(self.__testbeds[name_testbed][2])
			dpos = [1.3, 1.3, 1.3]
			mean = (min_pos + max_pos) / 2 + offset
			if name_testbed == "Vicon":
				angle = 2 * math.pi * (self.__round) / 50.0 + 2 * math.pi * drone_id / 3 + math.pi
				mean[2] = 1.5
			else:
				angle = 2 * math.pi * (self.__round) / 50.0 + 2 * math.pi * drone_id / 2 + math.pi
			return np.array([dpos[0] * math.cos(angle), dpos[1] * math.sin(angle), 0]) + mean + np.array([0.2, 0.2, +0.2*drone_id])
		else:
			if drone_id == 1:
				return np.array([1.5, -1.1, 1.2])
			if drone_id == 3:
				return np.array([1.1, -1.5, 1.2])

	def generate_new_message_loss_crash_setpoint(self, drone_id):
		if drone_id > 2:
			return np.array([0.0, 0.0, 0.0])
		if self.__round <= 150:
			targets = {1: [-1.2, 0.0, 1], 2: [1.2, 0, 1]}
		else:
			targets = {1: [1.2, 0.0, 1], 2: [-1.2, 0, 1]}
		return np.array(targets[drone_id])

	def generate_new_random_setpoint(self, drone_id):
		if drone_id not in self.__random_setpoints_calculated:
			self.__random_setpoints_calculated[drone_id] = False
		if (self.__round % 25 == 0 and not self.__random_setpoints_calculated[drone_id]) or drone_id not in self.__random_setpoints:
			name_testbed = self.__drones[drone_id]
			min_pos = np.array(self.__testbeds[name_testbed][0]) * 0.8
			max_pos = np.array(self.__testbeds[name_testbed][1]) * 0.8
			offset = np.array(self.__testbeds[name_testbed][2])
			self.__random_setpoints_calculated[drone_id] = True
			self.__random_setpoints[drone_id] = (np.random.rand(3) * (max_pos - min_pos) + min_pos)
			if self.__random_setpoints[drone_id][2] < 0.8:
				self.__random_setpoints[drone_id][2] = 0.8

			self.__random_setpoints[drone_id] += offset
		elif self.__round % 25 != 0:
			self.__random_setpoints_calculated[drone_id] = False

		return self.__random_setpoints[drone_id]



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


