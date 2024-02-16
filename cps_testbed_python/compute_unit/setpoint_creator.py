import numpy as np
from abc import ABC, abstractmethod
import copy
import math

BASIS_HEIGHT = 1.2

CIRCLE = 0
DEMO = 1
CIRCLE_DYNAMIC = 2
MESSAGE_LOSS_CRASH = 3
DEMO_CIRCLE = 4
CIRCLE_COMPARE = 5
RANDOM = 6
DYNAMIC_SWARM = 7
MULTI_HOP = 8
DEMO_AI_WEEK = 9
CIRCLE_PERIODIC = 10
CIRCLE_PYRAMID = 11
PHOTO = 12
DEMO_CRASH = 14
CIRCLE_PERIODIC_SMALL = 15

DEMO_AI_WEEK_IDLE = 0
DEMO_AI_WEEK_CIRCLE = 1
DEMO_AI_WEEK_CIRCLE2 = 2  # like DEMO_AI_WEEK_CIRCLE, just +pi on angle
DEMO_AI_WEEK_D = 3
DEMO_AI_WEEK_S = 4
DEMO_AI_WEEK_M = 5
DEMO_AI_WEEK_E = 6
DEMO_AI_WEEK_GO_BACK = 7

def get_circle_point(radius, idx, num_drones, z, angle_offset=0):
	angle = 2 * math.pi / num_drones * idx + angle_offset
	return np.array([np.sin(angle)*radius, np.cos(angle)*radius, z])

def get_pos_D(drone_id):
	if drone_id > 7:
		return np.array([0, 0, 0])
	pos = np.array([[-1.0, 1.0, 0.7], [-0.1, 0.75, 0.7], [0.2, 0.0, 0.7], [-0.1, -0.75, 0.7], [-1.0, -1.0, 0.7], [-1.0, -0.33, 0.7], [-1.0, 0.33, 0.7]])
	return turn_plane(pos[drone_id-1]) + np.array([0.5, 0.0, 0.0])


def get_pos_S(drone_id):
	if drone_id > 7:
		return np.array([0, 0, 0])
	pos = np.array([[0.25, 1.0, 0.7], [-0.25, 0.75, 0.7], [-0.25, 0.25, 0.7], [0.0, 0.0, 0.7], [0.25, -0.25, 0.7], [0.25, -0.75, 0.7], [-0.25, -1.0, 0.7]])
	return turn_plane(pos[drone_id-1])


def get_pos_M(drone_id):
	if drone_id > 7:
		return np.array([0, 0, 0])
	pos = np.array([[-1.0, 1.0, 0.7], [-1.0, -1.0, 0.7], [-0.5, 0.5, 0.7], [0.0, 0.0, 0.7], [0.5, 0.5, 0.7], [1.0, 1.0, 0.7], [1.0, -1.0, 0.7]])
	return turn_plane(pos[drone_id-1])


def get_pos_E(drone_id):
	if drone_id > 7:
		return np.array([0, 0, 0])
	pos = np.array([[-1.0, 1.0, 0.7], [-1.0, 0.33, 0.7], [-1.0, -0.33, 0.7], [-1.0, -1.0, 0.7], [0.0, 1.0, 0.7], [0.0, 0.0, 0.7], [0.0, -1.0, 0.7]])
	return turn_plane(pos[drone_id-1]) + np.array([0.5, 0.0, 0.0])


def turn_plane(pos):
	pos = copy.deepcopy(pos)
	point1 = [0, -1.0, BASIS_HEIGHT]
	point2 = [0, 1.0, 2.5]

	m = (point2[2] - point1[2]) / (point2[1] - point1[1])

	pos[2] = m * (pos[1] - point1[1]) + point1[2]

	return pos


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

		self.__round = -1

		self.__current_setpoints_reached = {drone_id: False for drone_id in drones}

		self.__current_setpoints = {drone_id: self.generate_new_setpoint(drones[drone_id]) for drone_id in drones}
		self.__current_setpoints_age = {drone_id: 0 for drone_id in drones}

		self.__old_setpoints = copy.deepcopy(self.__current_setpoints)

		self.__random_setpoints = {}
		self.__random_setpoints_calculated = {}

		self.__starting_rounds = {}
		self.__angles = {}

		self.__circle_pyramid_idx = None

		self.__state_demo_ai_week = DEMO_AI_WEEK_IDLE

		np.random.seed(1000)

	@property
	def drones(self):
		return self.__drones

	@property
	def testbeds(self):
		return self.__testbeds

	def next_setpoints(self, round_nmbr):
		"""

		Args:
			drones_states (Dict): Dictionary containing the states of the drones.

		Returns:

		"""
		if round_nmbr != self.__round:
			self.__old_setpoints = copy.deepcopy(self.__current_setpoints)
		self.__round = round_nmbr

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
			elif self.__demo_setpoints == DYNAMIC_SWARM:
				self.__current_setpoints[drone_id] = self.generate_new_dynamic_swarm_setpoint(drone_id)
			elif self.__demo_setpoints == MULTI_HOP:
				self.__current_setpoints[drone_id] = self.generate_new_multi_hop_setpoint(drone_id)
			elif self.__demo_setpoints == DEMO_AI_WEEK:
				self.__current_setpoints[drone_id] = self.generate_new_demo_ai_week_setpoint(drone_id)
			elif self.__demo_setpoints == CIRCLE_PERIODIC:
				self.__current_setpoints[drone_id] = self.generate_circle_periodic_setpoint(drone_id)
			elif self.__demo_setpoints == CIRCLE_PERIODIC_SMALL:
				self.__current_setpoints[drone_id] = self.generate_circle_periodic_setpoint(drone_id, r=0.8)
			elif self.__demo_setpoints == CIRCLE_PYRAMID:
				self.__current_setpoints[drone_id] = self.generate_circle_pyramid_setpoint(drone_id)
			elif self.__demo_setpoints == PHOTO:
				self.__current_setpoints[drone_id] = self.generate_photo_setpoint(drone_id)
			elif self.__demo_setpoints == DEMO_CRASH:
				self.__current_setpoints[drone_id] = self.generate_demo_crash_setpoint(drone_id)

		setpoints_changed = False
		for k in self.__current_setpoints:
			if np.linalg.norm(self.__current_setpoints[k] - self.__old_setpoints[k]) > 1e-5:
				setpoints_changed = True

		return self.__current_setpoints, setpoints_changed

	def get_current_setpoints(self, drone_id):
		return self.__current_setpoints[drone_id]

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
			angle = 2 * math.pi * (self.__round) / 65.0 + 2 * math.pi * drone_id / 6 + math.pi
			mean[2] = 0.7 #+ drone_id*0.1 if drone_id != 13 else 1.0
			mean[1] -= 0.0
			mean[0] -= 0.0
		else:
			dpos = [0.5, 0.5]
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
		#if drone_id == 11 or drone_id == 12:
		#	return self.generate_new_dynamic_circle_setpoint(drone_id)

		if drone_id not in self.__random_setpoints_calculated:
			self.__random_setpoints_calculated[drone_id] = False
		if (self.__round % 50 == 0 and not self.__random_setpoints_calculated[drone_id]) or drone_id not in self.__random_setpoints:
			name_testbed = self.__drones[drone_id]
			min_pos = np.array(self.__testbeds[name_testbed][0]) * 0.8
			max_pos = np.array(self.__testbeds[name_testbed][1]) * 0.8
			offset = np.array(self.__testbeds[name_testbed][2])
			self.__random_setpoints_calculated[drone_id] = True
			self.__random_setpoints[drone_id] = (np.random.rand(3) * (max_pos - min_pos) + min_pos)
			if self.__random_setpoints[drone_id][2] < 0.8:
				self.__random_setpoints[drone_id][2] = 0.8

			self.__random_setpoints[drone_id] += offset
		elif self.__round % 50 != 0:
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

	def generate_new_dynamic_swarm_setpoint(self, drone_id):
		if drone_id in self.__starting_rounds:
			if self.__round - self.__starting_rounds[drone_id] > 310:
				return np.array([1.5, -1.6, 1.0])
			name_testbed = self.__drones[drone_id]
			min_pos = np.array(self.__testbeds[name_testbed][0])
			max_pos = np.array(self.__testbeds[name_testbed][1])
			offset = np.array(self.__testbeds[name_testbed][2])
			mean = np.array([0.2, 0.2, 0.8 + 0.05*drone_id])

			angle = 2 * math.pi * (self.__round) / 50.0 + 2 * math.pi * drone_id / 3 + math.pi
			return np.array([1.2 * math.cos(angle), 1.2 * math.sin(angle), 0.0]) + mean + offset

		return np.array([0.0, 0.0, 0.0])

	def generate_new_multi_hop_setpoint(self, drone_id):
		name_testbed = self.__drones[drone_id]
		min_pos = np.array(self.__testbeds[name_testbed][0])
		max_pos = np.array(self.__testbeds[name_testbed][1])
		offset = np.array(self.__testbeds[name_testbed][2])
		dpos = (max_pos - min_pos) / 2 * 0.8
		dpos = [1.5, 1.5]
		mean = (min_pos + max_pos) / 2
		if name_testbed == "Vicon" and drone_id in self.__angles:
			temp = 0 # if self.__round % 200
			angle = self.__angles[drone_id] + temp + 2 * math.pi * self.__round / 60.0
			mean[2] = 0.7 + 0.05*drone_id  # + drone_id*0.1 if drone_id != 13 else 1.0
		else:
			dpos = [0.5, 0.5]
			angle = 2 * math.pi * (self.__round) / 60.0 + 2 * math.pi * drone_id / 2 + math.pi
			mean[2] = 0.6
		return np.array([dpos[0] * math.cos(angle), dpos[1] * math.sin(angle), 0]) + mean + offset

	def generate_new_demo_ai_week_setpoint(self, drone_id):
		if drone_id not in self.__angles:
			return np.array([0, 0, 0])
		self.__state_demo_ai_week = DEMO_AI_WEEK_IDLE

		if self.__round > 150:
			self.__state_demo_ai_week = DEMO_AI_WEEK_CIRCLE
		if self.__round > 240:
			self.__state_demo_ai_week = DEMO_AI_WEEK_CIRCLE2
		if self.__round > 310:
			self.__state_demo_ai_week = DEMO_AI_WEEK_CIRCLE

		# show dsme.
		if self.__round > 380:
			self.__state_demo_ai_week = DEMO_AI_WEEK_D
		if self.__round > 440:
			self.__state_demo_ai_week = DEMO_AI_WEEK_S
		if self.__round > 500:
			self.__state_demo_ai_week = DEMO_AI_WEEK_M
		if self.__round > 560:
			self.__state_demo_ai_week = DEMO_AI_WEEK_E

		if self.__round > 620:
			self.__state_demo_ai_week = DEMO_AI_WEEK_GO_BACK

		if self.__state_demo_ai_week == DEMO_AI_WEEK_IDLE:
			if drone_id not in self.__angles:
				return np.array([0, 0, 0])
			name_testbed = self.__drones[drone_id]
			min_pos = np.array(self.__testbeds[name_testbed][0])
			max_pos = np.array(self.__testbeds[name_testbed][1])
			offset = np.array(self.__testbeds[name_testbed][2])
			dpos = [1.5, 1.5]
			mean = (min_pos + max_pos) / 2
			angle = self.__angles[drone_id] + 2 * math.pi * self.__round / 60.0
			mean[2] = BASIS_HEIGHT  # + 0.05 * drone_id  # + drone_id*0.1 if drone_id != 13 else 1.0
			return np.array([dpos[0] * math.cos(angle), dpos[1] * math.sin(angle), 0]) + mean + offset

		if self.__state_demo_ai_week == DEMO_AI_WEEK_CIRCLE:
			if drone_id not in self.__angles:
				return np.array([0, 0, 0])
			name_testbed = self.__drones[drone_id]
			min_pos = np.array(self.__testbeds[name_testbed][0])
			max_pos = np.array(self.__testbeds[name_testbed][1])
			offset = np.array(self.__testbeds[name_testbed][2])
			dpos = [1.5, 1.5]
			mean = (min_pos + max_pos) / 2
			angle = self.__angles[drone_id]
			mean[2] = BASIS_HEIGHT
			return np.array([dpos[0] * math.cos(angle), dpos[1] * math.sin(angle), 0]) + mean + offset

		if self.__state_demo_ai_week == DEMO_AI_WEEK_CIRCLE2:
			if drone_id not in self.__angles:
				return np.array([0, 0, 0])
			name_testbed = self.__drones[drone_id]
			min_pos = np.array(self.__testbeds[name_testbed][0])
			max_pos = np.array(self.__testbeds[name_testbed][1])
			offset = np.array(self.__testbeds[name_testbed][2])
			dpos = [1.5, 1.5]
			mean = (min_pos + max_pos) / 2
			angle = self.__angles[drone_id] + math.pi
			mean[2] = BASIS_HEIGHT
			return np.array([dpos[0] * math.cos(angle), dpos[1] * math.sin(angle), 0]) + mean + offset

		if self.__state_demo_ai_week == DEMO_AI_WEEK_CIRCLE2:
			name_testbed = self.__drones[drone_id]
			min_pos = np.array(self.__testbeds[name_testbed][0])
			max_pos = np.array(self.__testbeds[name_testbed][1])
			offset = np.array(self.__testbeds[name_testbed][2])
			dpos = [1.5, 1.5]
			mean = (min_pos + max_pos) / 2
			angle = self.__angles[drone_id] + math.pi
			mean[2] = BASIS_HEIGHT
			return np.array([dpos[0] * math.cos(angle), dpos[1] * math.sin(angle), 0]) + mean + offset

		if self.__state_demo_ai_week == DEMO_AI_WEEK_D:
			pos = get_pos_D(drone_id)
			return pos
		if self.__state_demo_ai_week == DEMO_AI_WEEK_S:
			pos = get_pos_S(drone_id)
			return pos
		if self.__state_demo_ai_week == DEMO_AI_WEEK_M:
			pos = get_pos_M(drone_id)
			return pos
		if self.__state_demo_ai_week == DEMO_AI_WEEK_E:
			pos = get_pos_E(drone_id)
			return pos

		if self.__state_demo_ai_week == DEMO_AI_WEEK_GO_BACK:
			back_pos = np.array([[-1.0, 1.0, 0.7], [0.0, 1.0, 0.7], [1.0, 1.0, 0.7],
								[-1.5, 0.0, 0.7], [-0.5, 0.0, 0.7], [0.5, 0.0, 0.7], [1.5, 0.0, 0.7]])
			return back_pos[drone_id - 1]

	def generate_circle_periodic_setpoint(self, drone_id, r=1.5):
		if drone_id not in self.__angles:
			return np.array([0, 0, 0])
		name_testbed = self.__drones[drone_id]
		min_pos = np.array(self.__testbeds[name_testbed][0])
		max_pos = np.array(self.__testbeds[name_testbed][1])
		offset = np.array(self.__testbeds[name_testbed][2])
		dpos = [r, r]
		mean = (min_pos + max_pos) / 2
		angle = self.__angles[drone_id]
		mean[2] = BASIS_HEIGHT
		if self.__round % 300 < 150:
			angle += math.pi #* 0.9
		return np.array([dpos[0] * math.cos(angle), dpos[1] * math.sin(angle), 0]) + mean + offset

	def generate_circle_pyramid_setpoint(self, drone_id):
		if self.__circle_pyramid_idx is None or self.__round % 500 == 499:
			idx = np.arange(len(self.__drones))
			np.random.shuffle(idx)
			self.__circle_pyramid_idx = {drone_id_: idx[i] for i, drone_id_ in enumerate(self.__drones)}

		num_lower_drones = 7
		num_middle1_drones = 5
		num_middle2_drones = 3
		if self.__circle_pyramid_idx[drone_id] < num_lower_drones:
			return get_circle_point(radius=1.6, num_drones=num_lower_drones, idx=self.__circle_pyramid_idx[drone_id],
									z=0.5)
		if self.__circle_pyramid_idx[drone_id] < num_lower_drones + num_middle1_drones:
			return get_circle_point(radius=1.0, num_drones=num_middle1_drones, idx=self.__circle_pyramid_idx[drone_id] - num_lower_drones,
									z=1.0, angle_offset=2*math.pi/num_lower_drones/2)
		if self.__circle_pyramid_idx[drone_id] < num_lower_drones + num_middle1_drones + num_middle2_drones:
			return get_circle_point(radius=0.4, num_drones=num_middle2_drones,
									idx=self.__circle_pyramid_idx[drone_id] - num_middle1_drones - num_lower_drones,
									z=1.5, angle_offset=2*math.pi/num_middle1_drones/2)
		return np.array([0, 0, 2.0])

	def generate_photo_setpoint(self, drone_id):
		if drone_id > 16:
			return np.array([0.0, 0.0, 0.0])

		offset = np.array([0.0, 0.0, 0.0])

		setpoints = np.array([[1.2, 0.0, 1.5],
							  [0.6, 0.0, 2.2],
							  [1.4, -0.3, 1.3],
							  [1.4, 0.3, 1.3],

							  [0.8, -0.4, 1.5],
							  [0.8, 0.4, 1.5],

							  [0.5, 0.5, 2.0],
							  [0.5, -0.5, 2.0],

							  [0.3, -0.7, 1.6],
							  [0.3, 0.7, 1.6],

							  [0.3, -1.5, 1.4],
							  [0.3, 1.5, 1.4],

							  [-0.3, -1.2, 2.0],
							  [-0.3, 1.2, 2.0],

							  [-0.6, 0.7, 1.9],
							  [-0.6, -0.7, 1.9],



							  ], dtype=np.float32)
		return setpoints[drone_id-1] + offset

	def generate_demo_crash_setpoint(self, drone_id):
		setpoints = np.array([[-0.5, 1.5, 1.0],
							  [0.5, 1.5, 1.0],

							  [-0.5, 1.0, 1.0],
							  [0.5, 1.0, 1.0],

							  [-0.5, 0.5, 1.0],
							  [0.5, 0.5, 1.0],

							  [-0.5, 0, 1.0],
							  [0.5, 0, 1.0],

							  [-0.5, -0.5, 1.0],
							  [0.5, -0.5, 1.0],

							  [-0.5, -1.0, 1.0],
							  [0.5, -1.0, 1.0],

							  [-0.5, -1.5, 1.0],
							  [0.5, -1.5, 1.0],

							  [0.0, 1.7, 1.0],
							  [0.0, -1.7, 1.0],

							  ])

		if drone_id > 16:
			return np.array([0.0, 0.0, 0.0])

		p = setpoints[drone_id-1]
		if self.__round % 300 >= 150:
			p[0] = -p[0]
			if drone_id >= 15:
				p[1] = -p[1]
		return p

	def add_drone(self, drone_id, state, round):
		"""

		Args:
			drone_id:  id of the drone to be added
			state:  state of the drone

		Returns:

		"""
		self.__round = round
		assert drone_id in self.__drones, f"Drone id {drone_id} not in the registered drones."
		self.__starting_rounds[drone_id] = self.__round

		if self.__drones[drone_id] == "Vicon":
			self.__angles[drone_id] = 0
			i = 0
			for key in self.__angles:
				self.__angles[key] = 2*math.pi / len(self.__angles) * i
				i += 1

		self.__current_setpoints[drone_id] = self.generate_new_setpoint(self.__drones[drone_id])
		self.__circle_pyramid_idx = None


