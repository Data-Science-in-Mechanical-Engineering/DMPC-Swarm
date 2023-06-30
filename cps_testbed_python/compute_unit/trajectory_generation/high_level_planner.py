import numpy as np
import math
import matplotlib.pyplot as plt
import time
import threading


class HLPThread(threading.Thread):

	def __init__(self, thread_id, cu, current_pos, current_target_positions, horizon, step_size, downwash_scaling_factor,
				max_position, min_position):
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.__cu = cu
		self.__current_pos = current_pos
		self.__current_target_positions = current_target_positions
		self.__horizon = horizon
		self.__step_size = step_size
		self.__downwash_scaling_factor = downwash_scaling_factor
		self.__max_position = max_position
		self.__min_position = min_position

	def run(self):
		high_level_setpoint_trajectory = calculate_setpoints(self.__current_pos, self.__current_target_positions,
								   							 horizon=self.__horizon,
								   						     step_size=self.__step_size,
															 downwash_scaling_factor=self.__downwash_scaling_factor,
								   							 max_position=self.__max_position,
								   							 min_position=self.__min_position
								   							)
		self.__cu.set_high_level_setpoint_trajectory(high_level_setpoint_trajectory)
		self.__cu.hlp_finished_callback()


def rotation_matrix(axis, theta):
	"""
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
	axis = np.asarray(axis)
	axis = axis / math.sqrt(np.dot(axis, axis))
	assert(not np.any(np.isnan(axis)))
	a = math.cos(theta / 2.0)
	b, c, d = -axis * math.sin(theta / 2.0)
	aa, bb, cc, dd = a * a, b * b, c * c, d * d
	bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
	return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
					 [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
					 [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def distance_to_line(p1, p2, p):
	# the distance calculation does not work, if p lies on the line between p1 and p2. In this
	# case we have to return 0
	if np.linalg.norm(p1-p) < 1e-3 or np.linalg.norm(p2-p) < 1e-3:
		return 0

	p12 = (p2-p1) / (np.linalg.norm(p2-p1)+1e-9)
	if np.dot(p-p1, p12) < 0:
		return np.linalg.norm(p1-p)
	if np.dot(p-p1, p12) > np.linalg.norm(p2-p1):
		return np.linalg.norm(p2 - p)
	dp = p-p1 - np.dot(p-p1, p12)*p12
	return np.linalg.norm(dp)


def distance_to_line_inf(p1, p2, p):
	# the distance calculation does not work, if p lies on the line between p1 and p2. In this
	# case we have to return 0
	if np.linalg.norm(p1-p) < 1e-3 or np.linalg.norm(p2-p) < 1e-3:
		return 0
	p12 = (p2-p1) / (np.linalg.norm(p2-p1)+1e-9)
	dp = p-p1 - np.dot(p-p1, p12)*p12
	return np.linalg.norm(dp)


def is_valid(point, setpoints, agent_id, min_distance=0.49, downwash_scaling=np.eye(3),
			 max_position=np.array([1.5, 1.5, 3]),
			 min_position=np.array([-1.5, -1.5, 0.1])):
	if np.any(point < min_position):
		return False
	if np.any(point > max_position):
		return False
	for ids in setpoints.keys():
		if ids != agent_id:
			if distance_to_line(downwash_scaling@setpoints[agent_id][-1], downwash_scaling@point, downwash_scaling@setpoints[ids][-1]) < min_distance:#np.linalg.norm(downwash_scaling@(point - setpoints[ids][-1]).T) < min_distance:
				return False
	return True


def calculate_setpoints(current_pos, target_pos, horizon=5, step_size=1.0, step_angle=5 * math.pi / 180,
						downwash_scaling_factor=1,
						max_position=np.array([1.5, 1.5, 3]),
						min_position=np.array([-1.5, -1.5, 0.1])
						):
	downwash_coefficient = downwash_scaling_factor
	downwash_scaling = np.diag([1, 1, 1.0 / float(downwash_coefficient)])
	setpoints = {agent_id: [current_pos[agent_id]] for agent_id in current_pos.keys()}
	for i in range(horizon):
		for agent_id in current_pos.keys():
			best_direction = target_pos[agent_id] - setpoints[agent_id][-1]
			# if we are near the target, make step size smaller
			current_step_size = step_size
			if np.linalg.norm(best_direction) < step_size:
				current_step_size = np.linalg.norm(best_direction)
				#setpoints[agent_id].append(target_pos[agent_id])
				#continue
			# if we are very close to the target set this as setpoint
			if np.linalg.norm(best_direction) < 1e-3:
				setpoints[agent_id].append(target_pos[agent_id])
				continue
			best_direction = best_direction / np.linalg.norm(best_direction)
			x_step = np.cross(np.array([1, 0, 0]), best_direction) if np.linalg.norm(
				np.array([1, 0, 0]) - np.abs(best_direction)) > 0.01 else np.cross(np.array([0, 1, 0]), best_direction)
			y_step = np.cross(x_step, best_direction)
			best_direction *= current_step_size
			start_time = time.time()
			x_rot_pos = rotation_matrix(x_step, step_angle)
			x_rot_neg = rotation_matrix(-x_step, step_angle)
			y_rot_pos = rotation_matrix(y_step, 3*step_angle)
			y_rot_neg = rotation_matrix(-y_step, 3*step_angle)
			# spiral around the first calculated initial direction.
			sign = 1
			use_x_axis = True
			steps = 1
			x_angle = 0
			y_angle = 0
			best_direction_temp = np.copy(best_direction)
			found_solution = False
			#print("5555555555555555555555555555")
			#print(step_size)
			if is_valid(setpoints[agent_id][-1]+best_direction_temp, setpoints, agent_id):
				found_solution = True
			else:
				while abs(x_angle) <= math.pi and abs(y_angle) <= math.pi:
					#print("---------------")
					#print(x_angle)
					#print(y_angle)
					#print(steps)
					for angle_steps in range(steps):
						rot_axis = None
						if sign == 1:
							if use_x_axis:
								rot_axis = x_rot_pos
								x_angle += step_angle
							else:
								rot_axis = y_rot_pos
								y_angle += step_angle
						else:
							if use_x_axis:
								rot_axis = x_rot_neg
								x_angle -= step_angle
							else:
								rot_axis = y_rot_neg
								y_angle -= step_angle
						#best_direction_temp = np.dot(rotation_matrix(x_step, x_angle), best_direction)
						#best_direction_temp = np.dot(rotation_matrix(y_step, y_angle), best_direction_temp)
						best_direction_temp = np.dot(rot_axis, best_direction_temp)
						#print("dddddddddddddddd")
						if is_valid(setpoints[agent_id][-1]+best_direction_temp, setpoints, agent_id, downwash_scaling=downwash_scaling,
									max_position=max_position,
									min_position=min_position):
							found_solution = True
							break
					if not use_x_axis:
						steps += 1
						sign *= -1
					use_x_axis = not use_x_axis
					if found_solution:
						break
			if found_solution:
				setpoints[agent_id].append(setpoints[agent_id][-1]+best_direction_temp)
			else:
				setpoints[agent_id].append(setpoints[agent_id][-1])
			#print(time.time() - start_time)
	return setpoints


def calculate_setpoints_one_drone(current_pos, target_pos, agent_id, horizon=5, step_size=1.0, step_angle=15 * math.pi / 180,
								downwash_scaling_factor=1,
								max_position=np.array([1.5, 1.5, 3]),
								min_position=np.array([-1.5, -1.5, 0.1]), r_min=0.3
								):
	downwash_coefficient = downwash_scaling_factor
	downwash_scaling = np.diag([1, 1, 1.0 / float(downwash_coefficient)])
	setpoints = {agent_id: [current_pos[agent_id]] for agent_id in current_pos.keys()}
	for i in range(horizon):
		best_direction = target_pos[agent_id] - setpoints[agent_id][-1]
		# if we are near the target, make step size smaller
		current_step_size = step_size
		if np.linalg.norm(best_direction) < step_size:
			current_step_size = np.linalg.norm(best_direction)
			#setpoints[agent_id].append(target_pos[agent_id])
			#continue
		# if we are very close to the target set this as setpoint
		if np.linalg.norm(best_direction) < 1e-3:
			setpoints[agent_id].append(target_pos[agent_id])
			continue
		best_direction = best_direction / np.linalg.norm(best_direction)
		x_step = np.cross(np.array([1, 0, 0]), best_direction) if np.linalg.norm(
			np.array([1, 0, 0]) - np.abs(best_direction)) > 0.01 else np.cross(np.array([0, 1, 0]), best_direction)
		y_step = np.cross(x_step, best_direction)
		best_direction *= current_step_size
		start_time = time.time()
		x_rot_pos = rotation_matrix(x_step, step_angle)
		x_rot_neg = rotation_matrix(-x_step, step_angle)
		y_rot_pos = rotation_matrix(y_step, step_angle)
		y_rot_neg = rotation_matrix(-y_step, step_angle)
		# spiral around the first calculated initial direction.
		sign = 1
		use_x_axis = True
		steps = 1
		x_angle = 0
		y_angle = 0
		best_direction_temp = np.copy(best_direction)
		found_solution = False
		#print("5555555555555555555555555555")
		#print(step_size)
		if is_valid(setpoints[agent_id][-1]+best_direction_temp, setpoints, agent_id, min_distance=r_min):
			found_solution = True
		else:
			while abs(x_angle) <= math.pi/2 and abs(y_angle) <= math.pi/2:
				#print("---------------")
				#print(x_angle)
				#print(y_angle)
				#print(steps)
				for angle_steps in range(steps):
					rot_axis = None
					if sign == 1:
						if use_x_axis:
							rot_axis = x_rot_pos
							x_angle += step_angle
						else:
							rot_axis = y_rot_pos
							y_angle += step_angle
					else:
						if use_x_axis:
							rot_axis = x_rot_neg
							x_angle -= step_angle
						else:
							rot_axis = y_rot_neg
							y_angle -= step_angle
					#best_direction_temp = np.dot(rotation_matrix(x_step, x_angle), best_direction)
					#best_direction_temp = np.dot(rotation_matrix(y_step, y_angle), best_direction_temp)
					best_direction_temp = np.dot(rot_axis, best_direction_temp)
					#print("dddddddddddddddd")
					if is_valid(setpoints[agent_id][-1]+best_direction_temp, setpoints, agent_id, downwash_scaling=downwash_scaling,
								max_position=max_position,
								min_position=min_position, min_distance=r_min):
						found_solution = True
						break
				if not use_x_axis:
					steps += 1
					sign *= -1
				use_x_axis = not use_x_axis
				if found_solution:
					break
		if found_solution:
			setpoints[agent_id].append(setpoints[agent_id][-1]+best_direction_temp)
		else:
			setpoints[agent_id].append(setpoints[agent_id][-1])
		#print(time.time() - start_time)
	return setpoints


def limit_to_max(current_setpoint, max_position):
	current_setpoint[0] = current_setpoint[0] if current_setpoint[0] < max_position[0] else max_position[0]
	current_setpoint[1] = current_setpoint[1] if current_setpoint[1] < max_position[1] else max_position[1]
	current_setpoint[2] = current_setpoint[2] if current_setpoint[2] < max_position[2] else max_position[2]


def limit_to_min(current_setpoint, min_position):
	current_setpoint[0] = current_setpoint[0] if current_setpoint[0] > min_position[0] else min_position[0]
	current_setpoint[1] = current_setpoint[1] if current_setpoint[1] > min_position[1] else min_position[1]
	current_setpoint[2] = current_setpoint[2] if current_setpoint[2] > min_position[2] else min_position[2]


def calculate_setpoints_one_droneV2(current_pos, target_pos, agent_id, horizon=5, step_size=1.0, step_angle=10 * math.pi / 180,
								downwash_scaling_factor=1,
								max_position=np.array([1.5, 1.5, 3]),
								min_position=np.array([-1.5, -1.5, 0.1]), r_min=0.5, max_iterations=200,
								):
	downwash_coefficient = downwash_scaling_factor
	downwash_scaling = np.diag([1, 1, 1.0 / float(downwash_coefficient)])
	current_pos_transform = {current_agent_id: downwash_scaling @ current_pos[current_agent_id] for current_agent_id in current_pos.keys()}
	target_pos_transform = {current_agent_id: downwash_scaling @ target_pos[current_agent_id] for current_agent_id in current_pos.keys()}
	setpoints = {current_agent_id: [current_pos_transform[current_agent_id]] for current_agent_id in current_pos.keys()}
	num_iteration = 0
	for i in range(horizon):
		if np.linalg.norm(setpoints[agent_id][-1] - target_pos_transform[agent_id]) < 1e-6:
			break
		valid = False
		current_setpoint = np.copy(target_pos_transform[agent_id])
		last_setpoint = setpoints[agent_id][-1]
		while not valid:
			valid = True
			for other_agent_id in current_pos.keys():
				if other_agent_id != agent_id:
					if distance_to_line(current_setpoint, last_setpoint, current_pos_transform[other_agent_id]) < r_min:
						valid = False
						a = current_pos_transform[other_agent_id] - last_setpoint
						b = current_setpoint - last_setpoint
						c = np.cross(a, b)
						d = np.cross(b, c)
						if np.linalg.norm(d) > 1e-5 and False:
							d = d / np.linalg.norm(d)
							d += np.random.randn(3)*0.0
							d = d / np.linalg.norm(d)
						else:
							d = np.random.randn(3)
							#d[2] = 0
							d = d / np.linalg.norm(d)

						s = 10
						for dist in range(s):
							current_setpoint = current_pos_transform[other_agent_id] - d * (r_min*(1+1e-5 + dist/s))
							limit_to_max(current_setpoint, downwash_scaling@max_position)
							limit_to_min(current_setpoint, downwash_scaling@min_position)
							if distance_to_line_inf(last_setpoint, current_setpoint, current_pos_transform[other_agent_id]) > r_min + 1e-5:
								break
						break
			num_iteration += 1
			if num_iteration < max_iterations:
				print("Max IT!!!!!!!!!")
				break
		setpoints[agent_id].append(np.copy(current_setpoint))
	downwash_scaling_back = np.diag([1, 1, float(downwash_coefficient)])
	setpoints = {current_agent_id: [downwash_scaling_back@s for s in setpoints[current_agent_id]] for current_agent_id in current_pos.keys()}
	print(setpoints)
	return setpoints


if __name__ == "__main__":
	r = 1.0
	num_agents = 10
	pos = {i: np.array([r*math.cos(2*math.pi/num_agents*i), r*math.sin(2*math.pi/num_agents*i), 1]) for i in range(num_agents)}
	targets = {i: np.array([r*math.cos(2*math.pi/num_agents*i+math.pi), r*math.sin(2*math.pi/num_agents*i+math.pi), 1]) for i in range(num_agents)}

	pos = {1: np.array([-0.6, -1.0, 1.0]), 2: np.array([0.0, -1.0, 1.0]), 3: np.array([0.6, -1.0, 1.0]), 4: np.array([-0.3, 1.0, 1.0]), 5: np.array([0.3, 1.0, 1.0])}
	targets = {1: np.array([-0.6, 1.0, 1.0]), 2: np.array([0.0, 1.0, 1.0]), 3: np.array([0.6, 1.0, 1.0]), 4: np.array([-0.3, -1.0, 1.0]), 5: np.array([0.3, -1.0, 1.0])}

	start_time = time.time()
	setpoints = calculate_setpoints(pos, targets, step_size=1.0)
	print(setpoints)
	print(time.time()-start_time)

	ax = plt.figure().add_subplot(projection='3d')

	for agent_ids in pos.keys():
		x = []
		y = []
		z = []
		for i in range(len(setpoints[agent_ids])):
			x.append(setpoints[agent_ids][i][0])
			y.append(setpoints[agent_ids][i][1])
			z.append(setpoints[agent_ids][i][2])
		ax.plot(np.array(x), np.array(y), np.array(z))
	ax.legend()

	plt.show()
