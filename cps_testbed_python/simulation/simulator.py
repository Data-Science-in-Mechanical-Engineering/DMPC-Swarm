import copy
import sys
import os

from simulation.gym_pybullet_drones.envs.BaseAviary import DroneModel
from simulation.gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from simulation.gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from simulation.gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from simulation.gym_pybullet_drones.utils.utils import sync

from network import network as net

import useful_scripts.custom_logger as custom_logger
import evaluation.logger as log

import compute_unit.agents as da
import compute_unit.trajectory_generation.trajectory_generation as tg

import time
import math
import numpy as np
import pybullet as p
import pickle

import cv2 as cv


# calculations of the round length.
def calculate_num_slots(num_messages):
	base_num_rounds = 100
	return max(3 * num_messages, base_num_rounds)


def calculate_round_time(message_size, num_messages):
	base_num_rounds = 100
	num_rounds = calculate_num_slots(num_messages)
	T_slot = calculate_slot_time(message_size, num_messages)
	return T_slot*num_rounds / 1000


def calculate_slot_time(message_size, num_messages):
	S_v = math.ceil(num_messages/8)
	S = 12 + 2*S_v + message_size
	T_a = (440+4*S)*1.037  # 4 us for BLW, 32 for IEEE 802.15.4
	T_p = 600 + (26+0.155*(S_v+message_size))*num_messages+1.8*S
	T_slot = max(T_a, T_p)
	return T_slot


def calculate_num_messages(message_size, message_list):
	num_messages = 0
	for m in message_list:
		num_messages += math.ceil(m / message_size - 1e-6)
	return num_messages + 1 # because of initator message


def get_min(arr):
	min_value = arr[0]
	min_ind = 0
	for i in range(1, len(arr)):
		if min_value > arr[i]:
			min_ind = i
			min_value = arr[i]

	return min_ind, min_value


def calculate_min_round_time(size_cf_messages, size_cu_messages, num_cf, num_cu):
    message_list = [size_cf_messages] * num_cf + [size_cu_messages] * num_cu
    sizes = [i for i in range(10, 1000)]
    num_messages = [calculate_num_messages(s, message_list) for s in sizes]
    round_times = [calculate_round_time(sizes[i], num_messages[i]) for i in range(len(sizes))]

    best_ind, best_time = get_min(round_times)
    return best_time


class Simulation:
    """
    defines a simulation instance

    Methods:
        run: runs the simulation
    """
    def __init__(self, ARGS, cus=[]):
        """constructor

        Parameters:
            ARGS:
                options for the simulation

        """
        self.__id = ARGS.sim_id
        # copy ARGS in case it gets modified later
        self.__ARGS = copy.deepcopy(ARGS)

        # initialize the simulation
        self.__testbed = self.__ARGS.testbed

        self.__INIT_XYZS = self.__ARGS.INIT_XYZS
        self.__INIT_TARGETS = self.__ARGS.INIT_TARGETS

        # self.__INIT_XYZS = np.array([[0.5, 1, 2], [4, 1.4, 2]])
        # self.__INIT_TARGETS = np.array([[4, 1, 2], [0.5, 1.4, 2]])
        self.__INIT_RPYS = np.array([[0, 0, 0] for i in range(self.__ARGS.num_drones)])  # initial rotation
        self.__INIT_VELOCITY = np.array([[0, 0, 0]])  # initialize velocities

        # 0.1s for calculation
        self.__round_time = calculate_min_round_time(size_cf_messages=2+2*9+1+2*3+2+1+1, num_cf=self.__ARGS.num_drones,
                                                     size_cu_messages=2+2*3*self.__ARGS.prediction_horizon+2*9+2+1+1,
                                                     num_cu=self.__ARGS.num_computing_agents)/1000.0 + 0.1 \
            if not self.__ARGS.use_communication_freq_hz else 1.0 / self.__ARGS.communication_freq_hz

        print(f"Round length: {self.__round_time}")

        self.__AGGR_PHY_STEPS = self.__ARGS.sim_steps_per_control
        self.__ARGS.communication_freq_hz = 1 / self.__round_time
        self.__ARGS.control_freq_hz = self.__ARGS.communication_freq_hz * self.__ARGS.control_steps_per_round
        self.__ARGS.simulation_freq_hz = self.__ARGS.control_freq_hz * self.__AGGR_PHY_STEPS

        # create environment without video capture
        self.__env = CtrlAviary(drone_model=self.__ARGS.drone,
                                num_drones=self.__ARGS.num_drones,
                                initial_xyzs=self.__INIT_XYZS,
                                initial_rpys=self.__INIT_RPYS,
                                physics=self.__ARGS.physics,
                                neighbourhood_radius=10,
                                freq=self.__ARGS.simulation_freq_hz,
                                aggregate_phy_steps=self.__AGGR_PHY_STEPS,
                                gui=self.__ARGS.gui,
                                record=self.__ARGS.record_video,
                                obstacles=self.__ARGS.obstacles,
                                user_debug_gui=self.__ARGS.user_debug_gui
                                )

        # Obtain the PyBullet Client ID from the environment
        self.__PYB_CLIENT = self.__env.getPyBulletClient()

        # Remove empty menus
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # Initialize the logger
        self.__desample = 1
        self.__logger = custom_logger.CustomLogger(ARGS=self.__ARGS,
                                                   logging_freq_hz=int(
                                                       self.__ARGS.simulation_freq_hz / self.__AGGR_PHY_STEPS / self.__desample),
                                                   num_drones=self.__ARGS.num_drones,
                                                   duration_sec=self.__ARGS.duration_sec)


        # Initialize the controllers for each drone
        if self.__ARGS.drone in [DroneModel.CF2X, DroneModel.CF2P]:
            self.__ctrl = [DSLPIDControl(drone_model=self.__ARGS.drone) for i in range(self.__ARGS.num_drones)]
        elif self.__ARGS.drone in [DroneModel.HB]:
            self.__ctrl = [SimplePIDControl(drone_model=self.__ARGS.drone) for i in range(self.__ARGS.num_drones)]

        # initialize the network
        self.__agents = []
        self.__computing_agents = []
        delta_t = self.__round_time
        collision_constraint_sample_points = np.linspace(0, self.__ARGS.prediction_horizon * delta_t,
                    int(((self.__ARGS.prediction_horizon) * 2+1)))
        collision_constraint_sample_points = collision_constraint_sample_points[1:]
        trajectory_generator_options = tg.TrajectoryGeneratorOptions(
            # different amount of sample point to increase resolution
            objective_function_sample_points=np.linspace(delta_t, self.__ARGS.prediction_horizon * delta_t,
                                                         int((self.__ARGS.prediction_horizon * 1))),
            objective_function_sample_points_pos=np.array([self.__ARGS.prediction_horizon * delta_t]),
            state_constraint_sample_points=np.linspace(delta_t, self.__ARGS.prediction_horizon * delta_t,
                                                       int((self.__ARGS.prediction_horizon * 1))),
            collision_constraint_sample_points=collision_constraint_sample_points,
            weight_state_difference=np.eye(3) * self.__ARGS.weight_state,
            weight_state_derivative_1_difference=np.eye(3) * self.__ARGS.weight_speed,
            weight_state_derivative_2_difference=np.eye(3) * self.__ARGS.weight_acc,
            weight_state_derivative_3_difference=np.eye(3) * self.__ARGS.weight_jerk,
            max_speed=np.array([2.0, 2.0, 2.0]),  #np.array([1.5, 1.5, 1.5])
            max_position=self.__ARGS.max_positions,
            max_acceleration=np.array([5.0, 5.0, 5.0]),  #np.array([5, 5, 5])
            max_jerk=np.array([5.0, 5.0, 5.0]),
            min_position=self.__ARGS.min_positions,
            r_min=self.__ARGS.r_min,
            optimization_variable_sample_time=delta_t / 1.0,  # can be tuned
            num_drones=self.__ARGS.num_drones,
            skewed_plane_BVC=self.__ARGS.skewed_plane_BVC,
            use_qpsolvers=self.__ARGS.use_qpsolvers,
            downwash_scaling_factor=self.__ARGS.downwash_scaling_factor,
            use_soft_constraints=self.__ARGS.use_soft_constraints,
            guarantee_anti_collision=self.__ARGS.guarantee_anti_collision,
            soft_constraint_max=self.__ARGS.soft_constraint_max,
            weight_soft_constraint=self.__ARGS.weight_soft_constraint,
            min_distance_cooperative=self.__ARGS.min_distance_cooperative,
            weight_cooperative=self.__ARGS.weight_cooperative,
            cooperative_normal_vector_noise=self.__ARGS.cooperative_normal_vector_noise,
            width_band=self.__ARGS.width_band
        )

        slot_group_trajectory = net.SlotGroup(0, False, self.__ARGS.num_computing_agents)
        slot_group_state = net.SlotGroup(2, False, self.__ARGS.num_drones - self.__ARGS.num_computing_agents)
        slot_group_setpoints = net.SlotGroup(3, False, 1)

        agent_ids = {}
        for i in range(0, self.__ARGS.num_drones):
            if self.__ARGS.load_cus:
                trajectory_starting_times = []
                trajectory_cu_id = 0
                for cu in self.__ARGS.cus:
                    print(cu.current_time)
                    if len(cu.get_trajectory_tracker().get_information(self.__ARGS.drone_ids[i]).content) == 1:
                        trajectory = cu.get_trajectory_tracker().get_information(self.__ARGS.drone_ids[i]).content[0]
                    else:
                        trajectory = cu.get_trajectory_tracker().get_information(self.__ARGS.drone_ids[i]).content[1]
                        print("44444444444")
                        print(trajectory.trajectory_start_time)
                        if abs(trajectory.trajectory_start_time - (self.__ARGS.load_cus_round_nmbr)*0.2) < 1e-5:
                            assert False
                            trajectory = cu.get_trajectory_tracker().get_information(self.__ARGS.drone_ids[i]).content[0]
                            print(trajectory.trajectory_start_time)
                    trajectory_starting_times.append(trajectory.trajectory_start_time)
                    trajectory_cu_id = trajectory.trajectory_calculated_by
                print(trajectory_starting_times)
                trajectory_start_time = min(trajectory_starting_times)

            drone_id = self.__ARGS.drone_ids[i]
            agent = da.RemoteDroneAgent(ID=drone_id,
                                        slot_group_planned_trajectory_id=slot_group_trajectory.id,
                                        slot_group_state_id=slot_group_state.id,
                                        init_position=self.__ARGS.INIT_XYZS_id[drone_id],
                                        target_position=self.__INIT_TARGETS[drone_id],
                                        communication_delta_t=delta_t,
                                        trajectory_generator_options=trajectory_generator_options,
                                        prediction_horizon=self.__ARGS.prediction_horizon,
                                        order_interpolation=self.__ARGS.interpolation_order,
                                        target_positions=self.__INIT_TARGETS[drone_id],
                                        other_drones_ids=self.__ARGS.drone_ids,
                                        load_cus_round_nmbr=0 if not self.__ARGS.load_cus else self.__ARGS.load_cus_round_nmbr + 1,
                                        trajectory_start_time= 0 if not self.__ARGS.load_cus else trajectory_start_time,
                                        trajectory_cu_id=-1 if not self.__ARGS.load_cus else trajectory_cu_id)

            agent_ids[i] = drone_id
            self.__agents.append(agent)

        prio = 0
        for i in range(self.__ARGS.num_drones, self.__ARGS.num_drones + min(self.__ARGS.num_computing_agents, 10000)):  #, self.__ARGS.num_drones)):
            if len(cus) == 0:
                cu_id = self.__ARGS.computing_agent_ids[i-self.__ARGS.num_drones]
                computing_agent = da.ComputeUnit(ID=cu_id, slot_group_planned_trajectory_id=slot_group_trajectory.id,
                                                 slot_group_drone_state=slot_group_state.id,
                                                 computing_agents_ids=self.__ARGS.computing_agent_ids,
                                                 communication_delta_t=delta_t,
                                                 trajectory_generator_options=trajectory_generator_options,
                                                 prediction_horizon=self.__ARGS.prediction_horizon,
                                                 num_computing_agents=self.__ARGS.num_computing_agents,
                                                 offset=(cu_id - self.__ARGS.num_drones) * int(
                                                          self.__ARGS.num_drones / max(
                                                              (self.__ARGS.num_computing_agents), 1)),
                                                 alpha_1=self.__ARGS.alpha_1,
                                                 alpha_2=self.__ARGS.alpha_2, alpha_3=self.__ARGS.alpha_3, alpha_4=self.__ARGS.alpha_4,
                                                 remove_redundant_constraints=self.__ARGS.remove_redundant_constraints,
                                                 ignore_message_loss=self.__ARGS.ignore_message_loss,
                                                 use_high_level_planner=self.__ARGS.use_high_level_planner,
                                                 agent_dodge_distance=self.__ARGS.agent_dodge_distance,
                                                 use_own_targets=self.__ARGS.use_own_targets,
                                                 slot_group_setpoints_id=slot_group_setpoints.id,
                                                 send_setpoints=i==self.__ARGS.num_drones,
                                                 simulated=self.__ARGS.simulated,
                                                 use_optimized_constraints=self.__ARGS.use_optimized_constraints,
                                                 setpoint_creator=self.__ARGS.setpoint_creator,
                                                 pos_offset=self.__ARGS.pos_offset,
                                                 weight_band=self.__ARGS.weight_band,
                                                 simulate_quantization=self.__ARGS.simulate_quantization,
                                                 save_snapshot_times=self.__ARGS.save_snapshot_times,
                                                 show_animation=i==self.__ARGS.num_drones and self.__ARGS.show_animation)

                for drone_id in self.__ARGS.drone_ids:
                    computing_agent.add_new_drone(drone_id)
                prio += 1
            else:
                computing_agent = cus[i - self.__ARGS.num_drones]
                # computing_agent.set_simulate_quantization(self.__ARGS.simulate_quantization)
                computing_agent.set_current_time((self.__ARGS.load_cus_round_nmbr) * 0.2)
                computing_agent.set_save_snapshot_times([])
                # computing_agent.round_started()
            self.__agents.append(computing_agent)
            self.__computing_agents.append(computing_agent)

        self.__network = net.Network(self.__agents, rounds_lost=self.__ARGS.rounds_lost,
                                     message_loss=self.__ARGS.message_loss_probability)
        self.__network.add_slot_group(slot_group_trajectory)
        self.__network.add_slot_group(slot_group_state)
        self.__network.add_slot_group(slot_group_setpoints)

        self.__easy_logger = log.EasyLogger()

        self.__easy_logger.add_data_point("INIT_XYZS", self.__ARGS.INIT_XYZS)
        self.__easy_logger.add_data_point("INIT_targets", self.__INIT_TARGETS)

                                          # load testbed
        #urdf_path = os.path.dirname(os.path.abspath(__file__)) + "/../../cube/"
        #p.loadURDF(os.path.join(urdf_path, "testbed.urdf"))

    def run(self):
        """
        runs the simulation

        Return:
            simulation_result: logger
                result of simulation including:
                    - success: bool
                    - state of drones: np.array
                    - INIT_XYZS: np.array
                    - INIT_TARGETS: np.array
        """
        num_image = 0
        resolution_video = [self.__ARGS.resolution_video, int(self.__ARGS.resolution_video*9/16)]
        projectionMatrix_video = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=16 / 9,
            nearVal=0.1,
            farVal=100.1)
        focal_length_video = resolution_video[1] / (2 * math.tan(math.pi / 180 * 45 / 2))

        viewMatrix4 = p.computeViewMatrix(
            cameraEyePosition=[2, -4, 1.8],
            cameraTargetPosition=[2, 2, 3],
            cameraUpVector=[0, 0, 1])

        viewMatrix4 = p.computeViewMatrix(
            cameraEyePosition=[0, 0, 8],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[1, 0, 0])

        """viewMatrix4 = p.computeViewMatrix(
            cameraEyePosition=[2, 2, 7],
            cameraTargetPosition=[2, 2, 0],
            cameraUpVector=[1, 0, 0])

        viewMatrix4 = p.computeViewMatrix(
            cameraEyePosition=[8, 1.8, 4],
            cameraTargetPosition=[1, 2.5, 0.5],
            cameraUpVector=[0, 0, 1])"""

        """viewMatrix4 = p.computeViewMatrix(
            cameraEyePosition=[5, 1.8, 2],
            cameraTargetPosition=[1, 2.5, 0.5],
            cameraUpVector=[0, 0, 1])

        viewMatrix4 = p.computeViewMatrix(
            cameraEyePosition=[5, 2.0, 2],
            cameraTargetPosition=[1, 2.0, 1],
            cameraUpVector=[0, 0, 1])"""

        #viewMatrix4 = p.computeViewMatrix(
        #    cameraEyePosition=[2, 2, 8],
        #    cameraTargetPosition=[2, 2, 0],
        #    cameraUpVector=[1, 0, 0])

        intrinsic_matrix = np.array([[-focal_length_video, 0, resolution_video[0] / 2, 0],
                                     [0, focal_length_video, resolution_video[1] / 2, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 0]])
        trans = np.reshape(np.array(viewMatrix4), (4, 4), 'F')

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        COM_EVERY_N_STEPS = self.__ARGS.control_steps_per_round * self.__AGGR_PHY_STEPS
        action = {str(i): np.array([0, 0, 0, 0]) for i in range(self.__ARGS.num_drones)}
        START = time.time()
        critical_dist_to_target = 0.05

        num_image = 0
        current_time = 0
        print('Starting Simulation No. ' + str(self.__id))
        for ind in range(0, 1):
            pos_last = -1
            ball_position_estimator = None
            desample_time = 0
            for i in range(0, int(self.__ARGS.duration_sec * self.__env.SIM_FREQ), self.__AGGR_PHY_STEPS):
                #### Step the simulation ###################################
                obs, reward, done, info = self.__env.step(action)

                if i % (COM_EVERY_N_STEPS) == 0:
                    # the current position should be set before the network transmits its data (in reality the measurments
                    # are taken after the end of communication round)

                    all_targets_reached = True
                    for j in range(0, self.__ARGS.num_drones):
                        self.__agents[j].position = obs[str(j)]["state"][0:3]

                        dist_to_target = np.linalg.norm(self.__agents[j].position - self.__agents[j].target_position)
                        # set transition time only, if it's not already set to not override it
                        reached_target = self.__agents[j].target_reached
                        if dist_to_target < critical_dist_to_target and not reached_target:
                            self.__agents[j].transition_time = i / self.__env.SIM_FREQ
                            self.__agents[j].target_reached = True
                        elif dist_to_target >= critical_dist_to_target:
                            self.__agents[j].transition_time = None
                            self.__agents[j].target_reached = False

                        all_targets_reached = all_targets_reached and reached_target
                        # print(self.__agents[j].traj_state[:3])

                    # if all_targets_reached:
                    #     print("all targets reached")

                    # step network
                    self.__network.step()

                    #for jgh in range(6):
                    #    temp = self.__computing_agents[0].get_pos(jgh)
                    #    self.__easy_logger.add_data_point(f"state_set{jgh}", copy.deepcopy(temp))

                    # check if agents do the correct thing
                    ordered_indexes = None
                    correct = True
                    for agent_idx in range(self.__ARGS.num_computing_agents):
                        if self.__computing_agents[agent_idx].ordered_indexes is not None:
                            if ordered_indexes is None:
                                ordered_indexes = self.__computing_agents[agent_idx].ordered_indexes
                            else:
                                if np.any(self.__computing_agents[agent_idx].ordered_indexes != ordered_indexes):
                                    correct = False
                    for ca in self.__computing_agents:
                        if ca.last_calc_time is not None:
                            self.__logger.log_calc_time(ca.last_calc_time)
                        if ca.last_num_constraints is not None:
                            self.__logger.log_num_constraints(ca.last_num_constraints)

                    #### do processing of current measurements which take longer and are performed during the communication
                    #### round


                #### do drone low level flight controller computations
                # (on our real system this is performed in parallel to everything else) ##############

                # do low level control
                if self.__ARGS.save_video and i % 16 == 0:
                    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                        width=resolution_video[0],
                        height=resolution_video[1],
                        viewMatrix=viewMatrix4,
                        projectionMatrix=projectionMatrix_video)

                    rgbImg = np.copy(np.reshape(np.array(rgbImg), newshape=(height, width, 4)))
                    bgrImg = np.reshape(np.array(rgbImg), newshape=(height, width, 4))[:, :, 0:3]
                    bgrImg[:, :, 0] = rgbImg[:, :, 2]
                    bgrImg[:, :, 2] = rgbImg[:, :, 0]
                    bgrImg = np.copy(bgrImg)

                    # paint planned trajectories
                    for agent_idx in range(self.__ARGS.num_drones):
                        # draw target
                        if self.__agents[self.__ARGS.num_drones].get_targets()[self.__agents[agent_idx].ID] is not None:
                            draw_circle(bgrImg,
                                       self.__agents[self.__ARGS.num_drones].get_targets()[self.__agents[agent_idx].ID],
                                       trans, intrinsic_matrix,
                                       colors[agent_idx % len(colors)]
                                       )
                        if self.__agents[self.__ARGS.num_drones].get_drone_intermediate_setpoint(self.__agents[agent_idx].ID) is not None:
                                draw_cross(bgrImg,
                                            self.__agents[self.__ARGS.num_drones].get_drone_intermediate_setpoint(
                                                self.__agents[agent_idx].ID),
                                            trans, intrinsic_matrix,
                                            colors[agent_idx % len(colors)],
                                            size=10, width=3
                                            )
                        # get trajectory
                        traj = self.__agents[agent_idx].get_planned_trajectory(0.2)
                        for idx in range(len(traj)-1):
                            draw_line(bgrImg, traj[idx], traj[idx+1], trans, intrinsic_matrix,
                                       colors[agent_idx % len(colors)])

                    isExist = os.path.exists(self.__ARGS.path + "_simnr_" + str(self.__id))
                    if not isExist:
                        # Create a new directory because it does not exist
                        os.makedirs(self.__ARGS.path + "_simnr_" + str(self.__id))
                    cv.imwrite(os.path.join(self.__ARGS.path + "_simnr_" + str(self.__id) , 'ImgPredict1_' + str(num_image) + '.jpg'), bgrImg)
                    num_image += 1

                next_state = np.zeros((self.__ARGS.num_drones, 9))
                #### Compute control for the current way point #############
                for j in range(self.__ARGS.num_drones):
                    control_interval = 1.0 / self.__ARGS.control_freq_hz
                    self.__agents[j].state = np.array([obs[str(j)]["state"][0], obs[str(j)]["state"][1],
                                                       obs[str(j)]["state"][2], obs[str(j)]["state"][10],
                                                       obs[str(j)]["state"][11], obs[str(j)]["state"][12],
                                                       0, 0, 0])
                    next_state[j, :] = np.copy(self.__agents[j].next_planned_state(control_interval))
                    # print(str(i%COM_EVERY_N_STEPS / CTRL_EVERY_N_STEPS * control_interval) + '\t' + str(next_pos[j, :]))
                    action[str(j)], _, _ = self.__ctrl[j].computeControlFromState(
                        control_timestep=self.__AGGR_PHY_STEPS * self.__env.TIMESTEP,
                        state=obs[str(j)]["state"],
                        target_pos=next_state[j, :3],
                        target_vel=next_state[j, 3:6],
                        target_rpy=self.__INIT_RPYS[j, :]
                    )
                    if self.__ARGS.log_state:
                        self.__easy_logger.add_data_point(f"state_{j}", self.__agents[j].state)
                        self.__easy_logger.add_data_point(f"state_set{j}", copy.deepcopy(next_state[j, :]))
                current_time += control_interval

                # check if drones have crashed
                for j in range(0, self.__ARGS.num_drones):
                    # for n in range(0, self.__ARGS.num_drones):
                    scaling_matrix = np.diag([1, 1, 1.0 / float(self.__ARGS.downwash_scaling_factor_crit)])
                    if any([np.linalg.norm(
                            (self.__agents[j].state[0:3] - self.__agents[
                                n].state[0:3]) @ scaling_matrix) < self.__ARGS.r_min_crit for n in
                            range(0, self.__ARGS.num_drones) if n != j]):
                        print([np.linalg.norm(
                            (self.__agents[j].position - self.__agents[n].position) @ scaling_matrix) for n in
                            range(0, self.__ARGS.num_drones) if n != j])
                        self.__agents[j].crashed = True

                #### Log the simulation ####################################
                if desample_time % self.__desample == 0:
                    for j in range(self.__ARGS.num_drones):
                        self.__logger.log(drone=j, timestamp=i / self.__env.SIM_FREQ, state=obs[str(j)]["state"],
                                          control=np.hstack((next_state[j, :3], next_state[j, 3:6], np.zeros(6))))
                        self.__logger.log_position(j, i / self.__env.SIM_FREQ, obs[str(j)]["state"][0:3], next_state[j, :3])

                desample_time += 1
                #### Sync the simulation ###################################
                if self.__ARGS.gui:
                    sync(i, START, self.__env.TIMESTEP)

                #### Stop the simulation, if all drones reached their targets or at least one agents has crashed
                stop = True
                stop2 = False
                for j in range(self.__ARGS.num_drones):

                    stop = stop and (self.__agents[j].crashed or self.__agents[j].all_targets_reached)

                    if self.__agents[j].crashed:
                        stop2 = True
                if (stop or stop2) and self.__ARGS.abort_simulation:
                    break

        # log results of simulation
        one_agent_crashed = False
        num_targets_reached = 0
        for j in range(self.__ARGS.num_drones):
            one_agent_crashed = one_agent_crashed or self.__agents[j].crashed
            num_targets_reached += 1 if self.__agents[j].all_targets_reached else 0
        self.__easy_logger.add_data_point("crashed", one_agent_crashed)

        num_optimizer_runs = 0
        num_succ_optimizer_runs = 0
        for j in range(self.__ARGS.num_computing_agents):
            num_optimizer_runs += self.__computing_agents[j].total_num_optimizer_runs
            num_succ_optimizer_runs += self.__computing_agents[j].num_succ_optimizer_runs

        self.__easy_logger.add_data_point("num_optimizer_runs", num_optimizer_runs)
        self.__easy_logger.add_data_point("num_succ_optimizer_runs", num_succ_optimizer_runs)
        self.__easy_logger.add_data_point("target_reached_time", current_time)
        self.__easy_logger.add_data_point("num_targets_reached", num_targets_reached)

        with open(
                os.path.join(self.__ARGS.path, "simulation_result-" + str(self.__ARGS.num_drones) + "_drones_simnr_" + str(self.__id) + ".pkl"), 'wb') \
                as out_file:
            pickle.dump(self.__easy_logger.get_data(), out_file)
        return True

    def save(self, simulation_logger):
        with open(
                os.path.join(self.__ARGS.path, "simulation_result-" + str(self.__ARGS.num_drones) + "_drones_simnr_" + str(self.__id) + ".pkl"), 'wb') \
                as out_file:
            pickle.dump(simulation_logger, out_file)


def draw_line(image, point3D1, point3D2, extrinsic_matrix, intrinsic_matrix, color):
    """draws a point into an image

    Parameters
    ----------
        image: np.array
            bgr image the point should be drawn into
        point3D: np.array
            3D point in world coordinates that should be drawn into the image
        extrinsic_matrix:
            extrinsic matrix of the camera that produces the image
        intrinsic_matrix: np.array
            intrinsic matrix of the camera that produces the image
        color: np.array
            color of the point
    """
    point_homogeneous_coordinates1 = np.ones((4,))
    point_homogeneous_coordinates1[0:3] = point3D1
    point_image1 = intrinsic_matrix @ extrinsic_matrix @ point_homogeneous_coordinates1

    point_homogeneous_coordinates2 = np.ones((4,))
    point_homogeneous_coordinates2[0:3] = point3D2
    point_image2 = intrinsic_matrix @ extrinsic_matrix @ point_homogeneous_coordinates2
    cv.line(image, (int(point_image1[0] / (point_image1[2]+1e-6)), int(point_image1[1] / (point_image1[2]+1e-6))),
             (int(point_image2[0] / (point_image2[2]+1e-6)), int(point_image2[1] / (point_image2[2]+1e-6))), color,
             thickness=3)

def draw_circle(image, point3D, extrinsic_matrix, intrinsic_matrix, color, size=5, width=3):
    """draws a point into an image

	Parameters
	----------
		image: np.array
			bgr image the point should be drawn into
		point3D: np.array
			3D point in world coordinates that should be drawn into the image
		extrinsic_matrix:
			extrinsic matrix of the camera that produces the image
		intrinsic_matrix: np.array
			intrinsic matrix of the camera that produces the image
		color: np.array
			color of the point
	"""
    point_homogeneous_coordinates = np.ones((4,))
    point_homogeneous_coordinates[0:3] = point3D
    point_image = intrinsic_matrix @ extrinsic_matrix @ point_homogeneous_coordinates
    cv.circle(image, (int(point_image[0] / point_image[2]), int(point_image[1] / point_image[2])), size,
              color, width)


def draw_cross(image, point3D, extrinsic_matrix, intrinsic_matrix, color, size=5, width=3):
    point_homogeneous_coordinates = np.ones((4,))
    point_homogeneous_coordinates[0:3] = point3D
    point_image = intrinsic_matrix @ extrinsic_matrix @ point_homogeneous_coordinates
    point_x = int(point_image[0] / point_image[2])
    point_y = int(point_image[1] / point_image[2])
    cv.line(image, (point_x-size//2, point_y), (point_x+size//2, point_y), color, width)
    cv.line(image, (point_x, point_y - size // 2), (point_x, point_y + size // 2), color, width)
