import math

from network import network as net
import compute_unit.trajectory_generation.trajectory_generation as tg

from compute_unit.trajectory_generation.statespace_model import TripleIntegrator
import copy
import numpy as np
from dataclasses import dataclass
from compute_unit.trajectory_generation.information_tracker import InformationTracker, Information, TrajectoryContent
import time
import compute_unit.trajectory_generation.high_level_planner as hlp

from compute_unit import real_compute_unit

import random

import pickle

NORMAL = 0
INFORMATION_DEPRECATED = 1
RECOVER_INFORMATION_NOTIFY = 2
RECOVER_INFORMATION_LISTEN = 3
WAIT_FOR_UPDATE = 4


@dataclass
class TrajectoryMessageContent:
    id: int  # the corresponding id for the coefficients
    coefficients: any
    init_state: any
    trajectory_start_time: any
    trajectory_calculated_by: int  # id of computing unit
    prios: any


@dataclass
class InformationDeprecatedContent:
    content: any  # we do not need this?


@dataclass
class RecoverInformationNotifyContent:
    cu_id: int
    drone_id: int


@dataclass
class EmtpyContent:
    prios: any


@dataclass
class StateMessageContent:
    state: any
    target_position: any
    trajectory_start_time: any
    trajectory_calculated_by: int   # stamp of trajectory the UAV is currently following
    target_position_idx: any
    coefficients: any
    init_state: any


@dataclass
class SetpointMessageContent:
    setpoints: any


def simulate_quantization_setpoints_message(message):
    message_quant = copy.deepcopy(message)
    if message.setpoints is None:
        return message_quant
    for k in message.setpoints:
        for i in range(3):
            message_quant.setpoints[k][i] = real_compute_unit.dequantize_pos(real_compute_unit.quantize_pos(message.setpoints[k][i]))
    return message_quant


def simulate_quantization_trajectory_message(message):
    message_quant = copy.deepcopy(message)
    coefficients = message.coefficients.coefficients if message.coefficients.valid else message.coefficients.alternative_trajectory
    for i in range(len(coefficients)):
        for j in range(3):
            if message_quant.coefficients.valid:
                message_quant.coefficients.coefficients[i][j] = real_compute_unit.dequantize_input(real_compute_unit.quantize_input(coefficients[i][j]))
            else:
                message_quant.coefficients.alternative_trajectory[i][j] = real_compute_unit.dequantize_input(real_compute_unit.quantize_input(coefficients[i][j]))
    for j in range(3):
        message_quant.init_state[j] = real_compute_unit.dequantize_pos(real_compute_unit.quantize_pos(message.init_state[j]))

    for j in range(3, 6):
        message_quant.init_state[j] = real_compute_unit.dequantize_vel(real_compute_unit.quantize_vel(message.init_state[j]))

    for j in range(6, 9):
        message_quant.init_state[j] = real_compute_unit.dequantize_acc(real_compute_unit.quantize_acc(message.init_state[j]))

    #for j in range(0, 3):
    #    message_quant.init_state[j] = real_compute_unit.dequantize_pos(real_compute_unit.quantize_pos(message.init_state[j]))
    return message_quant


def trajectories_equal(trajectory1, trajectory2):
    return abs(trajectory1.trajectory_start_time - trajectory2.trajectory_start_time) < 1e-4 \
           and trajectory1.trajectory_calculated_by == trajectory2.trajectory_calculated_by


def hash_trajectory(trajectory, init_state):
    coeff = trajectory.coefficients if trajectory.valid \
        else trajectory.alternative_trajectory
    coeff = coeff.tobytes()
    coeff = coeff + init_state.tobytes
    return hash(str(coeff))


def calculate_band_weight(p_target, p_self, p_other, weight=1.0, weight_angle=2):
    dp_target = p_target - p_self
    weight_mult = 1
    if np.linalg.norm(dp_target) <= 0.8:
        weight_mult = 1e-7
    dp_other = p_other - p_self
    dp_target /= np.linalg.norm(dp_target) + 1e-7
    dp_other /= np.linalg.norm(dp_other) + 1e-7
    angle = np.arccos(np.clip(np.dot(dp_target, dp_other), -1.0, 1.0))
    return weight_mult * weight * math.exp(math.sin(angle)*weight_angle)


class ComputationAgent(net.Agent):
    """this class represents a computation agent inside the network.
    Every agent is allowed to send in multiple slot groups. This is needed if e.g. the agent should send his sensor
    measurements but also if he is in error mode.

    Methods
    -------
    get_prio(slot_group_id):
        returns current priority
    get_message(slot_group_id):
        returns message
    send_message(message):
        send a message to this agent
    round_finished():
        callback for the network to signal that a round has finished
    """

    def __init__(self, ID, slot_group_planned_trajectory_id, slot_group_trajectory_initial_state,
                 slot_group_drone_state, init_positions, target_positions, agents_ids,
                 communication_delta_t,
                 trajectory_generator_options, pos_offset, prediction_horizon, num_computing_agents, comp_agent_prio,
                 computing_agents_ids, setpoint_creator, offset=0, use_event_trigger=False,
                 alpha_1=10, alpha_2=1000, alpha_3=1*0, alpha_4=0, remove_redundant_constraints=False,
                 slot_group_state_id=None,
                 slot_group_ack_id=100000, ignore_message_loss=False, use_own_targets=False,
                 state_feedback_trigger_dist=0.5, simulated=True, use_high_level_planner=True,
                 agent_dodge_distance=0.5, slot_group_setpoints_id=100000, send_setpoints=False, use_given_init_pos=False,
                 use_optimized_constraints=True, weight_band=1.0,
                 save_snapshot_times=[], snapshot_saving_path="",
                 simulate_quantization=False):
        """

        Parameters
        ----------
            ID:
                identification number of agent
            slot_group_planned_trajectory_id:
                ID of the slot group which is used
            init_positions:
                3D position of agents in a hash_map id of agent is key
            num_agents: int
                total number of agents in the system
            trajectory_generator_options: tg.TrajectoryGeneratorOptions
                options for data generator
            agents_ids: numpy.array, shape (num_agents,)
                ids of all agents in the system
            communication_delta_t: float
                time difference between two communication rounds
            prediction_horizon: int
                prediction horizon of optimizer
            order_interpolation: int
                order of interpolation
            comp_agent_prio:
                priority of the computaiton agent. agent with 0 gets the agent with the highes priority...

        """
        super().__init__(ID, [slot_group_planned_trajectory_id] if not send_setpoints else [slot_group_planned_trajectory_id, slot_group_setpoints_id])
        self.__comp_agent_prio = computing_agents_ids.index(ID)
        self.__slot_group_planned_trajectory_id = slot_group_planned_trajectory_id
        self.__slot_group_setpoints_id = slot_group_setpoints_id
        self.__send_setpoints = send_setpoints
        self.__slot_group_drone_state = slot_group_drone_state
        self.__slot_group_state_id = slot_group_state_id
        self.__slot_group_ack_id = slot_group_ack_id
        self.__target_positions = copy.deepcopy(target_positions)
        self.__ignore_message_loss = ignore_message_loss
        self.__state_feedback_trigger_dist = state_feedback_trigger_dist
        self.__trajectory_generator_options = trajectory_generator_options
        self.__pos_offset = pos_offset
        self.__use_optimized_constraints = use_optimized_constraints
        self.__current_target_positions = {}
        self.__use_high_level_planner = use_high_level_planner
        self.__high_level_setpoints = None
        self.__deadlock_breaker_agents = []
        self.__received_setpoints = None
        self.__using_intermediate_targets = False
        self.__hlp_lock = 0  # for 5 rounds, we block the hlp after the hlp has been called
        self.__current_targets_reached = {agent_id: False for agent_id in agents_ids}
        self.__agent_dodge_distance = agent_dodge_distance
        self.__age_setpoint_trajectory = 0
        self.__recalculate_setpoints = False  # there might exist cases, where we need to recalculate the setpoints immediately
        self.__agent_target_id = {}
        self.__agent_state = {}
        self.__agents_starting_states = {}
        self.__use_given_init_pos = use_given_init_pos
        self.__init_positions = init_positions
        for id in agents_ids:
            self.__agent_state[id] = np.hstack(
                (copy.deepcopy(init_positions[id]), np.zeros(2 * init_positions[id].shape[0]))).ravel()
            self.__agents_starting_states[id] = np.hstack(
                (copy.deepcopy(init_positions[id]), np.zeros(2 * init_positions[id].shape[0]))).ravel()
            self.__agent_target_id[id] = 0
            self.__current_target_positions[id] = None#copy.deepcopy(target_positions[id][0])

        self.__agents_ids = agents_ids
        self.__computing_agents_ids = computing_agents_ids
        self.__communication_delta_t = communication_delta_t

        self.__alpha_1 = alpha_1
        self.__alpha_2 = alpha_2
        self.__alpha_3 = alpha_3
        self.__alpha_4 = alpha_4

        self.__state_feedback_triggered = []

        self.__current_time = 0
        self.__prediction_horizon = prediction_horizon

        self.__remove_redundant_constraints = remove_redundant_constraints

        # communication timepoints start with the time of the next communication. (we have a delay of one due to the
        # communication) * 2, because the current position cannot be influenced anymore
        self.__communication_timepoints = \
            np.linspace(self.__communication_delta_t * 2,
                        self.__communication_delta_t * 2 + communication_delta_t * prediction_horizon,
                        prediction_horizon)
        self.__breakpoints = \
            np.linspace(0, communication_delta_t * prediction_horizon, prediction_horizon * int(communication_delta_t
                                                                                                / trajectory_generator_options.optimization_variable_sample_time+1e-7) + 1)

        # planned for the next timestep

        self.__agents_coefficients = {}
        self.__agents_coefficients_calculated = {}
        self.__agents_starting_times = {}  # times the corresponding coefficents start.
        self.__agents_prios = {}
        for id in agents_ids:
            self.__agents_coefficients[id] = \
                tg.TrajectoryCoefficients(None,
                                          False,
                                          np.zeros((prediction_horizon * int(communication_delta_t /
                                                                             trajectory_generator_options.optimization_variable_sample_time+1e-7),
                                                    init_positions[id].shape[0])))

            self.__agents_coefficients_calculated[id] = \
                tg.TrajectoryCoefficients(None,
                                          False,
                                          np.zeros((prediction_horizon * int(communication_delta_t /
                                                                             trajectory_generator_options.optimization_variable_sample_time),
                                                    init_positions[id].shape[0])))
            self.__agents_starting_times[id] = 0
            self.__agents_prios[id] = id

        self.__trajectory_interpolation = TripleIntegrator(
            breakpoints=self.__breakpoints,
            dimension_trajectory=3,
            num_states=9,
            sampling_time=trajectory_generator_options.optimization_variable_sample_time)

        self.__trajectory_generator = tg.TrajectoryGenerator(
            options=trajectory_generator_options,
            trajectory_interpolation=self.__trajectory_interpolation)

        self.__number_rounds = 0
        self.__options = trajectory_generator_options

        self.__current_agent = offset % max(len(agents_ids), 1)
        self.__scheduled_agents_indexes = [i for i in range(num_computing_agents)]

        self.__num_agents = len(agents_ids)
        self.__num_computing_agents = num_computing_agents

        self.__use_event_trigger = use_event_trigger

        # precalculate such that optimization is faster (speedup of ~ 60%)
        self.__input_trajectory_vector_matrix = []
        self.__state_trajectory_vector_matrix = []
        for i in range(prediction_horizon + 1):
            self.__input_trajectory_vector_matrix.append(
                self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
                    i * self.__communication_delta_t + self.__options.collision_constraint_sample_points,
                    derivative_order=0))

            self.__state_trajectory_vector_matrix.append(
                self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
                    i * self.__communication_delta_t + self.__options.collision_constraint_sample_points,
                    derivative_order=0))

        self.__ids_to_recalculate = []

        delay = 1
        self.__last_trajectories = {}
        for id in agents_ids:
            self.__last_trajectories[id] = np.reshape(
                self.__state_trajectory_vector_matrix[delay] @ self.__agents_starting_states[id],
                (len(self.__options.collision_constraint_sample_points), init_positions[id].shape[0]))

        self.__trajectory_tracker = InformationTracker()
        self.__last_trajectory_shape = (len(self.__options.collision_constraint_sample_points), 3)
                                        #init_positions[id].shape[0])
        for id in agents_ids:
            last_trajectory = np.reshape(
                self.__state_trajectory_vector_matrix[delay] @ self.__agents_starting_states[id],
                self.__last_trajectory_shape)

            coeff = tg.TrajectoryCoefficients(None,
                                              False,
                                              np.zeros((prediction_horizon * int(communication_delta_t /
                                                                                 trajectory_generator_options.optimization_variable_sample_time),
                                                        init_positions[id].shape[0])))

            state = np.hstack((copy.deepcopy(init_positions[id]), np.zeros(2 * init_positions[id].shape[0]))).ravel()

            trajectory_content = TrajectoryContent(coefficients=coeff, last_trajectory=last_trajectory,
                                                   init_state=None, current_state=None,
                                                   trajectory_start_time=0, id=id, trajectory_calculated_by=id)

            self.__trajectory_tracker.add_unique_information(id, trajectory_content)

        self.__last_calc_time = None
        self.__last_received_messages = {self.ID: EmtpyContent([])}
        self.__received_drone_state_messages = []

        self.__all_targets_reached = False

        self.__init_configuration = True

        self.__ack_message = None
        self.__messages_rec = {}

        self.__system_state = NORMAL
        self.__last_system_state = NORMAL

        # if the cus should decide which targets the drones have or the drones should decide this.
        self.__use_own_targets = use_own_targets

        self.__num_trajectory_messages_received = 0

        self.ordered_indexes = None
        self.prios = None
        self.starting_times = None

        self.__total_num_optimizer_runs = 0
        self.__num_succ_optimizer_runs = 0

        self.__hlp_running = False  # flag that shows if the hlp thread is running.
        self.__simulated = simulated
        self.__hlp_agent_idx = 0

        self.__downwash_scaling = np.diag([1, 1, 1.0 / float(self.__trajectory_generator_options.downwash_scaling_factor)])
        self.__age_setpoint_trajectories = {agent_id: 0 for agent_id in self.__agents_ids}

        self.__index = 0

        # field for event trigger, all agents generate a consensus of prios.
        self.__prio_consensus = []

        self.__setpoint_creator = setpoint_creator

        self.__num_trigger_times = {agent_id: 0 for agent_id in self.__agents_ids}

        self.__selected_UAVs = []

        self.__time_last_deadlock = 0

        self.__weight_band = weight_band

        self.__save_snapshot_times = save_snapshot_times
        self.__snapshot_saving_path = snapshot_saving_path
        self.__setpoint_history = []
        
        self.__simulate_quantization = simulate_quantization

    def add_new_agent(self, m_id):
        last_trajectory = None

        coeff = tg.TrajectoryCoefficients(None,
                                          False,
                                          np.zeros((self.__prediction_horizon * int(round(self.__communication_delta_t /
                                                                             self.__trajectory_generator_options.optimization_variable_sample_time)),
                                                    3)))

        trajectory_content = TrajectoryContent(coefficients=coeff, last_trajectory=last_trajectory,
                                               init_state=None, current_state=None,
                                               trajectory_start_time=0, id=m_id, trajectory_calculated_by=-1)

        self.__trajectory_tracker.add_unique_information(m_id, trajectory_content)
        self.__trajectory_tracker.set_deprecated(m_id)

        if self.__system_state == NORMAL:
            self.__system_state = INFORMATION_DEPRECATED

        self.__agents_ids.append(m_id)
        self.__agents_ids.sort()
        self.__num_agents = len(self.__agents_ids)
        self.__agents_prios[m_id] = m_id

        # first every agent should fly to the origin.
        self.__setpoint_creator.add_drone(m_id, np.array([0.0, 0.0, 1.0]))

        self.__current_target_positions[m_id] = np.array([0.0, 0.0, 1.0])

        self.__num_trigger_times[m_id] = 0

    def remove_agent(self, m_id):
        self.__trajectory_tracker.delete_information(m_id)
        self.__agents_prios.pop(m_id)
        self.__agents_ids.remove(m_id)
        self.__agents_ids.sort()

        self.__num_agents = len(self.__agents_ids)

    def add_new_computation_agent(self, m_id):
        self.__computing_agents_ids.append(m_id)
        self.__computing_agents_ids.sort()
        self.__num_computing_agents = len(self.__computing_agents_ids)
        self.__comp_agent_prio = self.__computing_agents_ids.index(self.ID)

        self.__send_setpoints = self.ID == self.__computing_agents_ids[0]

    def remove_computation_agent(self, m_id):
        if m_id not in self.__computing_agents_ids:
            return

        self.__computing_agents_ids.remove(m_id)
        self.__computing_agents_ids.sort()
        self.__num_computing_agents = len(self.__computing_agents_ids)
        self.__comp_agent_prio = self.__computing_agents_ids.index(self.ID)

        self.__send_setpoints = self.ID == self.__computing_agents_ids[0]

    def get_prio(self, slot_group_id):
        """returns the priority for the schedulder, because the system is not event triggered, it just returns zero

        Returns
        -------
            prio:
                priority
            slot_group_id:
                the slot group id the prio should be calculated to
        """
        return 1

    def get_message(self, slot_group_id):
        """returns the current message that should be send.
        If the agent is not in the slot group, it will return None

        Parameters
        ----------
            slot_group_id:
                the slot group id the message should belong to

        Returns
        -------
            message: Message
                Message
        """
        if slot_group_id == self.__slot_group_planned_trajectory_id:
            # copy because the agent might change this value during the network simulation and otherwise this new
            # value will be transmitted
            if len(self.__last_received_messages) == 0:
                return None
            copy_message = copy.deepcopy(self.__last_received_messages[self.ID])
            if isinstance(copy_message, TrajectoryMessageContent):
                copy_message.init_state[0:3] -= self.__pos_offset[copy_message.id]
                copy_message = copy_message if not self.__simulate_quantization else simulate_quantization_trajectory_message(copy_message)
            return net.Message(self.ID, slot_group_id, copy_message)

        elif slot_group_id == self.__slot_group_state_id:
            pass
        elif slot_group_id == self.__slot_group_ack_id:
            pass
        elif slot_group_id == self.__slot_group_setpoints_id:
            if self.__send_setpoints:
                setpoints = copy.deepcopy(self.__high_level_setpoints)
                if setpoints is not None:
                    for agent_id in setpoints:
                        setpoints[agent_id] -= self.__pos_offset[agent_id]
                setpoint_message = SetpointMessageContent(setpoints)
                setpoint_message = setpoint_message if not self.__simulate_quantization else simulate_quantization_setpoints_message(setpoint_message)
                return net.Message(self.ID, slot_group_id, setpoint_message)
            else:
                return None

    def send_message(self, message):
        """send message to agent.

        Parameters
        ----------
            message: Message
                message to send.
        """
        if message is None:
            return
        if message.slot_group_id == self.__slot_group_ack_id:
            self.__ack_message = copy.deepcopy(message)
        else:
            self.__messages_rec[(message.ID, message.slot_group_id)] = message
        # if the message is from leader agent set new reference point
        if message.slot_group_id == self.__slot_group_planned_trajectory_id:
            self.__num_trajectory_messages_received += 1
            if isinstance(message.content, TrajectoryMessageContent):
                message.content.init_state[0:3] += self.__pos_offset[message.content.id]

            if message.ID not in self.__last_received_messages:
                # we cannot update our information yet. We first have to compare the information tracker against
                # the metadata of the state messages. The metadata in the state messages is one round old. This means
                # that if an agent has received a new trajectory at the beginning of this round, the metadata in his
                # state message still points to its old trajectory. If we now add the new trajectory into the tracker,
                # it will be compared against the metadata of the old trajectory and thus it will be deleted.
                self.__last_received_messages[message.ID] = copy.deepcopy(message.content)
            else:
                if message.ID != self.ID:
                    print("sssssssssss")
                    print(self.__system_state)
                    print(self.__last_system_state)
                assert message.ID == self.ID

        if message.slot_group_id == self.__slot_group_drone_state:
            message.content.state[0:3] += self.__pos_offset[message.ID]
            self.print(f"Received pos from {message.ID}: {message.content.state}")
            message.content.target_position += self.__pos_offset[message.ID]
            self.__received_drone_state_messages.append(message)
            # The state is measured at the beginning of the round and extrapolated by drone
            # init drone state if it has to be init. (The state is measured at the beginning of the last round.)
            # calulate number of timesteps, the data need to be delayed
            delay_timesteps = (self.__current_time - self.__trajectory_tracker.get_information(message.ID).content[0].trajectory_start_time) / self.__options.optimization_variable_sample_time
            delay_timesteps = int(np.round(delay_timesteps, 0))  # needs to be rounded because if of float inaccuracies
            if self.__trajectory_tracker.get_information(message.ID).content[0].init_state is None:
                if self.__use_given_init_pos:
                    trajectory = self.__trajectory_tracker.get_information(message.ID).content[0]
                    trajectory.init_state = np.array([self.__init_positions[message.ID][0], self.__init_positions[message.ID][1], self.__init_positions[message.ID][2], 0, 0, 0, 0, 0, 0])
                    trajectory.current_state = copy.deepcopy(trajectory.init_state)
                    trajectory.last_trajectory = None
                    self.__agent_state[message.ID] = copy.deepcopy(trajectory.init_state)
                    self.__recalculate_setpoints = True
                    assert False
                else:
                    # because the current coefficients are zero, we can set the init state equals the current state.
                    trajectory = self.__trajectory_tracker.get_information(message.ID).content[0]
                    trajectory.init_state = copy.deepcopy(message.content.state)
                    trajectory.current_state = copy.deepcopy(message.content.state)
                    trajectory.last_trajectory = None
                    self.__agent_state[message.ID] = copy.deepcopy(message.content.state)
                    self.__recalculate_setpoints = True

            for trajectory in self.__trajectory_tracker.get_information(message.ID).content:
                if np.linalg.norm(trajectory.current_state[0:3] - message.content.state[0:3]) > self.__state_feedback_trigger_dist and False:
                    print(f"{message.ID}: {trajectory.current_state} {message.content.state[0:3]}")
                    self.__state_feedback_triggered.append(message.ID)
                    # trajectory.current_state = np.zeros(trajectory.current_state.shape)
                    trajectory.current_state[0:3] = copy.deepcopy(message.content.state[0:3])

                    self.__agent_state[message.ID] = copy.deepcopy(message.content.state)

                    # update last_trajectory, because we do not have the init state anymore as a state on the trajectory
                    # (some state in between instead), we have to do a different calculation of the last_trajecotry than
                    # in round_finished()
                    coeff = None
                    if trajectory.coefficients.valid:
                        coeff = np.zeros(trajectory.coefficients.coefficients.shape)
                        trajectory.coefficients.coefficients = coeff
                    else:
                        coeff = np.zeros(trajectory.coefficients.alternative_trajectory.shape)
                        trajectory.coefficients.alternative_trajectory = coeff
                    delay_timesteps = (self.__current_time - trajectory.trajectory_start_time) \
                                      / self.__options.optimization_variable_sample_time
                    delay_timesteps = int(np.round(delay_timesteps, 0))  # needs to be rounded because if of float inaccuracies

                    coeff_shifted = np.array([coeff[min((i + delay_timesteps, len(self.__breakpoints) - 2))]
                                              for i in range(len(self.__breakpoints) - 1)])
                    coeff_resize = np.reshape(coeff_shifted, (coeff_shifted.size,))

                    trajectory.last_trajectory = np.reshape((self.__input_trajectory_vector_matrix[0] @ coeff_resize + \
                                                             self.__state_trajectory_vector_matrix[0] @
                                                             trajectory.current_state), self.__last_trajectory_shape)

                    print("3333333333333333333")


                    # also recalculate setpoints
                    self.__recalculate_setpoints = True

            # process targets from drone
            if not self.__use_own_targets:
                if message.ID in self.__current_target_positions.keys():
                    # change target position if target changed
                    if np.any(message.content.target_position != self.__current_target_positions[message.ID]):
                        self.__current_target_positions[message.ID] = copy.deepcopy(message.content.target_position)
                        self.__agents_prios[message.ID] = message.ID  # change prio for cooperative behaviour
                        # target changed, thus recalculate setpoints
                        self.__recalculate_setpoints = True
                else:
                    self.__current_target_positions[message.ID] = copy.deepcopy(message.content.target_position)

            if not self.__ignore_message_loss:
                trajectory_information = self.__trajectory_tracker.get_information(message.ID)
                trajectory_to_change = None
                for trajectory in trajectory_information.content:
                    # this trajectory is the trajectory the
                    # UAV is flying.
                    if trajectories_equal(trajectory, message.content):
                        trajectory_to_change = trajectory
                        break

                # if we do not have any information about the trajectory, something went wrong and our information is outdated
                if trajectory_to_change is not None:
                    trajectory_information.set_unique_content(trajectory_to_change)
                else:
                    trajectory_information.set_deprecated()

        if message.slot_group_id == self.__slot_group_setpoints_id:
            self.__received_setpoints = copy.deepcopy(message.content.setpoints)
            if self.__received_setpoints is not None:
                for agent_id in self.__received_setpoints:
                    self.__received_setpoints[agent_id] += self.__pos_offset[agent_id]
                self.print(f"New setpoints: {self.__received_setpoints}")
            pass

    def __update_information_tracker(self):
        # update trajectories (leads to a faster calculation)
        delay_timesteps = self.__communication_delta_t / (self.__options.collision_constraint_sample_points[1] -
                                                          self.__options.collision_constraint_sample_points[0])
        delay_timesteps = int(np.round(delay_timesteps, 0))  # needs to be rounded because if of float inaccuracies
        for i in range(0, self.__num_agents):
            id = self.__agents_ids[i]
            for trajectory in self.__trajectory_tracker.get_information(id).content:
                if trajectory.init_state is None:
                    continue
                if trajectory.last_trajectory is None:
                    coeff = trajectory.coefficients.coefficients if trajectory.coefficients.valid \
                        else trajectory.coefficients.alternative_trajectory
                    coeff_resize = np.reshape(coeff, (coeff.size,))

                    delay = int(round((self.__current_time - trajectory.trajectory_start_time +
                                       self.__communication_delta_t) / self.__communication_delta_t))
                    if delay > self.__prediction_horizon:
                        delay = self.__prediction_horizon  # agents will stand still at the end, so we can use this
                    trajectory.last_trajectory = np.reshape((self.__input_trajectory_vector_matrix[delay] @ coeff_resize + \
                                             self.__state_trajectory_vector_matrix[delay] @
                                             trajectory.init_state), self.__last_trajectory_shape)

                else:
                    trajectory.last_trajectory = np.array(
                        [trajectory.last_trajectory[min((j + delay_timesteps, len(trajectory.last_trajectory) - 1))]
                         for j in range(len(trajectory.last_trajectory))])
        # update states
        for i in range(0, self.__num_agents):
            id = self.__agents_ids[i]
            for trajectory in self.__trajectory_tracker.get_information(id).content:
                if trajectory.init_state is None:
                    continue
                trajectory.current_state = self.__trajectory_interpolation.interpolate(
                    self.__current_time - trajectory.trajectory_start_time + self.__communication_delta_t,
                    trajectory.coefficients,
                    x0=trajectory.current_state,
                    integration_start=self.__current_time - trajectory.trajectory_start_time)
                print(f"Current_pos: {trajectory.current_state[0:3]}")

    def round_finished(self, round_nmbr=None, received_network_members_message=True):
        """this function has to be called at the end of the round to tell the agent that the communication round is
        finished"""
        self.__selected_UAVs.append(-1)
        self.__last_system_state = self.__system_state
        self.__init_configuration = False
        self.ordered_indexes = None
        self.prios = None
        self.starting_times = None
        start_time = time.time()
        if round_nmbr is not None:
            self.__current_time = round_nmbr * self.__communication_delta_t

        # if information is not unique at this point, we know that one drone was not able to verify that it got the
        # trajectory (because the acknowledge-message was lost). This means that at this point the trajectories saved
        # are no longer fullfiling all constraints and thus are deprecated.
        for key, trajectory in self.__trajectory_tracker.get_all_information().items():
            if not trajectory.is_deprecated:
                if not trajectory.is_unique:
                    trajectory.set_deprecated()

        # update data which was calculated by the other CUS in the last round
        # and/or check if (not) received information leads to deprecated
        # information
        self.__prio_consensus = []
        for message_id, trajectory in self.__last_received_messages.items():
            # the other cus has not send a cus, but somthing else. This means it has not calculated a new trajectory
            # and we can ignore it.
            if not isinstance(trajectory, TrajectoryMessageContent):
                continue
            current_state = self.__trajectory_interpolation.interpolate(
                    self.__current_time - trajectory.trajectory_start_time,
                    trajectory.coefficients,
                    x0=trajectory.init_state,
                    integration_start=0)
            new_content = TrajectoryContent(coefficients=copy.deepcopy(trajectory.coefficients),
                                            last_trajectory=None,
                                            init_state=copy.deepcopy(trajectory.init_state),
                                            current_state=copy.deepcopy(current_state),
                                            trajectory_start_time=trajectory.trajectory_start_time, id=trajectory.id,
                                            trajectory_calculated_by=trajectory.trajectory_calculated_by)
            if self.__ignore_message_loss:
                self.__trajectory_tracker.add_unique_information(trajectory.id, new_content)
            else:
                self.__trajectory_tracker.add_information(trajectory.id, new_content)

        for message_id, trajectory in self.__last_received_messages.items():
            if not (isinstance(trajectory, TrajectoryMessageContent) or isinstance(trajectory, EmtpyContent)):
                continue
            # calculate consensus:
            if len(self.__prio_consensus) == 0:
                self.__prio_consensus = copy.deepcopy(trajectory.prios)
            else:
                for i in range(len(self.__prio_consensus)):
                    # when there is message loss, then the CUs might send an
                    # empty prio
                    if i >= len(trajectory.prios):
                        break

                    # if the prio is 0, an agent was recalculated before. Then it is important that this agent will not be
                    # recalculated, thus we should set its priority to the lowest value, such that all other agents have
                    # a low prio.
                    if trajectory.prios[i] == 0:
                        self.__prio_consensus[i] = trajectory.prios[i]
                    if not self.__prio_consensus[i] == 0 and self.__prio_consensus[i] < trajectory.prios[i]:
                        self.__prio_consensus[i] = trajectory.prios[i]

        if len(self.__prio_consensus) == 0:
            self.__prio_consensus = [1 for _ in range(len(self.__agents_ids))]

        # check if a message from another CU was not received. If one was not received, we do not know which information is
        # deprecated, we thus have to check this in the next round

        if (not self.__num_trajectory_messages_received == len(self.__computing_agents_ids)) and not self.__ignore_message_loss:
            for information in self.__trajectory_tracker.get_all_information().values():
                information.set_deprecated()
            print(f"Lost messages of CU! {self.__num_trajectory_messages_received}")
        self.__num_trajectory_messages_received = 0
        assert self.__num_trajectory_messages_received <= len(self.__computing_agents_ids)

        self.__update_information_tracker()

        # until the we have no received data from the drone agents not send (we do not have the initial state)
        all_init_states_known = True
        for trajectory_information in self.__trajectory_tracker.get_all_information().values():
            for trajectory in trajectory_information.content:
                if trajectory.init_state is None:
                    all_init_states_known = False
                    break
        if not all_init_states_known:
            # self.__update_information_tracker()
            self.__messages_rec = {}
            self.__number_rounds = self.__number_rounds + 1
            self.__last_received_messages = {self.ID: EmtpyContent([])}
            self.__current_time = self.__current_time + self.__communication_delta_t
            return

        # calculate current state of all agents
        for i in range(0, self.__num_agents):
            id = self.__agents_ids[i]
            if self.__trajectory_tracker.get_information(id).is_unique and not \
                    self.__trajectory_tracker.get_information(id).is_deprecated:
                self.__agent_state[id] = self.__trajectory_tracker.get_information(id).content[0].current_state
                self.__agents_starting_times[id] = self.__trajectory_tracker.get_information(id).content[0].trajectory_start_time
            else:
                self.__agent_state[id] = None
                self.__agents_starting_times[id] = None

        # if we are in the normal state and something is deprecated, switch the state
        if self.__system_state == NORMAL and not self.__trajectory_tracker.no_information_deprecated:
            self.__system_state = INFORMATION_DEPRECATED
        if self.__trajectory_tracker.no_information_deprecated and self.__system_state != WAIT_FOR_UPDATE:
            self.__system_state = NORMAL

        if self.__system_state == NORMAL:
            if self.__use_own_targets:
                self.update_target_ids()
                self.__current_target_positions = self.get_targets()
            if len(self.__agents_ids) <= self.__comp_agent_prio:
                prios = self.calc_prio()
                self.__last_received_messages = {self.ID: EmtpyContent(prios)}
            else:
                # if an agent leaves, the prios might still be using the old swarm setup,
                # then, do calculate nothing.
                if len(self.__prio_consensus) > len(self.__agents_ids):
                    prios = self.calc_prio()
                    self.__last_received_messages = {self.ID: EmtpyContent(prios)}
                else:
                    # select next agent
                    # ordered_indexes = self.order_agents_by_priority()
                    ordered_indexes = np.argsort(-np.array(self.__prio_consensus))  # self.order_agents_by_priority()#
                    self.__current_agent = ordered_indexes[self.__comp_agent_prio]
                    current_id = self.__agents_ids[self.__current_agent]
                    self.__num_trigger_times[current_id] += 1
                    self.__selected_UAVs[-1] = current_id

                    self.ordered_indexes = ordered_indexes
                    self.starting_times = self.__agents_starting_times

                    # if the information about the agent is not unique do not calculate something, because we will calculate
                    # something wrong. Same is true if we have not received the members message.
                    # if the agent is remove but was scheduled for recalculation in the last round, do nothing.
                    if not self.__trajectory_tracker.get_information(current_id).is_unique \
                            or not received_network_members_message or not self.__agents_ids[self.__current_agent] in self.__agents_ids:
                        prios = self.calc_prio()
                        self.__last_received_messages = {self.ID: EmtpyContent(prios)}
                    else:
                        # solve optimization problem
                        calc_coeff = self.__calculate_trajectory(current_id=current_id, ordered_indexes=ordered_indexes)
                        self.__total_num_optimizer_runs += 1
                        if calc_coeff.valid:
                            self.__num_succ_optimizer_runs += 1

                        prios = self.calc_prio()
                        self.__last_received_messages = {
                            self.ID: TrajectoryMessageContent(coefficients=calc_coeff,
                                                              init_state=self.__agent_state[current_id],
                                                              trajectory_start_time=self.__current_time + self.__communication_delta_t,
                                                              trajectory_calculated_by=self.ID,
                                                              id=current_id, prios=prios)}
                        self.print('Distance to target for Agent ' + str(current_id) + ': ' + str(np.linalg.norm(
                            self.__agent_state[current_id][0:3] - self.__current_target_positions[current_id])) + " m.")
                        self.print(f"agent_state: {self.__agent_state[current_id][0:3]}, setpoint: {self.__current_target_positions[current_id]}")
                        self.print(
                            'Optimization for Agent ' + str(current_id) + ' took ' + str(time.time() - start_time) + ' s.')

        elif self.__system_state == INFORMATION_DEPRECATED:
            # send an emtpy message such that other agents know this is not a lost message
            prios = self.calc_prio()
            self.__last_received_messages = {
                self.ID: EmtpyContent(prios)}
            self.__system_state = RECOVER_INFORMATION_NOTIFY
        elif self.__system_state == RECOVER_INFORMATION_NOTIFY:
            self.__last_received_messages = {}
            for drone_id in self.__trajectory_tracker.keys:
                if self.__trajectory_tracker.get_information(drone_id).is_deprecated:
                    self.__last_received_messages = {
                        self.ID: RecoverInformationNotifyContent(cu_id=self.ID, drone_id=drone_id)}
                    break
            if len(self.__last_received_messages) == 0:
                self.__system_state = NORMAL
                prios = self.calc_prio()
                self.__last_received_messages = {self.ID: EmtpyContent(prios)}
            else:
                self.__system_state = WAIT_FOR_UPDATE
        elif self.__system_state == WAIT_FOR_UPDATE:
            # send nothing and wait for drone to send
            self.__last_received_messages = {}
            self.__system_state = RECOVER_INFORMATION_NOTIFY

        # self.print(self.__agent_state[current_id])
        # start time of the newly calculated data.
        self.__number_rounds = self.__number_rounds + 1
        self.__current_time = self.__current_time + self.__communication_delta_t

        self.__last_calc_time = time.time() - start_time

        self.__state_feedback_triggered = []

        if round_nmbr is not None:
            if round_nmbr % 5 == 0 and not self.__simulated:
                with open(f'../../experiment_measurements/num_trigger_times{self.ID}_{int(self.__alpha_1)}_{int(self.__alpha_2)}_{int(self.__alpha_3)}_{int(self.__alpha_4)}.p', 'wb') as handle:
                    pickle.dump({"num_trigger_times": self.__num_trigger_times, "selected_UAVs": self.__selected_UAVs}, handle)

        if int(round(self.__current_time / self.__communication_delta_t)) % 5 == 0 and not self.__simulated:
            with open(f'../../experiment_measurements/num_trigger_times_sim{self.ID}_{int(self.__alpha_1)}_{int(self.__alpha_2)}_{int(self.__alpha_3)}_{int(self.__alpha_4)}.p', 'wb') as handle:
                pickle.dump({"num_trigger_times": self.__num_trigger_times, "selected_UAVs": self.__selected_UAVs}, handle)

        if int(round(self.__current_time / self.__communication_delta_t)) in self.__save_snapshot_times:
            with open("../../experiment_measurements" +
                      f"/CU{self.ID}snapshot{int(self.__current_time / self.__communication_delta_t)}.p", 'wb') \
                    as out_file:
                pickle.dump(self, out_file)


        if len(self.__save_snapshot_times) > 0:
            if int(round(self.__current_time / self.__communication_delta_t)) >= self.__save_snapshot_times[0]:
                setpoints = {}
                for agent_id in self.__agents_ids:
                    setpoints[agent_id] = self.__trajectory_tracker.get_information(agent_id).content[0].current_state[0:3]
                self.__setpoint_history.append(setpoints)
            if int(round(self.__current_time / self.__communication_delta_t)) == 500:
                with open("../../experiment_measurements" +
                          f"/CU{self.ID}setpoints{int(self.__current_time / self.__communication_delta_t)}.p", 'wb') \
                        as out_file:
                    pickle.dump(self.__setpoint_history, out_file)

    def __calculate_trajectory(self, current_id, ordered_indexes):
        # calulate number of timesteps, the alternative data need to be delayed
        delay_timesteps = (self.__current_time - self.__trajectory_tracker.get_information(current_id).content[0].trajectory_start_time +
                           self.__communication_delta_t) / self.__options.optimization_variable_sample_time
        delay_timesteps = int(np.round(delay_timesteps, 0))  # needs to be rounded because if of float inaccuracies

        # calculate how many trajectories we have
        num_trajectories = 0
        for information in self.__trajectory_tracker.get_all_information().values():
            for content in information.content:
                num_trajectories += 1

        num_dynamic_trajectories = 0
        num_static_trajectories = 0
        for j in range(self.__num_agents):
            id = self.__agents_ids[ordered_indexes[j]]
            if j < self.__num_computing_agents:
                num_dynamic_trajectories += len(self.__trajectory_tracker.get_information(id).content)
            else:
                num_static_trajectories += len(self.__trajectory_tracker.get_information(id).content)

        band_weights = []
        last_trajectory_current_agent = self.__trajectory_tracker.get_information(current_id).content[0].last_trajectory

        weight_band = self.__weight_band if not self.__using_intermediate_targets else 1e-6
        # if we ignore message loss, we can use the optimized version of DMPC. If not, we have to use more conservative
        # (dynamic) constraints?
        if self.__use_optimized_constraints or True:
            dynamic_trajectories = [None for i in range(num_dynamic_trajectories)]
            dynamic_target_points = [None for i in range(num_dynamic_trajectories)]
            dynamic_coop_prio = [0 for i in range(num_dynamic_trajectories)]
            dynamic_trajectory_age = [0 for i in range(num_dynamic_trajectories)]
            static_target_points = [None for i in range(num_static_trajectories)]
            static_trajectories = [None for i in range(num_static_trajectories)]
            static_coop_prio = [0 for i in range(num_static_trajectories)]
            static_trajectory_age = [0 for i in range(num_static_trajectories)]
            index = 0
            i = 0
            for j in range(self.__num_agents):
                id = self.__agents_ids[ordered_indexes[j]]
                if j < self.__num_computing_agents:
                    for trajectory in self.__trajectory_tracker.get_information(id).content:
                        dynamic_trajectories[i] = trajectory.last_trajectory
                        dynamic_target_points[i] = self.__current_target_positions[id]
                        if j == self.__comp_agent_prio: # or i == self. ?
                            index = i
                        self.__agents_prios[id] = self.__agents_prios[id] if np.linalg.norm(
                            dynamic_target_points[i] - dynamic_trajectories[i][-1]) > 2 * self.__options.r_min else -1
                        dynamic_coop_prio[i] = self.__agents_prios[id]

                        age_trajectory = (self.__current_time - trajectory.trajectory_start_time) / (
                                    self.__options.collision_constraint_sample_points[1] -
                                    self.__options.collision_constraint_sample_points[0])
                        age_trajectory = int(
                            np.round(age_trajectory, 0))  # needs to be rounded because if of float inaccuracies
                        dynamic_trajectory_age[i] = age_trajectory

                        band_weights.append(calculate_band_weight(p_target=self.__current_target_positions[current_id],
                                                                  p_self=last_trajectory_current_agent[-1],
                                                                  p_other=trajectory.last_trajectory[-1],
                                                                  weight=weight_band))
                        i += 1
                else:
                    for trajectory in self.__trajectory_tracker.get_information(id).content:
                        static_trajectories[i - num_dynamic_trajectories] = trajectory.last_trajectory
                        static_target_points[i - num_dynamic_trajectories] = self.__current_target_positions[id]
                        self.__agents_prios[id] = self.__agents_prios[id] if np.linalg.norm(
                            static_target_points[i - num_dynamic_trajectories] -
                            static_trajectories[i - num_dynamic_trajectories][-1]) > 2 * self.__options.r_min else -1
                        static_coop_prio[i - num_dynamic_trajectories] = self.__agents_prios[id]

                        age_trajectory = (self.__current_time - trajectory.trajectory_start_time) / (
                                    self.__options.collision_constraint_sample_points[1] -
                                    self.__options.collision_constraint_sample_points[0])
                        age_trajectory = int(
                            np.round(age_trajectory, 0))  # needs to be rounded because if of float inaccuracies
                        static_trajectory_age[i - num_dynamic_trajectories] = age_trajectory

                        band_weights.append(calculate_band_weight(p_target=self.__current_target_positions[current_id],
                                                                  p_self=last_trajectory_current_agent[-1],
                                                                  p_other=trajectory.last_trajectory[-1],
                                                                  weight=weight_band))
                        i += 1

        else:
            dynamic_trajectories = [None for i in range(num_trajectories)]
            dynamic_target_points = [None for i in range(num_trajectories)]
            dynamic_coop_prio = [0 for i in range(num_trajectories)]
            static_target_points = []
            static_trajectories = []
            static_coop_prio = []
            index = 0
            i = 0

            for information in self.__trajectory_tracker.get_all_information().values():
                for trajectory in information.content:
                    dynamic_trajectories[i] = trajectory.last_trajectory
                    dynamic_target_points[i] = self.__current_target_positions[trajectory.id]
                    if trajectory.id == current_id:
                        index = i

                    self.__agents_prios[trajectory.id] = self.__agents_prios[trajectory.id] if np.linalg.norm(
                        dynamic_target_points[i] - dynamic_trajectories[i][-1]) > 2 * self.__options.r_min else -1
                    dynamic_coop_prio[i] = self.__agents_prios[trajectory.id]
                    i += 1

        if self.__trajectory_tracker.get_information(current_id).content[0].coefficients.valid:
            previous_solution_shifted = self.__trajectory_tracker.get_information(current_id).content[0].coefficients.coefficients
            previous_solution_shifted = np.array(
                [previous_solution_shifted[min((i + delay_timesteps, len(self.__breakpoints) - 2))]
                 for i in range(len(self.__breakpoints) - 1)])
        else:
            previous_solution_shifted = self.__trajectory_tracker.get_information(current_id).content[0].coefficients.alternative_trajectory
            previous_solution_shifted = np.array(
                [previous_solution_shifted[min((i + delay_timesteps, len(self.__breakpoints) - 2))]
                 for i in range(len(self.__breakpoints) - 1)])

        # check if there exists a high level setpoint. If an agent is newly added to the swarm, there might exist no
        # high level setpoint for it.
        high_level_setpoint = None
        if self.__received_setpoints is not None:
            if current_id in self.__received_setpoints:
                high_level_setpoint = self.__received_setpoints[current_id]
        return self.__trajectory_generator.calculate_trajectory(
            current_id=current_id,
            current_state=self.__agent_state[current_id],
            target_position=self.__current_target_positions[current_id],
            index=index,
            dynamic_trajectories=np.array(dynamic_trajectories),
            static_trajectories=np.array(static_trajectories),
            optimize_constraints=self.__remove_redundant_constraints,
            previous_solution=previous_solution_shifted,
            dynamic_target_points=dynamic_target_points,
            static_target_points=static_target_points,
            dynamic_coop_prio=dynamic_coop_prio,
            static_coop_prio=static_coop_prio,
            band_weights=band_weights,
            dynamic_trajectory_age=dynamic_trajectory_age,
            static_trajectory_age=static_trajectory_age,
            use_nonlinear_mpc=False,
            high_level_setpoints=high_level_setpoint #None if self.__high_level_setpoints is None else self.__high_level_setpoints[current_id]
        )

    def get_targets(self):
        """
        Returns the current target position of each agent

        Parameters
        ----------
            agent_state:
                state of the agents
            agents_starting_times:
                the time the agents were replanned last
            ordered_indexes:
                priority based order of ondexes. The first M agents will be scheduled
        """
        if self.__use_own_targets:
            setpoints, self.__recalculate_setpoints = self.__setpoint_creator.next_setpoints(self.__agent_state, round_nmbr=round(self.__current_time / self.__communication_delta_t))
            return setpoints
            """
            targets = {}
            for id in self.__agents_ids:
                targets[id] = self.__target_positions[id][self.__agent_target_id[id]]
                # self.print("AGENT NO: " + str(id) + " TARGET NO: " + str(self.__agent_target_id[id]))
            return targets
            """
        else:
            return self.__current_target_positions

    def order_agents_by_priority(self):
        prios = self.calc_prio()
        self.prios = prios
        return np.argsort(-prios)

    def select_next_agent(self):
        """ selects the next agent to update its data
        Returns:
            next_agent: int
                number of the agent to update the data for"""

        prios = self.calc_prio()
        slot_no = self.ID - self.__num_agents  # better way?
        next_agent_prios = sorted(prios)[-self.__num_computing_agents:]
        next_agents = []
        for next_agent_prio in next_agent_prios:
            next_agent = prios.tolist().index(next_agent_prio)
            prios[
                next_agent] = -999  # dirty fix to prevent that the same agent is chosen by the computing nodes if multiple have the same priority
            next_agents.append(next_agent)
        return next_agents[slot_no]

    def update_target_ids(self):
        """ updates the target IDs of the drones. If a drone has multiple targets it should fly to, the next target
        is selected, if the previous one is reached """
        return
        threshold_dist = 0.001  # m
        targets = self.get_targets()
        all_drones_reached_target = True
        all_drones_reached_current_target = True
        for id in self.__agents_ids:
            if self.__agent_state[id] is not None and self.__simulated:
                dist_to_target = np.linalg.norm(targets[id] - self.__agent_state[id][0:3])
            else:
                dist_to_target = np.linalg.norm(targets[id] -
                                                self.__trajectory_tracker.get_information(id).content[0].init_state[0:3])
            if dist_to_target < threshold_dist:
                #self.print("AGENT " + str(id) + " REACHED TARGET " + str(self.__agent_target_id[id]))
                #self.print(targets[id])
                pass
            else:
                all_drones_reached_current_target = False
                all_drones_reached_target = False

        if all_drones_reached_current_target:
            for id in self.__agents_ids:
                if self.__agent_target_id[id] < len(self.__target_positions[id]) - 1:
                    self.__agent_target_id[id] += 1
                    all_drones_reached_target = False
                    self.__recalculate_setpoints = True
                elif self.__agent_target_id[id] == (len(self.__target_positions[id]) - 1):
                    pass  # placeholder for action if drone reached its final target
                self.__agents_prios[id] = id

        self.__all_targets_reached = all_drones_reached_target

    def calc_prio(self):
        """ calculates the prio for each agent """
        state_feeback_triggered_prio = 10000
        quantization_bit_number = 8
        max_time = self.__prediction_horizon * self.__communication_delta_t

        cone_angle = 60.0 * math.pi / 180

        prios = np.zeros((self.__num_agents,))
        for i in range(0, self.__num_agents):
            own_id = self.__agents_ids[i]
            max_dist = np.linalg.norm(self.__options.max_position[own_id] - self.__options.min_position[own_id])
            target_pos = self.get_targets()[own_id]
            if self.__received_setpoints is not None:
                if own_id in self.__received_setpoints:
                    target_pos = self.__received_setpoints[own_id]
            d_target = target_pos - copy.deepcopy(self.__trajectory_tracker.get_information(own_id).content[0].current_state[0:3])

            dist_to_target = np.linalg.norm(d_target)

            prios[i] /= (1.0 * self.__num_agents * max_dist)  # normalize prio

            max_starting_time = 0
            for j in range(len(self.__trajectory_tracker.get_information(own_id).content)):
                if max_starting_time < self.__trajectory_tracker.get_information(own_id).content[j].trajectory_start_time:
                    max_starting_time = self.__trajectory_tracker.get_information(own_id).content[j].trajectory_start_time

            prios[i] += self.__alpha_2 * (self.__current_time - max_starting_time) / max_time #self.__agents_starting_times[own_id]) / max_time  # 0.1
            prios[i] += self.__alpha_1 * dist_to_target / max_dist * (self.__current_time - max_starting_time) / max_time
            prios[i] += self.__alpha_3 * dist_to_target / max_dist

            if own_id in self.__state_feedback_triggered:
                prios[i] += state_feeback_triggered_prio

            if i in np.argsort(-np.array(self.__prio_consensus))[0:len(self.__computing_agents_ids)]:
                prios[i] += -100000000 * self.__alpha_4


        # quantize prios
        for i in range(len(prios)):
            if prios[i] >= state_feeback_triggered_prio:
                prios[i] = int(2**quantization_bit_number - 1)
            elif prios[i] < 0:
                prios[i] = int(0)
            else:
                prios[i] = int(round(prios[i] / (self.__alpha_1 + self.__alpha_2 + self.__alpha_3 + self.__alpha_4)*(2**quantization_bit_number-1)))
                if prios[i] > 2**quantization_bit_number-2:
                    prios[i] = int(2 ** quantization_bit_number - 2)
        # print("zzzzzzzzzzzzzzz")
        # print(prios)
        return prios

    def print(self, text):
        print("[" + str(self.ID) + "]: " + str(text))

    def deadlock_breaker_condition(self, agent_id, other_agent_id, check_time=False):
        # self.print("000000111111111111111111111111")
        # self.print(self.__current_time - self.__time_last_deadlock)
        if self.__current_time - self.__time_last_deadlock <= 4 and check_time:   # only check time, if we are already in the deadlock breaker
            # print(".........................................................")
            return True

        if other_agent_id == agent_id:
            return False
        current_pos = {}
        trajectory = self.__trajectory_tracker.get_information(agent_id).content[0]
        current_pos[agent_id] = trajectory.current_state[0:3]
        trajectory = self.__trajectory_tracker.get_information(other_agent_id).content[0]
        current_pos[other_agent_id] = trajectory.current_state[0:3]

        own_dist_to_target = np.linalg.norm(current_pos[agent_id] - self.__current_target_positions[agent_id])
        other_agent_dist_to_target = np.linalg.norm(
            current_pos[other_agent_id] - self.__current_target_positions[other_agent_id])
        if other_agent_dist_to_target < self.__options.r_min * 0.9 and other_agent_dist_to_target < own_dist_to_target:
            return False
        elif other_agent_dist_to_target > own_dist_to_target:
            if self.__trajectory_tracker.get_information(other_agent_id).content[0].current_state is not None:
                #if self.__scaled_norm(
                #        current_pos[other_agent_id][0:2] - current_pos[agent_id][0:2]) < self.__options.r_min * 1.05:
                if np.linalg.norm(
                        current_pos[other_agent_id][0:2] - current_pos[agent_id][0:2]) < self.__options.r_min * 2.0:
                    if np.dot(
                            self.__trajectory_tracker.get_information(other_agent_id).content[0].current_state[3:6],
                            current_pos[agent_id] - current_pos[other_agent_id]) > 0 or np.dot(
                        self.__current_target_positions[other_agent_id] -
                        self.__trajectory_tracker.get_information(other_agent_id).content[0].current_state[0:3],
                            current_pos[agent_id] - current_pos[other_agent_id]) > 0:
                        return True
                    if np.dot(self.__current_target_positions[other_agent_id][0:3] - current_pos[other_agent_id][0:3],
                              current_pos[agent_id][0:3] - current_pos[other_agent_id][0:3]) > 0:
                        return True
        return False

    def round_started(self):
        print("round started!")
        if not self.__use_high_level_planner:
            return

        if self.__system_state != NORMAL:
            return

        # sometimes it happens (at the beginning of the expoeriment, when the drones stand still and send their current
        # position to the drone agents, the high level planner is in the wrong state)
        if len(self.__deadlock_breaker_agents) == 0:
            self.__using_intermediate_targets = False

        # if the CU is not the one calculating the high level targets, do not calculate them.
        if not self.__send_setpoints: #and self.__simulated:
            return
        # get the current position of all agents.
        current_pos = {}
        for agent_id in self.__agents_ids:
            if len(self.__trajectory_tracker.get_information(agent_id).content) != 0:
                trajectory = self.__trajectory_tracker.get_information(agent_id).content[0]
                if trajectory.current_state is not None:
                    current_pos[agent_id] = trajectory.current_state[0:3]
                else:
                    current_pos[agent_id] = self.__current_target_positions[agent_id]

        # check if we already know the target positions. If we do not know all (one drones has not sent it),
        # do not calculate
        for agent_id in self.__agents_ids:
            if self.__current_target_positions[agent_id] is None or current_pos[agent_id] is None:
                return

        # if we do not block the hlp, after it has been called, it will be called multiple times in a row,
        # because the drones need a while to move and it will otherwise think, the drones are in a deadlock again
        self.__hlp_lock += 1
        if self.__hlp_lock < 10 and not self.__recalculate_setpoints:
            return

        # check if all agents are close to target, then do nothing
        all_agents_close_to_target = True
        for agent_id in self.__agents_ids:
            if np.linalg.norm(
                    current_pos[agent_id] - self.__current_target_positions[agent_id]) > self.__options.r_min * 0.9:
                all_agents_close_to_target = False
        if all_agents_close_to_target:
            return

        if self.__recalculate_setpoints:
            self.__current_targets_reached = {agent_id: False for agent_id in self.__agents_ids}

        # check if agents are in a deadlock. If they are, either try to break the deadlock (or if we already trying
        # to break the deadlock go back to the original targets).
        all_agents_in_deadlock = True
        for agent_id in self.__agents_ids:
            if np.linalg.norm(self.__trajectory_tracker.get_information(agent_id).content[0].current_state[3:6]) > 1e-1:
                all_agents_in_deadlock = False
                break

        # check if not all deadlock breaker conditions are fulfilled, if not, then the deadlock is broken and we can
        # go back to the original targets.
        deadlock_broken = False
        if len(self.__deadlock_breaker_agents) > 0:
            deadlock_broken = True
            for dba in self.__deadlock_breaker_agents:
                if self.deadlock_breaker_condition(dba[0], dba[1], check_time=True):
                    deadlock_broken = False
                    break

        # if we now need to change targets, do it.
        if all_agents_in_deadlock or self.__recalculate_setpoints or deadlock_broken:
            # if we currently try to break a deadlock or received new setppoints, use real targets as "intermediate"
            # targets
            if self.__using_intermediate_targets or self.__recalculate_setpoints:
                self.__recalculate_setpoints = False
                self.__high_level_setpoints = {}
                for agent_id in self.__agents_ids:
                    self.__high_level_setpoints[agent_id] = self.__current_target_positions[agent_id]
                self.__using_intermediate_targets = 0
                if not self.__simulated and False:
                    self.__received_setpoints = copy.deepcopy(self.__high_level_setpoints)
                self.__deadlock_breaker_agents = []
                # do not calculate new intermediate setpoints
                self.__hlp_lock = 0
                return
            else:
                self.__using_intermediate_targets = True
            self.__hlp_lock = 0
            print("Deadlock!!!!")
        else:
            return

        # calculate intermediate targets based on a similar algorithm given in Park et al. 2020
        self.__high_level_setpoints = {agent_id: self.__current_target_positions[agent_id] for agent_id in self.__agents_ids}
        self.__deadlock_breaker_agents = []

        for agent_id in self.__agents_ids:
            # first determine, which agent, the agent_id has to dodge.
            agents_to_dodge = []
            for other_agent_id in self.__agents_ids:
                if other_agent_id == agent_id:
                    continue
                if self.deadlock_breaker_condition(agent_id, other_agent_id):
                    self.__deadlock_breaker_agents.append((agent_id, other_agent_id))
                    agents_to_dodge.append(other_agent_id)

            if len(agents_to_dodge) != 0:
                # first determine the closest of the agents, we need to dodge and then dodge only this one.
                closest_agent = agents_to_dodge[0]
                closest_distance = self.__scaled_norm(current_pos[closest_agent] - current_pos[agent_id])
                for other_agent_id in agents_to_dodge:
                    if other_agent_id == agent_id:
                        continue
                    other_agent_distance = self.__scaled_norm(current_pos[other_agent_id] - current_pos[agent_id])
                    if closest_distance > other_agent_distance:
                        closest_agent = other_agent_id
                        closest_distance = other_agent_distance
                # if an agent is closer to hlp_threshold_distance, use some sort of potential function approach
                hlp_threshold_distance = 0.7
                if closest_distance < hlp_threshold_distance:
                    to_target = self.__current_target_positions[closest_agent] - current_pos[closest_agent]  # self.__trajectory_tracker.get_information(closest_agent).content[0].current_state[3:6]
                    c = np.cross(current_pos[agent_id] - current_pos[closest_agent], to_target)
                    dodge_direction = current_pos[agent_id] - current_pos[closest_agent]
                    dodge_direction /= np.linalg.norm(dodge_direction)
                    # add some random noise. This helps breacking up deadlocks caused by symmetry.
                    self.__high_level_setpoints[agent_id] = current_pos[agent_id] + dodge_direction * self.__agent_dodge_distance + np.random.randn(3) * 0.1

    def __scaled_norm(self, vector):
        return np.linalg.norm(self.__downwash_scaling@vector)

    def reset_setup(self, ):
        """ resets the setup: Drone positions are set to their init positions and target positions are set to the 0th
        target """

        self.__agents_coefficients = {}
        self.__agents_coefficients_calculated = {}
        self.__agents_starting_times = {}  # times the corresponding coefficents start.
        self.__agents_prios = {}

        self.print("RESETTING")
        for id in self.__agents_ids:
            self.__agent_state[id] = np.hstack(
                (copy.deepcopy(self.__init_positions[id]), np.zeros(2 * self.__init_positions[id].shape[0]))).ravel()
            self.__agents_starting_states[id] = np.hstack(
                (copy.deepcopy(self.__init_positions[id]), np.zeros(2 * self.__init_positions[id].shape[0]))).ravel()
            self.__agent_target_id[id] = 0

            self.__agents_coefficients[id] = \
                tg.TrajectoryCoefficients(None,
                                          False,
                                          np.zeros((self.__prediction_horizon * int(self.__communication_delta_t /
                                                                                    self.__options.optimization_variable_sample_time),
                                                    self.__init_positions[id].shape[0])))

            self.__agents_coefficients_calculated[id] = \
                tg.TrajectoryCoefficients(None,
                                          False,
                                          np.zeros((self.__prediction_horizon * int(self.__communication_delta_t /
                                                                                    self.__options.optimization_variable_sample_time),
                                                    self.__init_positions[id].shape[0])))
            self.__agents_starting_times[id] = 0
            self.__agents_prios[id] = id

        self.__number_rounds = 0
        self.__all_targets_reached = False
        self.__init_configuration = True

    @property
    def init_configuration(self):
        """ return, whether the setup it in its initial configuration (no new trajectories etc.) """
        return self.__init_configuration

    @property
    def last_num_constraints(self):
        return self.__trajectory_generator.last_num_constraints

    @property
    def last_calc_time(self):
        return self.__last_calc_time

    @property
    def current_intermediate_target(self):
        return self.__high_level_setpoints

    @property
    def target_positions(self):

        return self.__target_positions

    @target_positions.setter
    def target_positions(self, target_positions):
        # reset prios, because target positions got changed
        for id in self.__agents_ids:
            self.__agents_prios[id] = id
        self.__target_positions = target_positions

    @property
    def comp_agent_prio(self):
        return self.__comp_agent_prio

    @property
    def current_time(self):
        return self.__current_time

    @property
    def num_computing_agents(self):
        return self.__num_computing_agents

    @property
    def agent_ids(self):
        return self.__agents_ids

    @property
    def all_targets_reached(self):
        return self.__all_targets_reached

    @property
    def agent_state(self):
        return self.__agent_state

    @property
    def last_trajectories(self):
        return self.__last_trajectories

    @property
    def current_target_positions(self):
        return self.__current_target_positions

    @property
    def total_num_optimizer_runs(self):
        return self.__total_num_optimizer_runs

    @property
    def num_succ_optimizer_runs(self):
        return self.__num_succ_optimizer_runs

    def get_trajectory_tracker(self):
        return self.__trajectory_tracker

    def get_agent_position(self, agent_id):
        return self.__trajectory_tracker.get_information(agent_id).content[0].current_state[0:3]

    def get_agent_input(self, agent_id):
        return self.__trajectory_tracker.get_information(agent_id).content[0].coefficients.coefficients

    def set_simulate_quantization(self, value):
        self.__simulate_quantization = value

    def set_current_time(self, value):
        self.__current_time = value

    def set_save_snapshot_times(self, value):
        self.__save_snapshot_times = value


class DemoSetpointGenerator:
    def __init__(self, radius=1.3, angle_speed=2*math.pi/8):
        self.__offset = random.random()*math.pi*2
        self.__radius = radius
        self.__angle_speed = angle_speed
        self.__angle = 0

    def next_setpoint(self, dt):
        self.__angle += dt*self.__angle_speed
        return np.array([math.cos(self.__offset + self.__angle)*self.__radius + 2,
                         math.sin(self.__offset + self.__angle)*self.__radius + 2, 1])


class RemoteDroneAgent(net.Agent):
    """this class represents a drone agent inside the network.
    Every agent is allowed to send in multiple slot groups. This is needed if e.g. the agent should send his sensor
    measurements but also if he is in error mode. This agent receives a new reference data and follows it.

    Methods
    -------
    get_prio(slot_group_id):
        returns current priority
    get_message(slot_group_id):
        returns message
    send_message(message):
        send a message to this agent
    round_finished():
        callback for the network to signal that a round has finished
    """

    def __init__(self, ID, slot_group_planned_trajectory_id, init_position, target_position, communication_delta_t,
                 trajectory_generator_options, prediction_horizon, other_drones_ids, target_positions, order_interpolation=4,
                 slot_group_state_id=None, slot_group_ack_id=100000, state_feedback_trigger_dist=0.3,
                 use_demo_setpoints=True, load_cus_round_nmbr=0,
                 trajectory_start_time=0,
                 trajectory_cu_id=-1):
        """

        Parameters
        ----------
            ID:
                identification number of agent
            slot_group_planned_trajectory_id:
                ID of the slot group which is used
            init_position: np.array
                3D position of agent
            trajectory_generator_options: tg.TrajectoryGeneratorOptions
                options for data generator
            communication_delta_t: float
                time difference between two communication rounds
            prediction_horizon: int
                prediction horizon of optimizer
            order_interpolation: int
                order of interpolation
            slot_group_state_id: int
                id of slot group needed to send state

        """
        super().__init__(ID, [slot_group_state_id, slot_group_planned_trajectory_id])
        self.__slot_group_planned_trajectory_id = slot_group_planned_trajectory_id
        self.__slot_group_state_id = slot_group_state_id
        self.__slot_group_ack_id = slot_group_ack_id
        self.__dim = init_position.shape[0]
        self.__traj_state = np.hstack((copy.deepcopy(init_position), np.zeros(
            2 * init_position.shape[0]))).ravel()
        self.__position = copy.deepcopy(init_position)
        self.__target_position = np.copy(target_position)
        self.__target_positions = np.copy(target_positions)
        self.__next_pos_trajectory = np.copy(target_position)
        self.__init_state = self.__traj_state
        self.__other_drones_ids = copy.deepcopy(other_drones_ids)
        self.__state_feedback_trigger_dist = state_feedback_trigger_dist
        self.__transition_time = None
        self.__target_reached = False
        self.__crit_distance_to_target = 0.2  # m
        self.__communication_delta_t = communication_delta_t
        self.__crashed = False

        self.__current_time = load_cus_round_nmbr * self.__communication_delta_t
        self.__planned_trajectory_start_time = trajectory_start_time
        self.__prediction_horizon = prediction_horizon

        # communication timepoints start with the time of the next communication. (we have a delay of one due to the
        # communication) * 2, because the current position cannot be influenced anymore
        self.__communication_timepoints = \
            np.linspace(self.__communication_delta_t * 2,
                        self.__communication_delta_t + communication_delta_t * prediction_horizon,
                        prediction_horizon)

        # breakpoints of optimization variable
        self.__breakpoints = \
            np.linspace(0, communication_delta_t * prediction_horizon, prediction_horizon * int(
                communication_delta_t / trajectory_generator_options.optimization_variable_sample_time) + 1)

        # planned for the next timestep
        self.__planned_trajectory_coefficients = tg.TrajectoryCoefficients(None, False,
                                                                           np.zeros((
                                                                               int(round(prediction_horizon * communication_delta_t /
                                                                                   trajectory_generator_options.optimization_variable_sample_time)),
                                                                               self.__dim)))

        self.__trajectory_interpolation = TripleIntegrator(
            breakpoints=self.__breakpoints,
            dimension_trajectory=3,
            num_states=9,
            sampling_time=trajectory_generator_options.optimization_variable_sample_time)

        self.__number_rounds = 0
        self.__options = trajectory_generator_options

        self.__updated = True

        self.__state = np.array([init_position[0], init_position[1], init_position[2],
                                 0, 0, 0, 0, 0, 0])

        self.__current_trajectory_calculated_by = trajectory_cu_id  #ID

        self.__send_trajectory_message_to = []

        self.__received_state_messages = []

        self.__agent_target_idx = 0

        self.__all_targets_reached = False

        self.__state_measured = False

        self.__current_setpoint = self.__init_state

        self.__use_demo_setpoints = use_demo_setpoints
        self.__demo_setpoint_generator = DemoSetpointGenerator()

    def get_prio(self, slot_group_id):
        """returns the priority for the scheulder, because the system is not event triggered, it just returns zero

        Returns
        -------
            prio:
                priority
            slot_group_id:
                the slot group id the prio should be calculated to
        """
        return 0

    def get_message(self, slot_group_id):
        if slot_group_id == self.__slot_group_state_id:
            setpoint = self.__demo_setpoint_generator.next_setpoint(self.__communication_delta_t)
            message = net.Message(self.ID, slot_group_id,
                                  StateMessageContent(self.__traj_state,
                                                      target_position=self.__target_positions[self.__agent_target_idx] if not self.__use_demo_setpoints else setpoint,
                                                      target_position_idx=self.__agent_target_idx,
                                                      trajectory_start_time=self.__planned_trajectory_start_time,
                                                      trajectory_calculated_by=self.__current_trajectory_calculated_by,
                                                      coefficients=self.__planned_trajectory_coefficients,
                                                      init_state=self.__init_state))
            self.__received_state_messages = [message.content]
            return message
        if slot_group_id == self.__slot_group_planned_trajectory_id:
            message = []
            for ID in self.__send_trajectory_message_to:
                message.append(net.Message(ID, slot_group_id,
                                      TrajectoryMessageContent(id=self.ID,
                                                               coefficients=self.__planned_trajectory_coefficients,
                                                               trajectory_start_time=self.__planned_trajectory_start_time,
                                                               init_state=self.__init_state,
                                                               trajectory_calculated_by=self.__current_trajectory_calculated_by,
                                                               prios=np.array([255 for _ in range(len(self.__other_drones_ids))]))))
            self.__send_trajectory_message_to = []
            return message

    def send_message(self, message):
        """send message to agent.

        Parameters
        ----------
            message: Message
                message to send.
        """
        if message is None:
            return
        # if the message is from leader agent set new reference point
        if message.slot_group_id == self.__slot_group_planned_trajectory_id:
            if isinstance(message.content, TrajectoryMessageContent):
                # what if two CUs send a trajectory for the same UAV?
                if message.content.id == self.ID:
                    update_data = True
                    if self.__updated:
                        if self.__planned_trajectory_start_time > message.content.trajectory_start_time:
                            update_data = False
                        elif not self.__current_trajectory_calculated_by < message.content.trajectory_calculated_by:
                            update_data = False
                    self.__updated = True
                    if update_data:
                        self.__planned_trajectory_coefficients = copy.deepcopy(message.content.coefficients)
                        self.__planned_trajectory_start_time = copy.deepcopy(message.content.trajectory_start_time)
                        self.__updated = True
                        self.__traj_state = self.__trajectory_interpolation.interpolate(
                                        self.__current_time - self.__planned_trajectory_start_time,
                                        self.__planned_trajectory_coefficients, integration_start=0,
                                        x0=message.content.init_state)

                        self.__init_state = copy.deepcopy(message.content.init_state)
                        self.__current_trajectory_calculated_by = message.content.trajectory_calculated_by
            if isinstance(message.content, RecoverInformationNotifyContent):
                if message.content.drone_id == self.ID:
                    self.__send_trajectory_message_to.append(message.content.cu_id)

        elif message.slot_group_id == self.__slot_group_state_id:
            if message.ID != self.ID:
                self.__received_state_messages.append(message.content)
                # otherwise drones might not be synchronised regarding targets
                if message.content.target_position_idx > self.__agent_target_idx:
                    self.__agent_target_idx = message.content.target_position_idx
                    if self.__agent_target_idx == (len(self.__target_positions) - 1):
                        self.__all_targets_reached = True

        # we dont care if someone lost his state
        elif message.slot_group_id == self.__slot_group_ack_id:
            pass

    def round_finished(self):
        """this function has to be called at the end of the round to tell the agent that the communication round is
        finished"""
        self.__state_measured = False
        self.__updated = False
        if len(self.__received_state_messages) == len(self.__other_drones_ids):
            all_drones_reached_current_target = True
            for m in self.__received_state_messages:
                if np.linalg.norm(m.state[0:3] - m.target_position) > 0.05:
                    all_drones_reached_current_target = False
                    break
            if all_drones_reached_current_target:
                if self.__agent_target_idx < len(self.__target_positions) - 1:
                    self.__agent_target_idx += 1
                elif self.__agent_target_idx == (len(self.__target_positions) - 1):
                    self.__all_targets_reached = True
        self.__received_state_messages = []

        # check if trajectory deviates too much from the trajectory received
        if np.linalg.norm(self.__traj_state[0:3] - self.__state[0:3]) > self.__state_feedback_trigger_dist:
            self.__traj_state[0:3] = self.__state[0:3]
            #self.__traj_state[3:] = 0
            if self.__planned_trajectory_coefficients.valid:
                self.__planned_trajectory_coefficients.coefficients = np.zeros(self.__planned_trajectory_coefficients.coefficients.shape)
            else:
                self.__planned_trajectory_coefficients.alternative_trajectory = np.zeros(
                    self.__planned_trajectory_coefficients.alternative_trajectory.shape)

    def print(self, text):
        print("[" + str(self.ID) + "]: " + str(text))

    def round_started(self):
        pass

    def next_planned_state(self, delta_t):
        """
        Parameters
        ----------
            delta_t: float
                time between the last and this call of the function or time between the call of this function
                and the last call of round_started()
        Returns
        -------
            next_planned_state: int
                state for next position controller
        """
        if self.__crashed and False:
            self.__traj_state[self.__dim] = 0
            self.__current_time = self.__current_time + delta_t
            self.__current_setpoint = self.__traj_state
            return self.__traj_state
        if self.__planned_trajectory_start_time > 0:
            self.__traj_state = self.__trajectory_interpolation.interpolate(
                self.__current_time - self.__planned_trajectory_start_time + delta_t,
                self.__planned_trajectory_coefficients, integration_start=self.__current_time -
                                                                          self.__planned_trajectory_start_time,
                x0=self.__traj_state)

            self.__current_time = self.__current_time + delta_t

            self.__current_setpoint = self.__traj_state
            return self.__traj_state
        self.__current_time = self.__current_time + delta_t
        self.__current_setpoint = self.__init_state
        return self.__init_state

    def get_planned_trajectory(self, resolution):
        traj = []
        if self.__planned_trajectory_start_time <= 0:
            return traj
        for i in range(int(self.__prediction_horizon*self.__communication_delta_t / resolution)):
            state = self.__trajectory_interpolation.interpolate(
                self.__current_time - self.__planned_trajectory_start_time + resolution*i,
                self.__planned_trajectory_coefficients, integration_start=self.__current_time -
                                                                          self.__planned_trajectory_start_time,
                x0=self.__traj_state)
            traj.append(state[0:3])

        return traj

    @property
    def position(self):
        """returns current measured positions

        Returns
        -------
        positions: np.array
            3D position of drone
        """
        return self.__position

    @position.setter
    def position(self, pos):
        """sets current measured position of drone"""
        self.__position = copy.deepcopy(pos)

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, state):
        if not self.__state_measured:
            self.__state = self.__trajectory_interpolation.interpolate(
                self.__current_time - self.__planned_trajectory_start_time + self.__communication_delta_t,
                self.__planned_trajectory_coefficients,
                x0=np.array([state[0], state[1], state[2],
                             self.__traj_state[3], self.__traj_state[4], self.__traj_state[5],
                             self.__traj_state[6], self.__traj_state[7], self.__traj_state[8]]),
                integration_start=self.__current_time - self.__planned_trajectory_start_time)
            self.__state_measured = True

    @property
    def current_setpoint(self):
        return self.__current_setpoint

    @property
    def x_ref(self):
        """
        returns current reference position

        Returns
        -------
        x_ref: np.array
            3D reference position of agent
        """
        return self.__x_ref

    @x_ref.setter
    def x_ref(self, new_x_ref):
        """sets new reference value.
        If the id of the drone is not the id of the leader. This function does nothing."""
        if self.ID == 0:
            self.__x_ref = new_x_ref

    @property
    def target_position(self):
        """
        returns target position
        """
        return self.__target_position

    @target_position.setter
    def target_position(self, new_target_position):
        self.__target_position = new_target_position

    @property
    def target_reached(self):
        return self.__target_reached

    @target_reached.setter
    def target_reached(self, new_target_reached_state):
        self.__target_reached = new_target_reached_state

    @property
    def transition_time(self):
        return self.__transition_time

    @transition_time.setter
    def transition_time(self, new_transition_time):
        """ sets the transition time"""
        self.__transition_time = new_transition_time

    @property
    def current_time(self):
        """ returns the current time of the drone"""
        return self.__current_time

    @property
    def traj_state(self):
        """ returns current state of the data"""
        return self.__traj_state

    @property
    def crashed(self):
        """ returns the crashed status of the drone"""
        return self.__crashed

    @crashed.setter
    def crashed(self, new_crashed_status):
        """ sets the crashed status of the drone"""
        self.__crashed = new_crashed_status

    @property
    def all_targets_reached(self):
        return self.__all_targets_reached

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))