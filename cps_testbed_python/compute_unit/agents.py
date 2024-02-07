import math
import os

import yaml

# from compute_unit.neural_networks.jax_models import AMPCNN
from compute_unit.trajectory_generation.interpolation import TrajectoryCoefficients
from network import network as net
import compute_unit.trajectory_generation.trajectory_generation as tg

from compute_unit.trajectory_generation.statespace_model import TripleIntegrator
import copy
import numpy as np
from dataclasses import dataclass
from compute_unit.trajectory_generation.information_tracker import InformationTracker, TrajectoryContent
import time

from compute_unit import real_compute_unit

import random

import pickle

import cu_animator as cu_animator

from datetime import datetime

# import jax
# import jax.numpy as jnp

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
    trajectory_calculated_by: int  # stamp of trajectory the UAV is currently following
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
            old = copy.deepcopy(message_quant.setpoints[k][i])
            message_quant.setpoints[k][i] = real_compute_unit.dequantize_pos(
                real_compute_unit.quantize_pos(message.setpoints[k][i]))
    return message_quant


def simulate_quantization_trajectory_message(message):
    message_quant = copy.deepcopy(message)
    coefficients = message.coefficients.coefficients if message.coefficients.valid else message.coefficients.alternative_trajectory
    for i in range(len(coefficients)):
        for j in range(3):
            if message_quant.coefficients.valid:
                message_quant.coefficients.coefficients[i][j] = real_compute_unit.dequantize_input(
                    real_compute_unit.quantize_input(coefficients[i][j]))
            else:
                message_quant.coefficients.alternative_trajectory[i][j] = real_compute_unit.dequantize_input(
                    real_compute_unit.quantize_input(coefficients[i][j]))
    for j in range(3):
        message_quant.init_state[j] = real_compute_unit.dequantize_pos(
            real_compute_unit.quantize_pos(message.init_state[j]))

    for j in range(3, 6):
        message_quant.init_state[j] = real_compute_unit.dequantize_vel(
            real_compute_unit.quantize_vel(message.init_state[j]))

    for j in range(6, 9):
        message_quant.init_state[j] = real_compute_unit.dequantize_acc(
            real_compute_unit.quantize_acc(message.init_state[j]))

    # for j in range(0, 3):
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
    return weight_mult * weight * math.exp(math.sin(angle) * weight_angle)


def normalize(data, normalization_params):
    return (data - normalization_params[0]) / normalization_params[1]


def denormalize(data, normalization_params):
    return data * normalization_params[1] + normalization_params[0]


def call_ampc(x, model, normalization):
    x = normalize(x, normalization["x"])
    u = model(x)

    u = denormalize(u, normalization["u"])
    return u


def load_ampc_model(model_path, num_layers, num_neurons, iteration=0):
    path = f"{model_path}/model_{num_layers}x{num_neurons}/It{iteration}"
    parameter_path = path + "/params.yaml"
    with open(parameter_path, "r") as file:
        params = yaml.safe_load(file)

    normalization_path = path + "/normalization.p"
    with open(normalization_path, "rb") as file:
        normalization = pickle.load(file)
    init_key = jax.random.PRNGKey(1)
    model = AMPCNN(num_layers=params["num_layers"], num_neurons=params["num_neurons"],
                   num_sys_states=params["num_sys_states"], num_sys_inputs=params["num_sys_inputs"],
                   num_aug_params=params["num_aug_params"], rng_key=init_key,
                   activation_function=params["activation_function"])
    model = model.load_model_from_file(path)

    return model, normalization

class ComputeUnit(net.Agent):
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

    def __init__(self, ID, slot_group_planned_trajectory_id,
                 slot_group_drone_state,
                 communication_delta_t,
                 trajectory_generator_options, pos_offset, prediction_horizon, num_computing_agents,
                 computing_agents_ids, setpoint_creator, offset=0,
                 alpha_1=10, alpha_2=1000, alpha_3=1 * 0, alpha_4=0, use_kkt_trigger=False, remove_redundant_constraints=False,
                 slot_group_state_id=None,
                 slot_group_ack_id=100000, ignore_message_loss=False, use_own_targets=False,
                 state_feedback_trigger_dist=0.5, simulated=True, use_high_level_planner=True,
                 agent_dodge_distance=0.5, slot_group_setpoints_id=100000, send_setpoints=False,
                 use_given_init_pos=False,
                 use_optimized_constraints=True, weight_band=1.0,
                 save_snapshot_times=[], snapshot_saving_path="",
                 simulate_quantization=False,
                 show_animation=False,
                 min_num_drones=0,
                 show_print=True,
                 name_run="",
                 log_optimizer=False,
                 log_optimizer_path="",
                 use_dampc=False,
                 dampc_model_path="",
                 dampc_num_neurons="",
                 dampc_num_layers=""
                 ):
        """

        Parameters
        ----------
            ID:
                identification number of agent
            slot_group_planned_trajectory_id:
                ID of the slot group which is used
            init_positions:
                3D position of agents in a hash_map id of agent is key
            trajectory_generator_options: tg.TrajectoryGeneratorOptions
                options for data generator
            communication_delta_t: float
                time difference between two communication rounds
            prediction_horizon: int
                prediction horizon of optimizer
            order_interpolation: int
                order of interpolation
            comp_agent_prio:
                priority of the computaiton agent. agent with 0 gets the agent with the highes priority...

        """
        super().__init__(ID, [slot_group_planned_trajectory_id] if not send_setpoints else [
            slot_group_planned_trajectory_id, slot_group_setpoints_id])
        self.__comp_agent_prio = computing_agents_ids.index(ID)
        self.__comp_agent_idx = computing_agents_ids.index(self.ID)
        self.__slot_group_planned_trajectory_id = slot_group_planned_trajectory_id
        self.__slot_group_setpoints_id = slot_group_setpoints_id
        # self.__send_setpoints = send_setpoints
        self.__slot_group_drone_state = slot_group_drone_state
        self.__slot_group_state_id = slot_group_state_id
        self.__slot_group_ack_id = slot_group_ack_id
        self.__ignore_message_loss = ignore_message_loss
        self.__state_feedback_trigger_dist = state_feedback_trigger_dist
        self.__trajectory_generator_options = trajectory_generator_options
        self.__pos_offset = pos_offset
        self.__use_optimized_constraints = use_optimized_constraints
        self.__use_high_level_planner = use_high_level_planner
        self.__high_level_setpoints = None
        self.__deadlock_breaker_agents = []
        self.__using_intermediate_targets = False
        self.__hlp_lock = 0  # for 5 rounds, we block the hlp after the hlp has been called
        self.__agent_dodge_distance = agent_dodge_distance
        self.__recalculate_setpoints = False  # there might exist cases, where we need to recalculate the setpoints immediately

        self.__computing_agents_ids = computing_agents_ids
        self.__communication_delta_t = communication_delta_t

        self.__alpha_1 = alpha_1
        self.__alpha_2 = alpha_2
        self.__alpha_3 = alpha_3
        self.__alpha_4 = alpha_4
        self.__use_kkt_trigger = use_kkt_trigger

        self.__state_feedback_triggered = []

        self.__current_time = 0
        self.__prediction_horizon = prediction_horizon

        self.__remove_redundant_constraints = remove_redundant_constraints

        self.__breakpoints = \
            np.linspace(0, communication_delta_t * prediction_horizon, prediction_horizon * int(communication_delta_t
                                                                                                / trajectory_generator_options.optimization_variable_sample_time + 1e-7) + 1)

        self.__drones_prios = {}

        self.__trajectory_interpolation = TripleIntegrator(
            breakpoints=self.__breakpoints,
            dimension_trajectory=3,
            num_states=9,
            sampling_time=trajectory_generator_options.optimization_variable_sample_time)

        self.__trajectory_generator = tg.TrajectoryGenerator(
            options=trajectory_generator_options,
            trajectory_interpolation=self.__trajectory_interpolation)

        self.__number_rounds = 0  # same as current_time divided by communication_delta_t
        self.__options = trajectory_generator_options

        self.__current_agent = 0

        self.__drones_ids = []
        self.__num_computing_agents = num_computing_agents

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

        self.__trajectory_tracker = InformationTracker()
        self.__last_trajectory_shape = (len(self.__options.collision_constraint_sample_points), 3)

        self.__last_calc_time = None  # duration of last calculation in seconds (can be used to measre time algorithm takes)
        self.__last_received_messages = {self.ID: EmtpyContent([])}

        self.__ack_message = None  # currently we have no acknowledgement flag in the communication (so an agent is notified if its message was received)

        self.__system_state = NORMAL
        self.__last_system_state = NORMAL  # used for debugging if MLR module

        # if the cus should decide which targets the drones have or the drones should decide this.
        self.__use_own_targets = use_own_targets

        self.__num_trajectory_messages_received = 0

        self.ordered_indexes = None  # used for debugging of priority based trigger

        self.__total_num_optimizer_runs = 0
        self.__num_succ_optimizer_runs = 0

        self.__simulated = simulated

        self.__downwash_scaling = np.diag(
            [1, 1, 1.0 / float(self.__trajectory_generator_options.downwash_scaling_factor)])

        # field for event trigger, all agents generate a consensus of prios.
        self.__prio_consensus = []

        self.__setpoint_creator = setpoint_creator

        # the difference between those two is that current_target_positions are the target positions last received
        # from the corresponding drone. received_setpoints are the setpoints received from the CU that is the high level
        # planner. If they are none, the CU tries to steer the drones to current_target_positions else to
        # received_setpoints. See get_targets()
        self.__received_setpoints = None
        self.__current_target_positions = {}

        self.__num_trigger_times = {drone_id: 0 for drone_id in self.__drones_ids}  # used for debugging and plots

        self.__selected_UAVs = {"round": [], "selected": []}  # used for debugging and plots

        self.__weight_band = weight_band

        self.__save_snapshot_times = save_snapshot_times
        self.__snapshot_saving_path = snapshot_saving_path
        self.__setpoint_history = []

        self.__simulate_quantization = simulate_quantization

        self.__show_animation = show_animation
        self.__data_pipeline = cu_animator.DataPipeline()

        self.__trigger = None  # used to load trigger times e.g. from hardware experiments and then simulating the system with thos instead of its own trigger.

        self.__received_network_members_message = True

        self.__min_num_drones = min_num_drones  # minimum number of drones over which the CU starts running

        self.__show_print = show_print

        self.__name_run = name_run

        # logging the optimizer outputs and inputs for AMPC
        self.__log_optimizer = log_optimizer
        self.__log_optimizer_input_buffer = None
        self.__log_optimizer_output_buffer = None
        self.__log_optimizer_num_elements = 0
        self.__log_optimizer_path = log_optimizer_path

        self.__use_dampc = use_dampc
        if use_dampc:
            self.__dampc_model, self.__dampc_normalization = load_ampc_model(dampc_model_path, dampc_num_layers,
                                                                             dampc_num_neurons, iteration=0)

        if log_optimizer:
            assert log_optimizer == ignore_message_loss, "Not implemented."

    def load_trigger(self, trigger):
        self.__trigger = trigger

    def add_new_drone(self, m_id):
        last_trajectory = None

        coeff = tg.TrajectoryCoefficients(None,
                                          False,
                                          np.zeros((self.__prediction_horizon * int(round(self.__communication_delta_t /
                                                                                          self.__trajectory_generator_options.optimization_variable_sample_time)),
                                                    3)))

        trajectory_content = TrajectoryContent(coefficients=coeff, last_trajectory=last_trajectory,
                                               init_state=None, current_state=None,
                                               trajectory_start_time=0, id=m_id, trajectory_calculated_by=-100)

        self.__trajectory_tracker.add_unique_information(m_id, trajectory_content)
        self.__trajectory_tracker.set_deprecated(m_id)

        if self.__system_state == NORMAL:
            self.__system_state = INFORMATION_DEPRECATED

        self.__drones_ids.append(m_id)
        self.__drones_ids.sort()
        self.__drones_prios[m_id] = m_id

        # first every drone should fly to the origin, until the CU receives the target position from the drone
        self.__setpoint_creator.add_drone(m_id, np.array([0.0, 0.0, 1.0]),
                                          int(round(self.__current_time / self.__communication_delta_t)))

        self.__current_target_positions[m_id] = np.array([0.0, 0.0, 1.0])

        self.__num_trigger_times[m_id] = 0

    def remove_drone(self, m_id):
        self.__trajectory_tracker.delete_information(m_id)
        self.__drones_prios.pop(m_id)
        self.__drones_ids.remove(m_id)
        self.__drones_ids.sort()

    def add_new_computation_agent(self, m_id):
        self.__computing_agents_ids.append(m_id)
        self.__computing_agents_ids.sort()
        self.__num_computing_agents = len(self.__computing_agents_ids)
        self.__comp_agent_prio = self.__computing_agents_ids.index(self.ID)
        self.__comp_agent_idx = self.__computing_agents_ids.index(self.ID)

        # self.__send_setpoints = self.ID == self.__computing_agents_ids[0]

    def remove_computation_agent(self, m_id):
        if m_id not in self.__computing_agents_ids:
            return

        self.__computing_agents_ids.remove(m_id)
        self.__computing_agents_ids.sort()
        self.__num_computing_agents = len(self.__computing_agents_ids)
        self.__comp_agent_prio = self.__computing_agents_ids.index(self.ID)
        self.__comp_agent_idx = self.__computing_agents_ids.index(self.ID)
        # self.__send_setpoints = self.ID == self.__computing_agents_ids[0]

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
                copy_message = copy_message if not self.__simulate_quantization else simulate_quantization_trajectory_message(
                    copy_message)
            return net.Message(self.ID, slot_group_id, copy_message)

        elif slot_group_id == self.__slot_group_state_id:
            pass
        elif slot_group_id == self.__slot_group_ack_id:
            pass
        elif slot_group_id == self.__slot_group_setpoints_id:
            if self.__send_setpoints and self.__received_network_members_message:
                setpoints = copy.deepcopy(self.__high_level_setpoints)
                if setpoints is not None:
                    for drone_id in setpoints:
                        setpoints[drone_id] -= self.__pos_offset[drone_id]
                setpoint_message = SetpointMessageContent(setpoints)
                setpoint_message = setpoint_message if not self.__simulate_quantization else simulate_quantization_setpoints_message(
                    setpoint_message)
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
                    self.print("sssssssssss")
                    self.print(self.__system_state)
                    self.print(self.__last_system_state)
                assert message.ID == self.ID

        if message.slot_group_id == self.__slot_group_drone_state:
            if message.ID not in self.__trajectory_tracker.keys:
                return
            message.content.state[0:3] += self.__pos_offset[message.ID]
            self.print(f"Received pos from {message.ID}: {message.content.state}")

            message.content.target_position += self.__pos_offset[message.ID]
            # The state is measured at the beginning of the round and extrapolated by drone
            # init drone state if it has to be init. (The state is measured at the beginning of the last round.)
            # calulate number of timesteps, the data need to be delayed
            delay_timesteps = (self.__current_time - self.__trajectory_tracker.get_information(message.ID).content[
                0].trajectory_start_time) / self.__options.optimization_variable_sample_time
            delay_timesteps = int(np.round(delay_timesteps, 0))  # needs to be rounded because if of float inaccuracies
            if self.__trajectory_tracker.get_information(message.ID).content[0].init_state is None:
                # because the current coefficients are zero, we can set the init state equals the current state.
                trajectory = self.__trajectory_tracker.get_information(message.ID).content[0]
                trajectory.init_state = copy.deepcopy(message.content.state)
                trajectory.current_state = copy.deepcopy(message.content.state)
                trajectory.last_trajectory = None
                self.__recalculate_setpoints = True

            for trajectory in self.__trajectory_tracker.get_information(message.ID).content:
                if np.linalg.norm(trajectory.current_state[0:3] - message.content.state[
                                                                  0:3]) > self.__state_feedback_trigger_dist:
                    self.print(f"{message.ID}: {trajectory.current_state} {message.content.state[0:3]}")
                    self.__state_feedback_triggered.append(message.ID)
                    # trajectory.current_state = np.zeros(trajectory.current_state.shape)
                    trajectory.current_state[0:3] = copy.deepcopy(message.content.state[0:3])

                    # update last_trajectory, because we do not have the init state anymore as a state on the trajectory
                    # (some state in between instead), we have to do a different calculation of the last_trajecotry than
                    # in round_finished()
                    coeff = None
                    if trajectory.coefficients.valid:
                        # coeff = np.zeros(trajectory.coefficients.coefficients.shape)
                        # trajectory.coefficients.coefficients = coeff
                        coeff = trajectory.coefficients.coefficients
                    else:
                        # coeff = np.zeros(trajectory.coefficients.alternative_trajectory.shape)
                        # trajectory.coefficients.alternative_trajectory = coeff
                        coeff = trajectory.coefficients.alternative_trajectory
                    delay_timesteps = (self.__current_time - trajectory.trajectory_start_time) \
                                      / self.__options.optimization_variable_sample_time
                    delay_timesteps = int(
                        np.round(delay_timesteps, 0))  # needs to be rounded because if of float inaccuracies

                    coeff_shifted = np.array([coeff[min((i + delay_timesteps, len(self.__breakpoints) - 2))]
                                              for i in range(len(self.__breakpoints) - 1)])
                    coeff_resize = np.reshape(coeff_shifted, (coeff_shifted.size,))

                    trajectory.last_trajectory = np.reshape((self.__input_trajectory_vector_matrix[0] @ coeff_resize + \
                                                             self.__state_trajectory_vector_matrix[0] @
                                                             trajectory.current_state), self.__last_trajectory_shape)

                    self.print("3333333333333333333")

                    # also recalculate setpoints
                    self.__recalculate_setpoints = True

            # process targets from drone (note, if use_own_targets is true this is ignored)
            if message.ID in self.__current_target_positions.keys():
                # change target position if target changed
                if np.any(message.content.target_position != self.__current_target_positions[message.ID]):
                    self.__current_target_positions[message.ID] = copy.deepcopy(message.content.target_position)
                    self.__drones_prios[message.ID] = message.ID  # change prio for cooperative behaviour
                    # target changed, thus recalculate setpoints
                    if not self.__use_own_targets:
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
                        self.print(f"Trajectories equal: {message.ID}, {message.content.trajectory_calculated_by}, "
                                   f"{message.content.trajectory_start_time}")
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
                for drone_id in self.__received_setpoints:
                    self.__received_setpoints[drone_id] += self.__pos_offset[drone_id]
                self.print(f"New setpoints: {self.__received_setpoints}")
            pass

    def __update_information_tracker(self):
        # update trajectories (leads to a faster calculation)
        delay_timesteps = self.__communication_delta_t / (self.__options.collision_constraint_sample_points[1] -
                                                          self.__options.collision_constraint_sample_points[0])
        delay_timesteps = int(np.round(delay_timesteps, 0))  # needs to be rounded because if of float inaccuracies
        for i in range(0, len(self.__drones_ids)):
            id = self.__drones_ids[i]
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
                    trajectory.last_trajectory = np.reshape(
                        (self.__input_trajectory_vector_matrix[delay] @ coeff_resize + \
                         self.__state_trajectory_vector_matrix[delay] @
                         trajectory.init_state), self.__last_trajectory_shape)

                else:
                    trajectory.last_trajectory = np.array(
                        [trajectory.last_trajectory[min((j + delay_timesteps, len(trajectory.last_trajectory) - 1))]
                         for j in range(len(trajectory.last_trajectory))])
        # update states
        for i in range(0, len(self.__drones_ids)):
            id = self.__drones_ids[i]
            for trajectory in self.__trajectory_tracker.get_information(id).content:
                if trajectory.init_state is None:
                    continue
                trajectory.current_state = self.__trajectory_interpolation.interpolate(
                    self.__current_time - trajectory.trajectory_start_time + self.__communication_delta_t,
                    trajectory.coefficients,
                    x0=trajectory.current_state,
                    integration_start=self.__current_time - trajectory.trajectory_start_time)
                self.print(f"Current_pos: {trajectory.current_state[0:3]}")

        # only the CU responsible for the high level planning is responsible for the animation
        if self.__show_animation and self.__send_setpoints:
            trajectories = {}
            current_target_positions = {}
            intermediate_targets = {}
            for i in range(0, len(self.__drones_ids)):
                trajectories[self.__drones_ids[i]] = \
                self.__trajectory_tracker.get_information(self.__drones_ids[i]).content[0].last_trajectory

                # show targets and intermediate setpoints.
                if self.__drones_ids[i] in self.__current_target_positions:
                    current_target_positions[self.__drones_ids[i]] = self.get_targets()[self.__drones_ids[i]]
                if self.__received_setpoints is not None:
                    if self.__drones_ids[i] in self.__received_setpoints:
                        intermediate_targets[self.__drones_ids[i]] = self.get_drone_intermediate_setpoint(
                            self.__drones_ids[i])
            self.__data_pipeline.set_trajectories(trajectories, current_target_positions, intermediate_targets)

    def round_finished(self, round_nmbr=None, received_network_members_message=True):
        """this function has to be called at the end of the round to tell the agent that the communication round is
        finished"""
        self.__received_network_members_message = received_network_members_message
        self.__selected_UAVs["selected"].append(-1)
        self.__selected_UAVs["round"].append(round_nmbr)
        self.__last_system_state = self.__system_state
        self.ordered_indexes = None
        start_time = time.time()
        if round_nmbr is not None:
            self.__current_time = round_nmbr * self.__communication_delta_t

        if len(self.__drones_ids) == 0:
            self.__last_received_messages = {self.ID: EmtpyContent([])}
            return

        n = int(round(self.__current_time / self.__communication_delta_t))
        self.__comp_agent_prio = (self.__comp_agent_idx + n) % self.__num_computing_agents

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
            # the other cus has not send a trajectory message, but something else.
            # This means it has not calculated a new trajectory
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
            self.__prio_consensus = [1 for _ in range(len(self.__drones_ids))]

        # check if a message from another CU was not received. If one was not received, we do not know which information is
        # deprecated, we thus have to check this in the next round

        if (not self.__num_trajectory_messages_received == len(
                self.__computing_agents_ids)) and not self.__ignore_message_loss:
            for information in self.__trajectory_tracker.get_all_information().values():
                information.set_deprecated()
            self.print(f"Lost messages of CU! {self.__num_trajectory_messages_received}")
        self.__num_trajectory_messages_received = 0
        assert self.__num_trajectory_messages_received <= len(self.__computing_agents_ids)

        self.__update_information_tracker()

        # if we are in the normal state and something is deprecated, switch the state
        if self.__system_state == NORMAL and not self.__trajectory_tracker.no_information_deprecated:
            self.__system_state = INFORMATION_DEPRECATED
        if self.__trajectory_tracker.no_information_deprecated and self.__system_state != WAIT_FOR_UPDATE:
            self.__system_state = NORMAL

        if self.__system_state == NORMAL:
            # until then we have no received data from the drone agents not send (we do not have the initial state)
            all_init_states_known = True
            for trajectory_information in self.__trajectory_tracker.get_all_information().values():
                for trajectory in trajectory_information.content:
                    if trajectory.init_state is None:
                        all_init_states_known = False
                        break
            targets = self.get_targets()
            for drone_id in targets:
                if targets[drone_id] is None:
                    all_init_states_known = False
                    break
            if not all_init_states_known:
                # self.__update_information_tracker()
                self.__number_rounds = self.__number_rounds + 1
                self.__last_received_messages = {self.ID: EmtpyContent([])}
                self.__current_time = self.__current_time + self.__communication_delta_t
                return

            if len(self.__drones_ids) <= self.__comp_agent_prio:
                prios = self.calc_prio()
                self.__last_received_messages = {self.ID: EmtpyContent(prios)}
            else:
                # if an agent leaves, the prios might still be using the old swarm setup,
                # then, do calculate nothing.
                if len(self.__prio_consensus) > len(self.__drones_ids) or len(
                        self.__drones_ids) < self.__min_num_drones:
                    prios = self.calc_prio()
                    self.__last_received_messages = {self.ID: EmtpyContent(prios)}
                else:
                    # select next agent
                    # ordered_indexes = self.order_agents_by_priority()
                    self.print(f"self.__prio_consensus: {self.__prio_consensus}")
                    ordered_indexes = np.argsort(-np.array(self.__prio_consensus),
                                                 kind="stable")  # self.order_agents_by_priority()#

                    self.__current_agent = ordered_indexes[self.__comp_agent_prio]
                    current_id = self.__drones_ids[self.__current_agent]
                    self.__num_trigger_times[current_id] += 1
                    self.__selected_UAVs["selected"][-1] = current_id

                    current_agent_real = 0
                    current_id_real = 0
                    if self.__trigger is not None:
                        for lkj in range(len(self.__trigger["round"])):
                            if self.__trigger["round"][lkj] == int(
                                    round(self.__current_time / self.__communication_delta_t)):
                                for key in range(len(self.__drones_ids)):
                                    if self.__drones_ids[key] == self.__trigger["selected"][lkj]:
                                        current_agent_real = key
                                current_id_real = self.__drones_ids[current_agent_real]

                        self.print(f"current_id_real: {current_id_real}, current_id {current_id}")
                    self.ordered_indexes = ordered_indexes

                    # if the information about the agent is not unique do not calculate something, because we will calculate
                    # something wrong. Same is true if we have not received the members message.
                    # if the agent is remove but was scheduled for recalculation in the last round, do nothing.
                    if not self.__trajectory_tracker.get_information(current_id).is_unique \
                            or not received_network_members_message or not self.__drones_ids[
                                                                               self.__current_agent] in self.__drones_ids:
                        prios = self.calc_prio()
                        self.__last_received_messages = {self.ID: EmtpyContent(prios)}
                    else:
                        # solve optimization problem
                        calc_coeff = self.__calculate_trajectory(current_id=current_id, ordered_indexes=ordered_indexes)
                        self.__total_num_optimizer_runs += 1
                        if calc_coeff.valid:
                            self.__num_succ_optimizer_runs += 1

                        prios = self.calc_prio()
                        prios[self.__current_agent] = 0
                        self.__last_received_messages = {
                            self.ID: TrajectoryMessageContent(coefficients=calc_coeff,
                                                              init_state=self.__trajectory_tracker.get_information(
                                                                  current_id).content[0].current_state,
                                                              trajectory_start_time=self.__current_time + self.__communication_delta_t,
                                                              trajectory_calculated_by=self.ID,
                                                              id=current_id, prios=prios)}
                        self.print('Distance to target for Agent ' + str(current_id) + ': ' + str(np.linalg.norm(
                            self.__trajectory_tracker.get_information(current_id).content[0].current_state[0:3] -
                            self.get_targets()[current_id])) + " m.")
                        self.print(
                            f"agent_state: {self.__trajectory_tracker.get_information(current_id).content[0].current_state[0:3]}, setpoint: {self.get_targets()[current_id]}")
                        self.print(
                            'Optimization for Agent ' + str(current_id) + ' took ' + str(
                                time.time() - start_time) + ' s.')

        elif self.__system_state == INFORMATION_DEPRECATED:
            # send an emtpy message such that other agents know this is not a lost message
            prios = self.calc_prio()
            self.__last_received_messages = {
                self.ID: EmtpyContent(prios)}
            self.__system_state = RECOVER_INFORMATION_NOTIFY
        elif self.__system_state == RECOVER_INFORMATION_NOTIFY:
            self.print("RECOVER_INFORMATION_NOTIFY")
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
            self.print("WAIT_FOR_UPDATE")
            # send nothing and wait for drone to send
            self.__last_received_messages = {}
            self.__system_state = RECOVER_INFORMATION_NOTIFY

        self.print(f"final self.__system_state {self.__system_state}")
        # start time of the newly calculated data.
        self.__number_rounds = self.__number_rounds + 1
        self.__current_time = self.__current_time + self.__communication_delta_t

        self.__last_calc_time = time.time() - start_time

        self.__state_feedback_triggered = []

        if round_nmbr is not None:
            if round_nmbr % 5 == 0 and not self.__simulated:
                with open(
                        f'../../experiment_measurements/num_trigger_times{self.ID}_{int(self.__alpha_1)}_{int(self.__alpha_2)}_{int(self.__alpha_3)}_{int(self.__alpha_4)}.p',
                        'wb') as handle:
                    pickle.dump({"num_trigger_times": self.__num_trigger_times, "selected_UAVs": self.__selected_UAVs},
                                handle)

        if int(round(self.__current_time / self.__communication_delta_t)) % 5 == 0 and not self.__simulated:
            with open(
                    f'../../experiment_measurements/num_trigger_times_{self.__name_run}_{self.ID}_{int(self.__alpha_1)}_{int(self.__alpha_2)}_{int(self.__alpha_3)}_{int(self.__alpha_4)}.p',
                    'wb') as handle:
                pickle.dump({"num_trigger_times": self.__num_trigger_times, "selected_UAVs": self.__selected_UAVs},
                            handle)

        if int(round(self.__current_time / self.__communication_delta_t)) in self.__save_snapshot_times:
            with open("../../experiment_measurements" +
                      f"/CU{self.ID}snapshot{int(self.__current_time / self.__communication_delta_t)}.p", 'wb') \
                    as out_file:
                pickle.dump(self, out_file)

        if len(self.__save_snapshot_times) > 0:
            if int(round(self.__current_time / self.__communication_delta_t)) >= self.__save_snapshot_times[0]:
                setpoints = {}
                for drone_id in self.__drones_ids:
                    setpoints[drone_id] = self.__trajectory_tracker.get_information(drone_id).content[0].current_state[
                                          0:3]
                self.__setpoint_history.append(setpoints)
            if int(round(self.__current_time / self.__communication_delta_t)) == 450:
                with open("../../experiment_measurements" +
                          f"/CU{self.ID}setpoints{int(self.__current_time / self.__communication_delta_t)}.p", 'wb') \
                        as out_file:
                    pickle.dump(self.__setpoint_history, out_file)

    def __calculate_trajectory(self, current_id, ordered_indexes):
        if self.__log_optimizer or self.__use_dampc:
            if self.__log_optimizer_input_buffer is None:
                self.__log_optimizer_input_buffer = (
                    np.zeros((1000, len(self.__drones_ids) * (self.__prediction_horizon * 3 + 9 + 1) + 3)))
                self.__log_optimizer_output_buffer = np.zeros((1000, self.__prediction_horizon * 3))

        # calulate number of timesteps, the alternative data need to be delayed
        delay_timesteps = (self.__current_time - self.__trajectory_tracker.get_information(current_id).content[
            0].trajectory_start_time +
                           self.__communication_delta_t) / self.__options.optimization_variable_sample_time
        delay_timesteps = int(np.round(delay_timesteps, 0))  # needs to be rounded because if of float inaccuracies

        # calculate how many trajectories we have
        num_trajectories = 0
        for information in self.__trajectory_tracker.get_all_information().values():
            for content in information.content:
                num_trajectories += 1

        num_dynamic_trajectories = 0
        num_static_trajectories = 0
        for j in range(len(self.__drones_ids)):
            id = self.__drones_ids[ordered_indexes[j]]
            if j < self.__num_computing_agents:
                num_dynamic_trajectories += len(self.__trajectory_tracker.get_information(id).content)
            else:
                num_static_trajectories += len(self.__trajectory_tracker.get_information(id).content)

        band_weights = []
        last_trajectory_current_agent = self.__trajectory_tracker.get_information(current_id).content[0].last_trajectory

        weight_band = self.__weight_band if not self.__using_intermediate_targets else 1e-6

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
            for j in range(len(self.__drones_ids)):
                id = self.__drones_ids[ordered_indexes[j]]
                if j < self.__num_computing_agents:
                    for trajectory in self.__trajectory_tracker.get_information(id).content:
                        dynamic_trajectories[i] = trajectory.last_trajectory
                        dynamic_target_points[i] = self.get_targets()[id]
                        if j == self.__comp_agent_prio:  # or i == self. ?
                            index = i
                        self.__drones_prios[id] = 0  # self.__drones_prios[id] if np.linalg.norm(
                        # dynamic_target_points[i] - dynamic_trajectories[i][-1]) > 2 * self.__options.r_min else -1
                        dynamic_coop_prio[i] = self.__drones_prios[id]

                        age_trajectory = (self.__current_time - trajectory.trajectory_start_time) / (
                                self.__options.collision_constraint_sample_points[1] -
                                self.__options.collision_constraint_sample_points[0])
                        age_trajectory = int(
                            np.round(age_trajectory, 0))  # needs to be rounded because if of float inaccuracies
                        dynamic_trajectory_age[i] = age_trajectory

                        band_weights.append(calculate_band_weight(p_target=self.get_targets()[current_id],
                                                                  p_self=last_trajectory_current_agent[-1],
                                                                  p_other=trajectory.last_trajectory[-1],
                                                                  weight=weight_band))
                        i += 1
                else:
                    for trajectory in self.__trajectory_tracker.get_information(id).content:
                        static_trajectories[i - num_dynamic_trajectories] = trajectory.last_trajectory
                        static_target_points[i - num_dynamic_trajectories] = self.get_targets()[id]
                        self.__drones_prios[id] = self.__drones_prios[id] if np.linalg.norm(
                            static_target_points[i - num_dynamic_trajectories] -
                            static_trajectories[i - num_dynamic_trajectories][-1]) > 2 * self.__options.r_min else -1
                        static_coop_prio[i - num_dynamic_trajectories] = self.__drones_prios[id]

                        age_trajectory = (self.__current_time - trajectory.trajectory_start_time) / (
                                self.__options.collision_constraint_sample_points[1] -
                                self.__options.collision_constraint_sample_points[0])
                        age_trajectory = int(
                            np.round(age_trajectory, 0))  # needs to be rounded because if of float inaccuracies
                        static_trajectory_age[i - num_dynamic_trajectories] = age_trajectory

                        band_weights.append(calculate_band_weight(p_target=self.get_targets()[current_id],
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
                    dynamic_target_points[i] = self.get_targets()[trajectory.id]
                    if trajectory.id == current_id:
                        index = i

                    self.__drones_prios[trajectory.id] = self.__drones_prios[trajectory.id] if np.linalg.norm(
                        dynamic_target_points[i] - dynamic_trajectories[i][-1]) > 2 * self.__options.r_min else -1
                    dynamic_coop_prio[i] = self.__drones_prios[trajectory.id]
                    i += 1

        if self.__trajectory_tracker.get_information(current_id).content[0].coefficients.valid:
            previous_solution_shifted = self.__trajectory_tracker.get_information(current_id).content[
                0].coefficients.coefficients
            previous_solution_shifted = np.array(
                [previous_solution_shifted[min((i + delay_timesteps, len(self.__breakpoints) - 2))]
                 for i in range(len(self.__breakpoints) - 1)])
        else:
            previous_solution_shifted = self.__trajectory_tracker.get_information(current_id).content[
                0].coefficients.alternative_trajectory
            previous_solution_shifted = np.array(
                [previous_solution_shifted[min((i + delay_timesteps, len(self.__breakpoints) - 2))]
                 for i in range(len(self.__breakpoints) - 1)])

        if self.__log_optimizer or self.__use_dampc:
            num_elements_per_drone_trajectory = self.__prediction_horizon * 3 + 9
            offset = num_elements_per_drone_trajectory + 3 + 1
            for j in range(len(self.__drones_ids)):
                drone_id = self.__drones_ids[ordered_indexes[j]]
                if j == self.__comp_agent_prio:
                    self.__log_optimizer_input_buffer[self.__log_optimizer_num_elements,
                    0:num_elements_per_drone_trajectory] = (
                        self.log_optimizer_calculate_input_array(drone_id))
                    self.__log_optimizer_input_buffer[self.__log_optimizer_num_elements,
                    num_elements_per_drone_trajectory:num_elements_per_drone_trajectory + 3] = self.get_drone_intermediate_setpoint(
                        drone_id)
                    self.__log_optimizer_input_buffer[self.__log_optimizer_num_elements,
                    num_elements_per_drone_trajectory + 3] = band_weights[j]
                else:
                    self.__log_optimizer_input_buffer[self.__log_optimizer_num_elements, offset:offset + num_elements_per_drone_trajectory] = (
                        self.log_optimizer_calculate_input_array(drone_id))
                    self.__log_optimizer_input_buffer[self.__log_optimizer_num_elements, offset + num_elements_per_drone_trajectory] = band_weights[j]

                    offset += num_elements_per_drone_trajectory + 1

        if self.__use_dampc:
            coeff = np.array(call_ampc(self.__log_optimizer_input_buffer[self.__log_optimizer_num_elements], self.__dampc_model, self.__dampc_normalization))
            coeff = np.reshape(coeff, (self.__prediction_horizon, 3))
            coeff1 = TrajectoryCoefficients(coefficients=coeff, valid=True, alternative_trajectory=coeff)
            #print("-----------------------------------------")
            #print(coeff1.coefficients)
        else:
            coeff = self.__trajectory_generator.calculate_trajectory(
                current_id=current_id,
                current_state=self.__trajectory_tracker.get_information(current_id).content[0].current_state,
                target_position=self.get_drone_intermediate_setpoint(current_id),
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
                high_level_setpoints=None
            )
            # print(coeff.coefficients)
            # print(coeff.coefficients - coeff1.coefficients)

        if self.__log_optimizer:
            self.__log_optimizer_output_buffer[self.__log_optimizer_num_elements, :] = coeff.coefficients.flatten() if coeff.valid else coeff.alternative_trajectory.flatten()

            self.__log_optimizer_num_elements += 1
            self.__log_optimizer_num_elements %= 1000

        return coeff

    def log_optimizer_calculate_input_array(self, drone_id):
        # calulate number of timesteps, the data need to be delayed
        delay_timesteps = (self.__current_time - self.__trajectory_tracker.get_information(drone_id).content[
            0].trajectory_start_time + self.__communication_delta_t) / self.__options.optimization_variable_sample_time
        delay_timesteps = int(np.round(delay_timesteps, 0))  # needs to be rounded because if of float inaccuracies

        # calculate andy shift coefficients
        if self.__trajectory_tracker.get_information(drone_id).content[0].coefficients.valid:
            previous_solution_shifted = self.__trajectory_tracker.get_information(drone_id).content[
                0].coefficients.coefficients
            previous_solution_shifted = np.array(
                [previous_solution_shifted[min((i + delay_timesteps, len(self.__breakpoints) - 2))]
                 for i in range(len(self.__breakpoints) - 1)])
        else:
            previous_solution_shifted = self.__trajectory_tracker.get_information(drone_id).content[
                0].coefficients.alternative_trajectory
            previous_solution_shifted = np.array(
                [previous_solution_shifted[min((i + drone_id, len(self.__breakpoints) - 2))]
                 for i in range(len(self.__breakpoints) - 1)])

        return np.concatenate((previous_solution_shifted.flatten(),
                               self.__trajectory_tracker.get_information(drone_id).content[0].current_state.flatten()))

    def save_log_optimizer(self):
        if self.__log_optimizer:
            try:
                os.makedirs(self.__log_optimizer_path)
            except:
                pass
            finally:
                pass

            file_postfix = f"{self.ID}_{datetime.now().strftime('%Y%m%d%H%M%S%Z.%f').replace('.', '_')}"
            self.__log_optimizer_output_buffer[0:self.__log_optimizer_num_elements, :].tofile(self.__log_optimizer_path + f"/output_{file_postfix}.npy")
            self.__log_optimizer_input_buffer[0:self.__log_optimizer_num_elements, :].tofile(self.__log_optimizer_path + f"/input_{file_postfix}.npy")

    def get_drone_intermediate_setpoint(self, drone_id):
        """returns the current intermediate setpoint the CU should steer the drone to."""
        target_pos = None

        if drone_id in self.__current_target_positions:
            target_pos = self.__current_target_positions[drone_id]

        if self.__received_setpoints is not None:
            if drone_id in self.__received_setpoints:
                target_pos = self.__received_setpoints[drone_id]

        return target_pos

    def get_targets(self):
        """
        Returns the current target position of each agent, that an agent wishes to fly to (or if we want the CU to calculate
        their own targets, the ones of the setpoint creator)
        The CU that calculates the high level planner might assign different intermediate setpoints to the drone to resolve deadlocks.
        Those can be got via get_drone_intermediate_setpoint.

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
            setpoints, new_setpoints = self.__setpoint_creator.next_setpoints(
                round_nmbr=round(self.__current_time / self.__communication_delta_t))
            if new_setpoints:
                self.__recalculate_setpoints = new_setpoints
            return setpoints
        else:
            return self.__current_target_positions

    def calc_prio(self):
        """ calculates the prio for each agent """
        state_feeback_triggered_prio = 10000
        quantization_bit_number = 8
        max_time = self.__prediction_horizon * self.__communication_delta_t * self.__num_computing_agents
        max_time = self.__prediction_horizon * self.__communication_delta_t * len(self.__drones_ids) / self.__num_computing_agents
        max_time = self.__communication_delta_t * len(self.__drones_ids) / self.__num_computing_agents

        cone_angle = 60.0 * math.pi / 180

        is_in_deadlock = [False for i in range(len(self.__drones_ids))]
        prios = np.zeros((len(self.__drones_ids),))
        for i in range(0, len(self.__drones_ids)):
            own_id = self.__drones_ids[i]
            max_dist = np.linalg.norm(self.__options.max_position[own_id] - self.__options.min_position[own_id])

            target_pos = self.get_drone_intermediate_setpoint(own_id)

            if target_pos is None or self.__trajectory_tracker.get_information(own_id).content[0].current_state is None:
                d_target = 0
            else:
                d_target = target_pos - copy.deepcopy(
                    self.__trajectory_tracker.get_information(own_id).content[0].current_state[0:3])

            dist_to_target = np.linalg.norm(d_target)

            prios[i] /= (1.0 * len(self.__drones_ids) * max_dist)  # normalize prio

            max_starting_time = 0
            for j in range(len(self.__trajectory_tracker.get_information(own_id).content)):
                if max_starting_time < self.__trajectory_tracker.get_information(own_id).content[
                    j].trajectory_start_time:
                    max_starting_time = self.__trajectory_tracker.get_information(own_id).content[
                        j].trajectory_start_time

            prios[i] += self.__alpha_2 * (self.__current_time - max_starting_time) / max_time
            prios[i] += self.__alpha_1 * dist_to_target / max_dist * (
                        self.__current_time - max_starting_time) / max_time
            prios[i] += self.__alpha_3 * dist_to_target / max_dist

            if own_id in self.__state_feedback_triggered:
                prios[i] += state_feeback_triggered_prio

            if i in np.argsort(-np.array(self.__prio_consensus))[
                    0:len(self.__computing_agents_ids)] and self.__ignore_message_loss:
                prios[i] += -100000000 * self.__alpha_4

            # KKT trigger.
            if self.__trajectory_tracker.get_information(own_id).content[0].current_state is not None:
                own_pos = self.__trajectory_tracker.get_information(own_id).content[0].current_state[0:3]
                constraints_vecs = []
                if self.drone_stands_still(own_id) and np.linalg.norm(d_target) > 0.1:
                    for other_drone_id in self.__drones_ids:
                        if other_drone_id == own_id or not self.drone_stands_still(other_drone_id):
                            continue
                        for c in self.__trajectory_tracker.get_information(other_drone_id).content:
                            if c.current_state is not None:
                                d_pos = self.__downwash_scaling @ (own_pos - c.current_state[0:3])
                                # pushes in a different direction
                                if np.dot(d_target, d_pos) < 0 and np.linalg.norm(d_pos) < 1.1*self.__options.r_min:
                                    constraints_vecs.append(d_pos)
                if len(constraints_vecs) > 0:
                    #print("cccccccccccccccccccccccccccc")
                    constraints_vecs = np.array(constraints_vecs).T
                    if np.linalg.matrix_rank(constraints_vecs) == 3:
                        is_in_deadlock[i] = True
                    else:
                        b = np.linalg.lstsq(constraints_vecs, d_target)[0]
                        # print(np.linalg.norm(constraints_vecs @ b - d_target))
                        if np.linalg.norm(constraints_vecs @ b - d_target) < 1e-2:
                            is_in_deadlock[i] = True



        # quantize prios
        for i in range(len(prios)):
            if prios[i] >= state_feeback_triggered_prio:
                prios[i] = int(2 ** quantization_bit_number - 1)
            elif prios[i] < 0:
                prios[i] = int(0)
            else:
                prios[i] = int(round(
                    prios[i] / (self.__alpha_1 + self.__alpha_2 + self.__alpha_3 + self.__alpha_4 * 0) * (
                                2 ** quantization_bit_number - 1)))
                if prios[i] > 2 ** quantization_bit_number - 2:
                    prios[i] = int(2 ** quantization_bit_number - 2)
            if self.__use_kkt_trigger and is_in_deadlock[i]:
                print("öööööööööööööööööööö")
                prios[i] = 1
        return prios

    def drone_stands_still(self, drone_id):
        for c in self.__trajectory_tracker.get_information(drone_id).content:
            if c.current_state is not None:
                coefficients = c.coefficients.coefficients if c.coefficients.valid else c.coefficients.alternative_trajectory
                if np.linalg.norm(coefficients.flatten()) > 0.01*self.__prediction_horizon:
                    return False
        return True

    def print(self, text):
        if self.__show_print:
            print("[" + str(self.ID) + "]: " + str(text))

    def deadlock_breaker_condition(self, drone_id, other_drone_id, own_pos, other_pos, inter_dist, own_dist_to_target,
                                   other_agent_dist_to_target):
        if other_drone_id == drone_id:
            return False

        if other_agent_dist_to_target < self.__options.r_min * 0.9 and other_agent_dist_to_target < own_dist_to_target:
            return False
        elif other_agent_dist_to_target > own_dist_to_target:
            # if self.__scaled_norm(
            #        other_pos[0:2] - own_pos[0:2]) < self.__options.r_min * 1.05:
            if inter_dist < self.__options.r_min * 2.0:
                if np.dot(
                        self.__trajectory_tracker.get_information(other_drone_id).content[0].current_state[3:6],
                        own_pos - other_pos) > 0 or np.dot(
                    self.get_targets()[other_drone_id] -
                    self.__trajectory_tracker.get_information(other_drone_id).content[0].current_state[0:3],
                    own_pos - other_pos) > 0:
                    return True
                if np.dot(self.get_targets()[other_drone_id][0:3] - other_pos[0:3],
                          own_pos[0:3] - other_pos[0:3]) > 0:
                    return True
        return False

    def round_started(self):
        self.print("round started!")
        start_time = time.time()
        if not self.__use_high_level_planner:
            return

        if self.__system_state != NORMAL:
            return

        # sometimes it happens (at the beginning of the expoeriment, when the drones stand still and send their current
        # position to the drone agents, the high level planner is in the wrong state)
        if len(self.__deadlock_breaker_agents) == 0:
            self.__using_intermediate_targets = False

        # if the CU is not the one calculating the high level targets, do not calculate them.
        if not self.__send_setpoints:  # and self.__simulated:
            return
        self.print("round started!!!!")
        # get the current position of all agents.
        current_pos = {}
        current_pos_list = []
        target_pos_list = []
        drone_idxs = {}
        for i, drone_id in enumerate(self.__drones_ids):
            if len(self.__trajectory_tracker.get_information(drone_id).content) != 0:
                trajectory = self.__trajectory_tracker.get_information(drone_id).content[0]
                if trajectory.current_state is not None:
                    current_pos[drone_id] = trajectory.current_state[0:3]
                    current_pos_list.append(trajectory.current_state[0:3])
                    target_pos_list.append(self.get_targets()[drone_id])
                    drone_idxs[drone_id] = i
                else:
                    current_pos[drone_id] = None
                    print("---------------------------------------")

        # check if we already know the target or current positions. If we do not know all (one drones has not sent it),
        # do not calculate. This usually happens, when a drone is freshly added to the swarm.
        for drone_id in self.__drones_ids:
            if self.get_targets()[drone_id] is None or current_pos[drone_id] is None:
                return

        # print(f"pppppppppppppppppp {self.__recalculate_setpoints}")
        # if we do not block the hlp, after it has been called, it will be called multiple times in a row,
        # because the drones need a while to move and it will otherwise think, the drones are in a deadlock again
        self.__hlp_lock += 1
        if self.__hlp_lock < 10 and not self.__recalculate_setpoints:
            return

        # check if all agents are close to target, then do nothing
        all_agents_close_to_target = True
        for drone_id in self.__drones_ids:
            if np.linalg.norm(
                    current_pos[drone_id] - self.get_targets()[drone_id]) > self.__options.r_min * 0.9:
                all_agents_close_to_target = False
        if all_agents_close_to_target:
            self.__high_level_setpoints = {}
            for drone_id in self.__drones_ids:
                self.__high_level_setpoints[drone_id] = self.get_targets()[drone_id]
            return

        # check if agents are in a deadlock. If they are, either try to break the deadlock (or if we already trying
        # to break the deadlock go back to the original targets).
        all_agents_in_deadlock = True
        for drone_id in self.__drones_ids:
            if np.linalg.norm(self.__trajectory_tracker.get_information(drone_id).content[0].current_state[3:6]) > 0.1:
                all_agents_in_deadlock = False
                break

        drone_dists_mult = np.tile(np.array(current_pos_list), (1, len(current_pos_list))) - np.tile(np.array([np.array(current_pos_list).flatten()]), (len(current_pos_list), 1))

        drone_dists_mult = np.abs(drone_dists_mult)
        drone_dists_mult = np.reshape(drone_dists_mult, (drone_dists_mult.size//3, 3))

        drone_dists_norm = np.linalg.norm((self.__downwash_scaling @ drone_dists_mult.T).T, axis=1)
        drone_dists_norm[drone_dists_norm<0.01] = 1e5

        drones_dist_to_target = np.linalg.norm(np.array(target_pos_list) - np.array(current_pos_list), axis=1)

        # check if not all deadlock breaker conditions are fulfilled, if not, then the deadlock is broken and we can
        # go back to the original targets.
        deadlock_broken = False
        if len(self.__deadlock_breaker_agents) > 0:
            deadlock_broken = True
            for dba in self.__deadlock_breaker_agents:
                if self.deadlock_breaker_condition(dba[0], dba[1],
                                                   own_pos=current_pos[dba[0]],
                                                   other_pos=current_pos[dba[1]],
                                                   inter_dist=drone_dists_norm[len(self.__drones_ids)*drone_idxs[dba[0]] + drone_idxs[dba[1]]],
                                                   own_dist_to_target=drones_dist_to_target[drone_idxs[dba[0]]],
                                                   other_agent_dist_to_target=drones_dist_to_target[drone_idxs[dba[1]]]
                                                   ):
                    deadlock_broken = False
                    break

        # if we now need to change targets, do it.
        if all_agents_in_deadlock or self.__recalculate_setpoints or deadlock_broken:
            # if we currently try to break a deadlock or received new setppoints, use real targets as "intermediate"
            # targets
            if self.__using_intermediate_targets or self.__recalculate_setpoints:
                self.__recalculate_setpoints = False
                self.__high_level_setpoints = {}
                for drone_id in self.__drones_ids:
                    self.__high_level_setpoints[drone_id] = self.get_targets()[drone_id]
                self.__using_intermediate_targets = 0
                self.__deadlock_breaker_agents = []
                # do not calculate new intermediate setpoints
                self.__hlp_lock = 0
                return
            else:
                self.__using_intermediate_targets = True
            self.__hlp_lock = 0
            self.print("Deadlock!!!!")
        else:
            # no need to calculate the high level planner
            return

        print(f"detect: {start_time - time.time()}")
        start_time = time.time()
        # calculate intermediate targets based on a similar algorithm given in Park et al. 2020
        self.__high_level_setpoints = {drone_id: self.get_targets()[drone_id] for drone_id in self.__drones_ids}
        self.__deadlock_breaker_agents = []

        for drone_id in self.__drones_ids:
            # first determine, which agent, the drone_id has to dodge.
            agents_to_dodge = []
            drones_dists = []
            for other_drone_id in self.__drones_ids:
                if other_drone_id == drone_id:
                    continue
                print(".")
                if self.deadlock_breaker_condition(drone_id, other_drone_id,
                                                   own_pos=current_pos[drone_id],
                                                   other_pos=current_pos[other_drone_id],
                                                   inter_dist=drone_dists_norm[len(self.__drones_ids)*drone_idxs[drone_id] + drone_idxs[other_drone_id]],
                                                   own_dist_to_target=drones_dist_to_target[drone_idxs[drone_id]],
                                                   other_agent_dist_to_target=drones_dist_to_target[drone_idxs[other_drone_id]]
                                                   ):
                    self.__deadlock_breaker_agents.append((drone_id, other_drone_id))
                    agents_to_dodge.append(drone_idxs[other_drone_id])
                    drones_dists.append(drone_dists_norm[len(self.__drones_ids)*drone_idxs[drone_id] + drone_idxs[other_drone_id]],
                                                   )

            if len(agents_to_dodge) != 0:
                # first determine the closest of the agents, we need to dodge and then dodge only this one.
                closest_agent_idx = np.argmin(np.array(drones_dists))
                closest_agent = self.__drones_ids[agents_to_dodge[closest_agent_idx]]
                closest_distance = drones_dists[closest_agent_idx]
                # if an agent is closer to hlp_threshold_distance, use some sort of potential function approach
                hlp_threshold_distance = 0.6
                if closest_distance < hlp_threshold_distance:
                    to_target = self.get_targets()[closest_agent] - current_pos[
                        closest_agent]  # self.__trajectory_tracker.get_information(closest_agent).content[0].current_state[3:6]
                    dodge_direction = current_pos[drone_id] - current_pos[closest_agent]
                    dodge_direction /= np.linalg.norm(dodge_direction)
                    # add some random noise. This helps breacking up deadlocks caused by symmetry.
                    self.__high_level_setpoints[drone_id] = current_pos[
                                                                drone_id] + dodge_direction * self.__agent_dodge_distance + np.random.randn(
                        3) * 0.1
            print(f"calc: {start_time - time.time()}")

    def __scaled_norm(self, vector):
        return np.linalg.norm(self.__downwash_scaling @ vector)

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
    def comp_agent_prio(self):
        return self.__comp_agent_prio

    @property
    def current_time(self):
        return self.__current_time

    @property
    def num_computing_agents(self):
        return self.__num_computing_agents

    @property
    def drone_ids(self):
        return self.__drones_ids

    @property
    def current_target_positions(self):
        return self.__current_target_positions

    @property
    def total_num_optimizer_runs(self):
        return self.__total_num_optimizer_runs

    @property
    def num_succ_optimizer_runs(self):
        return self.__num_succ_optimizer_runs

    @property
    def __send_setpoints(self):
        return self.ID == self.__computing_agents_ids[0]

    @property
    def simulate_quantization(self):
        return self.__simulate_quantization

    @simulate_quantization.setter
    def simulate_quantization(self, v):
        self.__simulate_quantization = v

    def get_trajectory_tracker(self):
        return self.__trajectory_tracker

    def get_agent_position(self, drone_id):
        return self.__trajectory_tracker.get_information(drone_id).content[0].current_state[0:3]

    def get_agent_input(self, drone_id):
        return self.__trajectory_tracker.get_information(drone_id).content[0].coefficients.coefficients

    def set_simulate_quantization(self, value):
        self.__simulate_quantization = value

    def set_current_time(self, value):
        self.__current_time = value

    def set_save_snapshot_times(self, value):
        self.__save_snapshot_times = value

    def get_pos(self, drone_idx):
        return self.__trajectory_tracker.get_information(self.__drones_ids[drone_idx]).content[0].current_state[0:3]


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
                 trajectory_generator_options, prediction_horizon, other_drones_ids, target_positions,
                 order_interpolation=4,
                 slot_group_state_id=None, slot_group_ack_id=100000, state_feedback_trigger_dist=0.5,
                 load_cus_round_nmbr=0,
                 trajectory_start_time=0,
                 trajectory_cu_id=-1, show_print=True):
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

        # breakpoints of optimization variable
        self.__breakpoints = \
            np.linspace(0, communication_delta_t * prediction_horizon, prediction_horizon * int(
                communication_delta_t / trajectory_generator_options.optimization_variable_sample_time) + 1)

        # planned for the next timestep
        self.__planned_trajectory_coefficients = tg.TrajectoryCoefficients(None, False,
                                                                           np.zeros((
                                                                               int(round(
                                                                                   prediction_horizon * communication_delta_t /
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

        self.__current_trajectory_calculated_by = trajectory_cu_id  # ID

        self.__send_trajectory_message_to = []

        self.__received_state_messages = []

        self.__agent_target_idx = 0

        self.__all_targets_reached = False

        self.__state_measured = False

        self.__current_setpoint = self.__init_state

        self.__show_print = show_print

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
            message = net.Message(self.ID, slot_group_id,
                                  StateMessageContent(self.__traj_state,
                                                      target_position=self.__target_positions[self.__agent_target_idx],
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
                                                                    prios=np.array([1 for _ in range(
                                                                        len(self.__other_drones_ids))]))))
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
            # if self.ID == 6:
            #     if 121 > self.__current_time / self.__communication_delta_t > 118:
            #         return
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
            # self.__traj_state[3:] = 0
            """if self.__planned_trajectory_coefficients.valid:
                self.__planned_trajectory_coefficients.coefficients = np.zeros(self.__planned_trajectory_coefficients.coefficients.shape)
            else:
                self.__planned_trajectory_coefficients.alternative_trajectory = np.zeros(
                    self.__planned_trajectory_coefficients.alternative_trajectory.shape)"""

    def print(self, text):
        if self.__show_print:
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
        for i in range(int(self.__prediction_horizon * self.__communication_delta_t / resolution)):
            state = self.__trajectory_interpolation.interpolate(
                self.__current_time - self.__planned_trajectory_start_time + resolution * i,
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
