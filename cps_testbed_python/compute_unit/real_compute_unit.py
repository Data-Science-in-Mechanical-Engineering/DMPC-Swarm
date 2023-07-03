import compute_unit.uart.uart_interface as uart_interface
from compute_unit.uart.user_input_thread import UserInputThread as UI
import compute_unit.agents as da
import compute_unit.trajectory_generation.trajectory_generation as tg
import network.network as net

import numpy as np
import time
import compute_unit.uart.message as message

import copy


def parallel_simulation_wrapper(computation_unit):
    computation_unit.run()

MAX_INPUT = 5.0

MAX_POSITION = 5.0
MAX_VELOCITY = 5.0
MAX_ACCELERATION = 5.0

def quantize_float(float, range):
    lower = np.min(range)
    upper = np.max(range)
    resolution = np.iinfo(np.uint16).max
    return np.round((float - lower) * resolution / (upper - lower)).astype(np.uint16)


def dequantize_float(integer, range):
    lower = np.min(range)
    upper = np.max(range)
    resolution = np.iinfo(np.uint16).max
    return lower + integer / resolution * (upper - lower)

def quantize_pos(pos):
    return quantize_float(pos, [-MAX_POSITION, MAX_POSITION])

def quantize_vel(vel):
    return quantize_float(vel, [-MAX_VELOCITY, MAX_VELOCITY])

def quantize_acc(acc):
    return quantize_float(acc, [-MAX_ACCELERATION, MAX_ACCELERATION])

def quantize_input(input):
    return quantize_float(input, [-MAX_INPUT, MAX_INPUT])

def dequantize_pos(pos):
    return dequantize_float(pos, [-MAX_POSITION, MAX_POSITION])

def dequantize_vel(vel):
    return dequantize_float(vel, [-MAX_VELOCITY, MAX_VELOCITY])

def dequantize_acc(acc):
    return dequantize_float(acc, [-MAX_ACCELERATION, MAX_ACCELERATION])

def dequantize_input(input):
    return dequantize_float(input, [-MAX_INPUT, MAX_INPUT])



# message types
TYPE_ERROR = 0
TYPE_METADATA = 1
TYPE_AP_ACK = 2
TYPE_CP_ACK = 3
TYPE_ALL_AGENTS_READY = 4
TYPE_AP_DATA_REQ = 5
TYPE_LAUNCH_DRONES = 6
TYPE_TRAJECTORY = 7
TYPE_DRONE_STATE = 8
TYPE_REQ_TRAJECTORY = 9
TYPE_SYS_SHUTDOWN = 10
TYPE_EMTPY_MESSAGE = 11
TYPE_START_SYNC_MOVEMENT = 12
TYPE_SYNC_MOVEMENT_MESSAGE = 13
TYPE_DUMMY = 255

# crazyflie status flags
STATUS_IDLE = 0
STATUS_LAUNCHING = 1
STATUS_LAUNCHED = 2
STATUS_FLYING = 3
STATUS_CURRENT_TARGET_REACHED = 4
STATUS_ALL_TARGETS_REACHED = 5
STATUS_LANDING = 6
STATUS_LANDED = 7

MAX_NUM_DRONES = 10
MAX_NUM_AGENTS = 15


class TrajectoryMessage(message.MessageType):

    def __init__(self):
        super().__init__(
            ["type", "id", "trajectory", "init_state", "trajectory_start_time", "drone_id", "calculated_by", "prios"],
            [("uint8_t", 1), ("uint8_t", 1), ("uint16_t", 45), ("uint16_t", 9), ("uint16_t", 1), ("uint8_t", 1),
             ("uint8_t", 1), ("uint8_t", MAX_NUM_DRONES)])
        self.set_content({"type": np.array([TYPE_TRAJECTORY], dtype=self.get_data_type("type"))})

    def set_content(self, content):
        if "type" in content:
            assert content["type"][0] == TYPE_TRAJECTORY
        super().set_content(content)

    @property
    def m_id(self):
        return self.get_content("id")[0]

    @m_id.setter
    def m_id(self, mid):
        self.set_content({"id": np.array([mid], dtype=self.get_data_type("id"))})

    @property
    def trajectory(self):
        trajectory_raw = self.get_content("trajectory")
        trajectory = np.zeros(len(trajectory_raw))

        for i in range(len(trajectory_raw)):
            trajectory[i] = dequantize_input(trajectory_raw[i])

        return trajectory.reshape((len(trajectory)//3, 3))

    @trajectory.setter
    def trajectory(self, trajectory):
        trajectory_flat = trajectory.reshape((trajectory.size,))
        trajectory_raw = np.zeros(trajectory_flat.size, dtype=self.get_data_type("trajectory"))

        for i in range(len(trajectory_raw)):
            trajectory_raw[i] = quantize_input(trajectory_flat[i])

        self.set_content({"trajectory": trajectory_raw})

    @property
    def init_state(self):
        state_raw = self.get_content("init_state")
        state = np.zeros(9)
        state[0] = dequantize_pos(state_raw[0])
        state[1] = dequantize_pos(state_raw[1])
        state[2] = dequantize_pos(state_raw[2])

        state[3] = dequantize_vel(state_raw[3])
        state[4] = dequantize_vel(state_raw[4])
        state[5] = dequantize_vel(state_raw[5])

        state[6] = dequantize_acc(state_raw[6])
        state[7] = dequantize_acc(state_raw[7])
        state[8] = dequantize_acc(state_raw[8])
        return state

    @init_state.setter
    def init_state(self, state):
        state_raw = np.zeros(9, dtype=np.uint16)
        state_raw[0] = quantize_pos(state[0])
        state_raw[1] = quantize_pos(state[1])
        state_raw[2] = quantize_pos(state[2])

        state_raw[3] = quantize_vel(state[3])
        state_raw[4] = quantize_vel(state[4])
        state_raw[5] = quantize_vel(state[5])

        state_raw[6] = quantize_acc(state[6])
        state_raw[7] = quantize_acc(state[7])
        state_raw[8] = quantize_acc(state[8])
        self.set_content({"init_state": state_raw})

    @property
    def trajectory_start_time(self):
        return self.get_content("trajectory_start_time")[0]

    @trajectory_start_time.setter
    def trajectory_start_time(self, trajectory_start_time):
        self.set_content({"trajectory_start_time": np.array([trajectory_start_time], dtype=self.get_data_type("trajectory_start_time"))})

    @property
    def drone_id(self):
        return self.get_content("drone_id")[0]

    @drone_id.setter
    def drone_id(self, trajectory_start_time):
        self.set_content({"drone_id": np.array([trajectory_start_time], dtype=self.get_data_type("drone_id"))})

    @property
    def calculated_by(self):
        return self.get_content("calculated_by")[0]

    @calculated_by.setter
    def calculated_by(self, calculated_by):
        self.set_content({"calculated_by": np.array([calculated_by], dtype=self.get_data_type("calculated_by"))})

    @property
    def prios(self):
        return self.get_content("prios")

    @prios.setter
    def prios(self, prios):
        if isinstance(prios, list):
            prios = copy.deepcopy(prios)
        else:
            prios = copy.deepcopy(prios.tolist())
        while len(prios) < MAX_NUM_DRONES:
            prios.append(0)
        self.set_content({"prios": np.array(prios, dtype=self.get_data_type("prios"))})


class StateMessage(message.MessageType):
    def __init__(self):
        super().__init__(
            ["type", "id", "state", "status", "current_target", "target_position_idx", "trajectory_start_time", "drone_id", "calculated_by"],
            [("uint8_t", 1), ("uint8_t", 1), ("uint16_t", 9), ("uint8_t", 1), ("uint16_t", 3), ("uint8_t", 1), ("uint16_t", 1),
             ("uint8_t", 1), ("uint8_t", 1)])
        self.set_content({"type": np.array([TYPE_DRONE_STATE], dtype=self.get_data_type("type"))})

    def set_content(self, content):
        if "type" in content:
            assert content["type"] == TYPE_DRONE_STATE
        super().set_content(content)

    @property
    def m_id(self):
        return self.get_content("id")[0]

    @m_id.setter
    def m_id(self, mid):
        self.set_content({"id": np.array([mid], dtype=self.get_data_type("id"))})

    @property
    def state(self):
        state_raw = self.get_content("state")
        state = np.zeros(9)
        state[0] = dequantize_pos(state_raw[0])
        state[1] = dequantize_pos(state_raw[1])
        state[2] = dequantize_pos(state_raw[2])

        state[3] = dequantize_vel(state_raw[3])
        state[4] = dequantize_vel(state_raw[4])
        state[5] = dequantize_vel(state_raw[5])

        state[6] = dequantize_acc(state_raw[6])
        state[7] = dequantize_acc(state_raw[7])
        state[8] = dequantize_acc(state_raw[8])
        return state

    @state.setter
    def state(self, state):
        state_raw = np.zeros(9, dtype=np.uint16)
        state_raw[0] = quantize_pos(state[0])
        state_raw[1] = quantize_pos(state[1])
        state_raw[2] = quantize_pos(state[2])

        state_raw[3] = quantize_vel(state[3])
        state_raw[4] = quantize_vel(state[4])
        state_raw[5] = quantize_vel(state[5])

        state_raw[6] = quantize_acc(state[6])
        state_raw[7] = quantize_acc(state[7])
        state_raw[8] = quantize_acc(state[8])
        self.set_content({"state": state_raw})

    @property
    def status(self):
        return self.get_content("status")[0]

    @status.setter
    def status(self, status):
        self.set_content({"status": np.array([status], dtype=self.get_data_type("status"))})

    @property
    def current_target(self):
        target_raw = self.get_content("current_target")
        target = np.zeros(3)
        target[0] = dequantize_pos(target_raw[0])
        target[1] = dequantize_pos(target_raw[1])
        target[2] = dequantize_pos(target_raw[2])
        return target

    @current_target.setter
    def current_target(self, target):
        target_raw = np.zeros(3, dtype=np.uint16)
        target_raw[0] = quantize_pos(target[0])
        target_raw[1] = quantize_pos(target[1])
        target_raw[2] = quantize_pos(target[2])
        self.set_content({"current_target": target_raw})

    @property
    def target_position_idx(self):
        return self.get_content("target_position_idx")[0]

    @target_position_idx.setter
    def target_position_idx(self, target_position_idx):
        self.set_content({"target_position_idx": np.array([target_position_idx], dtype=self.get_data_type("target_position_idx"))})

    @property
    def trajectory_start_time(self):
        return self.get_content("trajectory_start_time")[0]

    @trajectory_start_time.setter
    def trajectory_start_time(self, trajectory_start_time):
        self.set_content({"trajectory_start_time": np.array([trajectory_start_time], dtype=self.get_data_type("trajectory_start_time"))})

    @property
    def drone_id(self):
        return self.get_content("drone_id")[0]

    @drone_id.setter
    def drone_id(self, mdrone_id):
        self.set_content({"drone_id": np.array([mdrone_id], dtype=self.get_data_type("drone_id"))})

    @property
    def calculated_by(self):
        return self.get_content("calculated_by")[0]

    @calculated_by.setter
    def calculated_by(self, mcalculated_by):
        self.set_content({"calculated_by": np.array([mcalculated_by], dtype=self.get_data_type("calculated_by"))})


class TrajectoryReqMessage(message.MessageType):

    def __init__(self):
        super().__init__(
            ["type", "id", "drone_id", "cu_id"],
            [("uint8_t", 1), ("uint8_t", 1), ("uint8_t", 1), ("uint8_t", 1)])
        self.set_content({"type": np.array([TYPE_REQ_TRAJECTORY], dtype=self.get_data_type("type"))})

    def set_content(self, content):
        if "type" in content:
            assert content["type"] == TYPE_REQ_TRAJECTORY
        super().set_content(content)

    @property
    def m_id(self):
        return self.get_content("id")[0]

    @m_id.setter
    def m_id(self, mid):
        self.set_content({"id": np.array([mid], dtype=self.get_data_type("id"))})

    @property
    def drone_id(self):
        return self.get_content("drone_id")[0]

    @drone_id.setter
    def drone_id(self, drone_id):
        self.set_content({"drone_id": np.array([drone_id], dtype=self.get_data_type("drone_id"))})

    @property
    def cu_id(self):
        return self.get_content("cu_id")[0]

    @cu_id.setter
    def cu_id(self, cu_id):
        self.set_content({"cu_id": np.array([cu_id], dtype=self.get_data_type("cu_id"))})


class EmptyMessage(message.MessageType):

    def __init__(self):
        super().__init__(
            ["type", "id", "cu_id"],
            [("uint8_t", 1), ("uint8_t", 1), ("uint8_t", 1)])
        self.set_content({"type": np.array([TYPE_EMTPY_MESSAGE], dtype=self.get_data_type("type"))})

    def set_content(self, content):
        if "type" in content:
            assert content["type"] == TYPE_EMTPY_MESSAGE
        super().set_content(content)

    @property
    def m_id(self):
        return self.get_content("id")[0]

    @m_id.setter
    def m_id(self, mid):
        self.set_content({"id": np.array([mid], dtype=self.get_data_type("id"))})

    @property
    def cu_id(self):
        return self.get_content("cu_id")[0]

    @cu_id.setter
    def cu_id(self, cu_id):
        self.set_content({"cu_id": np.array([cu_id], dtype=self.get_data_type("cu_id"))})


class MetadataMessage(message.MessageType):
    def __init__(self):
        super().__init__(["type", "id", "num_computing_units", "num_drones", "round_length_ms", "own_id", "round_mbr"],
                         [("uint8_t", 1), ("uint8_t", 1), ("uint8_t", 1), ("uint8_t", 1), ("uint16_t", 1),
                          ("uint8_t", 1), ("uint16_t", 1)])

    @property
    def type(self):
        return self.get_content("type")[0]

    @type.setter
    def type(self, type):
        self.set_content({"type": np.array([type], dtype=self.get_data_type("type"))})

    @property
    def m_id(self):
        return self.get_content("id")[0]

    @m_id.setter
    def m_id(self, mid):
        self.set_content({"id": np.array([mid], dtype=self.get_data_type("id"))})

    @property
    def num_computing_units(self):
        return self.get_content("num_computing_units")[0]

    @num_computing_units.setter
    def num_computing_units(self, num_computing_units):
        self.set_content({"num_computing_units": np.array([num_computing_units], dtype=self.get_data_type("num_computing_units"))})

    @property
    def num_drones(self):
        return self.get_content("num_drones")[0]

    @num_drones.setter
    def num_drones(self, num_drones):
        self.set_content({"num_drones": np.array([num_drones], dtype=self.get_data_type("num_drones"))})

    @property
    def round_length_ms(self):
        return self.get_content("round_length_ms")[0]

    @round_length_ms.setter
    def round_length_ms(self, round_length_ms):
        self.set_content({"round_length_ms": np.array([round_length_ms], dtype=self.get_data_type("round_length_ms"))})

    @property
    def own_id(self):
        return self.get_content("own_id")[0]

    @own_id.setter
    def own_id(self, own_id):
        self.set_content({"own_id": np.array([own_id], dtype=self.get_data_type("own_id"))})

    @property
    def round_mbr(self):
        return self.get_content("round_mbr")[0]

    @round_mbr.setter
    def round_mbr(self, round_mbr):
        self.set_content({"round_mbr": np.array([round_mbr], dtype=self.get_data_type("round_mbr"))})


class NetworkMemberChanged(message.MessageType):
    def __init__(self):
        super().__init__(["type", "id", "agent_id", "agent_type"],
                         [("uint8_t", 1), ("uint8_t", 1), ("uint8_t", 1), ("uint8_t", 1)])

    @property
    def type(self):
        return self.get_content("type")[0]

    @type.setter
    def type(self, type):
        self.set_content({"type": np.array([type], dtype=self.get_data_type("type"))})

    @property
    def m_id(self):
        return self.get_content("id")[0]

    @m_id.setter
    def m_id(self, mid):
        self.set_content({"id": np.array([mid], dtype=self.get_data_type("id"))})

    @property
    def agent_id(self):
        return self.get_content("agent_id")[0]

    @agent_id.setter
    def agent_id(self, agent_id):
        self.set_content({"agent_id": np.array([agent_id], dtype=self.get_data_type("agent_id"))})

    @property
    def agent_type(self):
        return self.get_content("agent_type")[0]

    @agent_type.setter
    def agent_type(self, agent_type):
        self.set_content({"agent_type": np.array([agent_type], dtype=self.get_data_type("agent_type"))})


class NetworkMembersMessage(message.MessageType):
    def __init__(self):
        super().__init__(["type", "id", "agent_ids", "agent_types"],
                         [("uint8_t", 1), ("uint8_t", 1), ("uint8_t", MAX_NUM_AGENTS), ("uint8_t", MAX_NUM_AGENTS)])

    @property
    def type(self):
        return self.get_content("type")[0]

    @type.setter
    def type(self, type):
        self.set_content({"type": np.array([type], dtype=self.get_data_type("type"))})

    @property
    def m_id(self):
        return self.get_content("id")[0]

    @m_id.setter
    def m_id(self, mid):
        self.set_content({"id": np.array([mid], dtype=self.get_data_type("id"))})

    @property
    def agent_ids(self):
        return self.get_content("agent_id")

    @agent_ids.setter
    def agent_id(self, agent_ids):
        self.set_content({"agent_ids": np.array(agent_ids, dtype=self.get_data_type("agent_ids"))})

    @property
    def agent_types(self):
        return self.get_content("agent_types")

    @agent_types.setter
    def agent_type(self, agent_types):
        self.set_content({"agent_types": np.array(agent_types, dtype=self.get_data_type("agent_type"))})


class SyncMovementMessage(message.MessageType):
    def __init__(self):
        super().__init__(["type", "id", "angle"],
                         [("uint8_t", 1), ("uint8_t", 1), ("uint16_t", 1)])
        self.set_content({"type": np.array([TYPE_SYNC_MOVEMENT_MESSAGE], dtype=self.get_data_type("type"))})

    @property
    def type(self):
        return self.get_content("type")[0]

    @type.setter
    def type(self, type):
        self.set_content({"type": np.array([type], dtype=self.get_data_type("type"))})

    @property
    def m_id(self):
        return self.get_content("id")[0]

    @m_id.setter
    def m_id(self, mid):
        self.set_content({"id": np.array([mid], dtype=self.get_data_type("id"))})

    @property
    def angle(self):
        return self.get_content("angle")[0]

    @angle.setter
    def angle(self, angle):
        self.set_content({"angle": np.array([angle], dtype=self.get_data_type("angle"))})


class ComputingUnit:
    """ Class that represents a computing unit. Each computing unit consists of a computation agent for data
    calculation and an uart interface for communication. The Computing Unit communicates with the computation agent
    by using the net.Message class. For the UART communication, the message.MixerMessage class is used. """

    def __init__(self, ARGS, cu_id, sync_movement=False, loose_messages=False, baudrate=921600):
        self.__ARGS = ARGS
        if cu_id not in ARGS.computing_agent_ids:
            raise ValueError('cu_id not in ARGS computation agent IDs')
        self.__cu_id = cu_id
        self.__sync_movement = sync_movement
        self.__loose_messages = loose_messages
        self.__message_type_trajectory_id = 0
        self.__message_type_trajectory_initital_state = 1
        self.__message_type_drone_state = 2

        self.__uart_interface = None
        self.__computation_agent = None

        self.init_uart_interface(baudrate)
        self.init_computation_agent()

        self.logging_freq = 1.0 / ARGS.communication_freq_hz

        self.__cu_idx = None
        for i in range(0, len(self.__ARGS.computing_agent_ids)):
            if self.__cu_id == self.__ARGS.computing_agent_ids[i]:
                self.__cu_idx = i
                break

        """
        if self.__cu_id == np.min(ARGS.computing_agent_ids) and (
                ARGS.log_planned_trajectory or ARGS.plot_planned_trajectory):

            self.HOST = "127.0.0.1"
            self.PORT = 65432
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.HOST, self.PORT))

            self.logger = None

            if ARGS.plot_planned_trajectory:
                self.plotter = None


        else:
        """
        self.logger = None
        self.plotter = None
        self.socket = None

        self.__drones_in_swarm = []

    def run_dynamic_swarm(self, fileno):
        self.connect_to_cp()

        """ DEFINE FREQUENTLY USED MESSAGES """
        ack_message = MetadataMessage()
        ack_message.type = TYPE_AP_ACK
        ack_message.m_id = self.__cu_id
        ack_message.num_computing_units = 0
        ack_message.num_drones = 0
        ack_message.round_length_ms = 200
        ack_message.own_id = self.__cu_id
        ack_message.round_mbr = 0

        STATE_IDLE = 0
        STATE_SYS_RUN = 3

        state = STATE_IDLE

        while True:
            new_round = False
            messages_rx = self.read_data_from_cp()
            messages_tx = []

            # untill the first drone is into the swarm, dont do anything.
            if state == STATE_IDLE:
                # wait until cp says all agents are ready
                for m in messages_rx:
                    if isinstance(m, StateMessage):
                        state = STATE_SYS_RUN
                        self.__uart_interface.print("First UAV connected")
                messages_tx = [ack_message]
            elif state == STATE_SYS_RUN:
                # check if a new agent is inside the swarm
                for m in messages_rx:
                    if isinstance(m, StateMessage):
                        if m.m_id not in self.__drones_in_swarm:
                            self.__computation_agent.add_new_agent(m.m_id)
                            self.__drones_in_swarm.append(m.m_id)

                messages_tx = self.dmpc_step(messages_rx)

            # send data to CP
            self.write_data_to_cp(messages_tx)

    def connect_to_cp(self):
        """ DEFINE FREQUENTLY USED MESSAGES """
        print("Start Connecting")
        ack_message = message.MixerMessage(message_type="TYPE_AP_ACK", agent_ID=self.__cu_id,
                                           data=np.array([1], dtype=np.uint8))

        # connect to CP
        self.uart_interface.initialize()  # initialize communication with CP
        ack_message = MetadataMessage()
        ack_message.type = TYPE_AP_ACK
        ack_message.m_id = self.__cu_id
        ack_message.num_computing_units = 0
        ack_message.num_drones = 0
        ack_message.round_length_ms = 200
        ack_message.own_id = self.__cu_id
        ack_message.round_mbr = 0
        # time.sleep(102e-3)  # the cp sleeps a short time, thus, we may have to sleep a bit longer
        self.write_data_to_cp([ack_message])

    def dmpc_step(self, messages_rx):
        messages_parsed = []
        messages_tx = []
        num_all_targets_reached = 0
        round_mbr = None
        for m in messages_rx:
            m_temp = None
            if isinstance(m, EmptyMessage):
                content_temp = da.EmtpyContent(empty=0)
                m_temp = net.Message(ID=m.m_id, slot_group_id=self.__message_type_trajectory_id,
                                     content=content_temp)
            elif isinstance(m, StateMessage):
                content_temp = da.StateMessageContent(state=m.state, target_position=m.current_target,
                                                      trajectory_start_time=m.trajectory_start_time / self.__ARGS.communication_freq_hz,
                                                      trajectory_calculated_by=m.calculated_by,
                                                      target_position_idx=m.target_position_idx,
                                                      coefficients=None, init_state=None)
                m_temp = net.Message(ID=m.m_id, slot_group_id=self.__message_type_drone_state,
                                     content=content_temp)
                if m.status == STATUS_ALL_TARGETS_REACHED:
                    num_all_targets_reached += 1
            elif isinstance(m, TrajectoryMessage):
                content_temp = da.TrajectoryMessageContent(id=m.drone_id, coefficients=tg.TrajectoryCoefficients(
                    coefficients=m.trajectory, valid=True, alternative_trajectory=None),
                                                           init_state=m.init_state,
                                                           trajectory_start_time=m.trajectory_start_time / self.__ARGS.communication_freq_hz,
                                                           trajectory_calculated_by=m.calculated_by,
                                                           prios=m.prios[0:len(self.__drones_in_swarm)])
                m_temp = net.Message(ID=m.m_id, slot_group_id=self.__message_type_trajectory_id,
                                     content=content_temp)
                print("Trajectory received")
                print(m.calculated_by)
                print(m.drone_id)
                print(m.trajectory_start_time / self.__ARGS.communication_freq_hz)
            elif isinstance(m, TrajectoryReqMessage):
                content_temp = da.RecoverInformationNotifyContent(cu_id=m.cu_id, drone_id=m.drone_id)
                m_temp = net.Message(ID=m.m_id, slot_group_id=self.__message_type_trajectory_id,
                                     content=content_temp)
            elif isinstance(m, MetadataMessage):
                if m.type == TYPE_METADATA:
                    round_mbr = m.round_mbr
                #if m.type == TYPE_SYS_SHUTDOWN:
                #    state = STATE_SYS_SHUTDOWN
            messages_parsed.append(m_temp)

        for m in messages_parsed:
            self.__computation_agent.send_message(m)
        # TRAJECTORY COMPUTATION
        self.uart_interface.print("NEW ROUND " + str(round_mbr))
        self.__computation_agent.round_finished(round_mbr)  # calculate a new trajectory
        print("round_finished")
        # read data from computing agent.
        traj_message = self.__computation_agent.get_message(self.__message_type_trajectory_id)
        if traj_message is None:
            m_temp = MetadataMessage()
            m_temp.type = TYPE_DUMMY
            m_temp.m_id = self.__cu_id
            m_temp.num_computing_units = 0
            m_temp.num_drones = 0
            m_temp.round_length_ms = 200
            m_temp.own_id = self.__cu_id
            m_temp.round_mbr = 0
            messages_tx.append(m_temp)
        elif isinstance(traj_message.content, da.TrajectoryMessageContent):
            m_temp = TrajectoryMessage()
            m_temp.m_id = traj_message.ID
            m_temp.trajectory = traj_message.content.coefficients.coefficients if traj_message.content.coefficients.valid else traj_message.content.coefficients.alternative_trajectory
            m_temp.init_state = traj_message.content.init_state
            m_temp.trajectory_start_time = int(
                round(traj_message.content.trajectory_start_time * self.__ARGS.communication_freq_hz))
            m_temp.calculated_by = traj_message.content.trajectory_calculated_by
            m_temp.drone_id = traj_message.content.id
            m_temp.prios = traj_message.content.prios
            messages_tx.append(m_temp)
        elif isinstance(traj_message.content, da.EmtpyContent):
            m_temp = EmptyMessage()
            m_temp.m_id = traj_message.ID
            m_temp.cu_id = self.__cu_id
            messages_tx.append(m_temp)
            print("Empty")
        elif isinstance(traj_message.content, da.RecoverInformationNotifyContent):
            m_temp = TrajectoryReqMessage()
            m_temp.m_id = traj_message.ID
            m_temp.drone_id = traj_message.content.drone_id
            m_temp.cu_id = traj_message.content.cu_id
            print(f"Requesting new trajectory! {m_temp.drone_id}, {traj_message.content.drone_id}, {m_temp.cu_id}")
            messages_tx.append(m_temp)

        return messages_tx

    def run(self, fileno):
        """ This is the main state machine of the computing unit. """

        """ SETUP """

        if self.__ARGS.dynamic_swarm:
            self.run_dynamic_swarm(fileno)

        # SETUP THE USER INPUT THREAD TO REGISTER KEYBOARD INPUTS
        if self.__cu_id == np.min(self.__ARGS.computing_agent_ids):
            user_input_thread = UI(fileno)
            user_input_thread.start()


        statusCodes = {0: "IDLE", 1: "AP_READY", 2: "SYS_READY", 3: "LAUNCH", 4: "SYS_RUN", 5: "SYS_SHUTDOWN"}
        STATE_IDLE = 0
        STATE_SYS_READY = 1
        STATE_LAUNCH = 2
        STATE_SYNC_MOVEMENT = 5
        STATE_SYS_RUN = 3
        STATE_SYS_SHUTDOWN = 4

        state = STATE_IDLE

        start = time.time()
        sys_rdy_time = None
        launch_time = None
        user_input_timeout = 5

        """ DEFINE FREQUENTLY USED MESSAGES """
        ack_message = message.MixerMessage(message_type="TYPE_AP_ACK", agent_ID=self.__cu_id, data=np.array([1], dtype=np.uint8))
        start_drones_message = message.MixerMessage(message_type="TYPE_LAUNCH_DRONES",
                                                    agent_ID=self.__cu_id + 1,
                                                    data=np.array([1], dtype=np.uint8))
        start_system_message = message.MixerMessage(message_type="TYPE_SYS_RUN",
                                                    agent_ID=self.__cu_id + 1, data=np.array([1], dtype=np.uint8))
        shutdown_message = message.MixerMessage(agent_ID=self.__cu_id + 1,
                                                message_type="TYPE_SHUTDOWN")

        # connect to CP
        self.uart_interface.initialize()  # initialize communication with CP
        ack_message = MetadataMessage()
        ack_message.type = TYPE_AP_ACK
        ack_message.m_id = self.__cu_id
        ack_message.num_computing_units = 0
        ack_message.num_drones = 0
        ack_message.round_length_ms = 200
        ack_message.own_id = self.__cu_id
        ack_message.round_mbr = 0
        """# because UART has no master/slave, if we send and the CP is not ready, the message will get lost.
        # thus the CP first sends a 2 byte request and we wait till we receive this request
        while np.frombuffer(self.uart_interface.read_from_uart_raw(2), dtype=np.uint16)[0] != 0:
            pass
        # sleep a short time to give CP time to start listening
        time.sleep(100e-6)"""
        #time.sleep(102e-3)  # the cp sleeps a short time, thus, we may have to sleep a bit longer
        self.write_data_to_cp([ack_message])

        sync_movement_angle = 0
        run_time = 0
        """ STATE MACHINE """
        while True:
            new_round = False
            messages_rx = self.read_data_from_cp()
            messages_tx = []

            if state == STATE_IDLE:
                # wait until cp says all agents are ready
                for m in messages_rx:
                    if isinstance(m, MetadataMessage):
                        if m.type == TYPE_ALL_AGENTS_READY:
                            state = STATE_SYS_READY
                            self.__uart_interface.print("All agents ready!")
                messages_tx = [ack_message]

            elif state == STATE_SYS_READY:
                if self.__cu_id == np.min(self.__ARGS.computing_agent_ids):
                    # wait till user says launch or aborts
                    if user_input_thread.state != state and user_input_thread.user_input != 0:
                        self.uart_interface.print("Launch drones?")
                        # sleep a short time such that the console prints and then the input starts. Sometimes they
                        # collide and cause a failure.
                        time.sleep(0.01)
                        user_input_thread.start_user_input(state)
                        messages_tx = [ack_message]

                    if user_input_thread.state == state and user_input_thread.user_input == 1:
                        self.uart_interface.print("SENDING LAUNCH DRONES MESSAGE")
                        ack_message.type = TYPE_LAUNCH_DRONES
                        messages_tx = [ack_message]
                        state = STATE_LAUNCH
                    elif user_input_thread.state == state and user_input_thread.user_input == 2:
                        self.uart_interface.print("SENDING SHUTDOWN MESSAGE")
                        ack_message.type = TYPE_SYS_SHUTDOWN
                        messages_tx = [ack_message]
                        state = STATE_SYS_SHUTDOWN
                    else:
                        messages_tx = [ack_message]
                else:
                    # if an agent is launching,change state
                    for m in messages_rx:
                        if isinstance(m, StateMessage):
                            if m.status == STATUS_LAUNCHED or m.status == STATUS_LAUNCHING:
                                state = STATE_LAUNCH
                                ack_message.type = TYPE_LAUNCH_DRONES
                                self.uart_interface.print("Drones are launching")
                        elif isinstance(m, MetadataMessage):
                            if m.type == TYPE_LAUNCH_DRONES:
                                state = STATE_LAUNCH
                                ack_message.type = TYPE_LAUNCH_DRONES
                                self.uart_interface.print("Drones launched")
                    messages_tx = [ack_message]
            elif state == STATE_LAUNCH:
                # count how many drones said they have launched. If the number of launched drones is equal to
                # number of drones
                num_drones = self.__ARGS.num_drones
                num_launched_drones = 0
                for m in messages_rx:
                    if isinstance(m, StateMessage):
                        if m.status == STATUS_LAUNCHED:
                            num_launched_drones += 1

                if num_launched_drones == num_drones:
                    self.uart_interface.print("All drones launched sucessfully.")
                    if self.__sync_movement:
                        state = STATE_SYNC_MOVEMENT
                        sync_movement_angle = 0
                    else:
                        state = STATE_SYS_RUN
                # continue sending the launch message or start the sync movement
                ack_message.type = TYPE_LAUNCH_DRONES if state != STATE_SYNC_MOVEMENT else TYPE_START_SYNC_MOVEMENT
                messages_tx = [ack_message]
            elif state == STATE_SYNC_MOVEMENT:
                m_temp = SyncMovementMessage()
                m_temp.m_id = self.__cu_id
                m_temp.angle = sync_movement_angle
                messages_tx = [m_temp]
                sync_movement_angle += 1
            elif state == STATE_SYS_RUN:
                start = time.time()
                messages_parsed = []
                num_all_targets_reached = 0
                for m in messages_rx:
                    if self.__loose_messages:
                        if (not (m.m_id == self.__cu_id or m.m_id == self.__ARGS.drone_ids[self.__cu_idx])) and (run_time > 0 and run_time < 10):
                            pass
                            #continue
                    m_temp = None
                    round_mbr = None
                    if isinstance(m, EmptyMessage):
                        content_temp = da.EmtpyContent(empty=0)
                        m_temp = net.Message(ID=m.m_id, slot_group_id=self.__message_type_trajectory_id,
                                             content=content_temp)
                    elif isinstance(m, StateMessage):
                        content_temp = da.StateMessageContent(state=m.state, target_position=m.current_target,
                                                              trajectory_start_time=m.trajectory_start_time/self.__ARGS.communication_freq_hz,
                                                              trajectory_calculated_by=m.calculated_by,
                                                              target_position_idx=m.target_position_idx,
                                                              coefficients=None, init_state=None)
                        m_temp = net.Message(ID=m.m_id, slot_group_id=self.__message_type_drone_state,
                                             content=content_temp)
                        if m.status == STATUS_ALL_TARGETS_REACHED:
                            num_all_targets_reached += 1
                    elif isinstance(m, TrajectoryMessage):
                        content_temp = da.TrajectoryMessageContent(id=m.drone_id, coefficients=tg.TrajectoryCoefficients(coefficients=m.trajectory, valid=True, alternative_trajectory=None),
                                                                   init_state=m.init_state,
                                                                   trajectory_start_time=m.trajectory_start_time/self.__ARGS.communication_freq_hz,
                                                                   trajectory_calculated_by=m.calculated_by, prios=m.prios[0:num_drones])
                        m_temp = net.Message(ID=m.m_id, slot_group_id=self.__message_type_trajectory_id,
                                             content=content_temp)
                        print("Trajectory received")
                        print(m.calculated_by)
                        print(m.drone_id)
                        print(m.trajectory_start_time/self.__ARGS.communication_freq_hz)
                    elif isinstance(m, TrajectoryReqMessage):
                        content_temp = da.RecoverInformationNotifyContent(cu_id=m.cu_id, drone_id=m.drone_id)
                        m_temp = net.Message(ID=m.m_id, slot_group_id=self.__message_type_trajectory_id,
                                             content=content_temp)
                    elif isinstance(m, MetadataMessage):
                        round_mbr = m.round_mbr
                        if m.type == TYPE_SYS_SHUTDOWN:
                            state = STATE_SYS_SHUTDOWN
                    messages_parsed.append(m_temp)

                for m in messages_parsed:
                    self.__computation_agent.send_message(m)
                # TRAJECTORY COMPUTATION
                self.uart_interface.print("NEW ROUND " + str(round_mbr))
                self.__computation_agent.round_finished(round_mbr)  # calculate a new trajectory
                print("round_finished")
                # read data from computing agent.
                traj_message = self.__computation_agent.get_message(self.__message_type_trajectory_id)
                if traj_message is None:
                    m_temp = MetadataMessage()
                    m_temp.type = TYPE_DUMMY
                    m_temp.m_id = self.__cu_id
                    m_temp.num_computing_units = 0
                    m_temp.num_drones = 0
                    m_temp.round_length_ms = 200
                    m_temp.own_id = self.__cu_id
                    m_temp.round_mbr = 0
                    messages_tx.append(m_temp)
                elif isinstance(traj_message.content, da.TrajectoryMessageContent):
                    m_temp = TrajectoryMessage()
                    m_temp.m_id = traj_message.ID
                    m_temp.trajectory = traj_message.content.coefficients.coefficients if traj_message.content.coefficients.valid else traj_message.content.coefficients.alternative_trajectory
                    m_temp.init_state = traj_message.content.init_state
                    m_temp.trajectory_start_time = int(round(traj_message.content.trajectory_start_time * self.__ARGS.communication_freq_hz))
                    m_temp.calculated_by = traj_message.content.trajectory_calculated_by
                    m_temp.drone_id = traj_message.content.id
                    m_temp.prios = traj_message.content.prios
                    messages_tx.append(m_temp)
                elif isinstance(traj_message.content, da.EmtpyContent):
                    m_temp = EmptyMessage()
                    m_temp.m_id = traj_message.ID
                    m_temp.cu_id = self.__cu_id
                    messages_tx.append(m_temp)
                    print("Empty")
                elif isinstance(traj_message.content, da.RecoverInformationNotifyContent):
                    m_temp = TrajectoryReqMessage()
                    m_temp.m_id = traj_message.ID
                    m_temp.drone_id = traj_message.content.drone_id
                    m_temp.cu_id = traj_message.content.cu_id
                    print(f"Requesting new trajectory! {m_temp.drone_id}, {traj_message.content.drone_id}, {m_temp.cu_id}")
                    messages_tx.append(m_temp)

                # all drones have reached all targets, send a land message
                if num_all_targets_reached == self.__ARGS.num_drones or self.__computation_agent.all_targets_reached:
                    state = STATE_SYS_SHUTDOWN
                run_time += 1
            elif state == STATE_SYS_SHUTDOWN:
                print("Shutdown")
                num_landed_drones = 0
                for m in messages_rx:
                    if isinstance(m, StateMessage):
                        if m.status == STATUS_LANDED:
                            num_landed_drones += 1
                print(f"num_landed_drones: {num_landed_drones}")
                if num_landed_drones == self.__ARGS.num_drones:
                    state = STATE_SYS_READY
                    user_input_thread.state = STATE_SYS_SHUTDOWN

                ack_message.type = TYPE_SYS_SHUTDOWN
                messages_tx.append(ack_message)

            """# because UART has no master/slave, if we send and the CP is not ready, the message will get lost.
            # thus the CP first sends a 2 byte request and we wait till we receive this request
            while np.frombuffer(self.uart_interface.read_from_uart_raw(2), dtype=np.uint16)[0] != 0:
                pass
            # sleep a short time to give CP time to start listening
            time.sleep(100e-6)"""
            # send data to CP
            self.write_data_to_cp(messages_tx)

            if state == STATE_SYS_RUN:
                self.__computation_agent.round_started()


    def init_uart_interface(self, baudrate):
        """ initializes the UART interface for MIXER communication """

        self.uart_interface = uart_interface.UartInterface(id=self.__cu_id,
                                                           baudrate=baudrate)

    def read_data_from_cp(self):
        messages_received = []
        data = self.uart_interface.read_from_uart()
        data_idx = 0
        while data_idx < len(data):
            type = int.from_bytes([data[data_idx]], 'little')
            message_rec = None
            if type == TYPE_TRAJECTORY:
                message_rec = TrajectoryMessage()
            elif type == TYPE_DRONE_STATE:
                message_rec = StateMessage()
            elif type == TYPE_EMTPY_MESSAGE:
                message_rec = EmptyMessage()
            elif type == TYPE_REQ_TRAJECTORY:
                message_rec = TrajectoryReqMessage()
            elif type == TYPE_SYNC_MOVEMENT_MESSAGE:
                message_rec = SyncMovementMessage()
            else:
                message_rec = MetadataMessage()
            message_rec.set_content_bytes(data[data_idx:data_idx+message_rec.size])
            data_idx += message_rec.size
            messages_received.append(message_rec)
        return messages_received

    def write_data_to_cp(self, messages):
        # because UART has no master/slave, if we send and the CP is not ready, the message will get lost.
        # thus the CP first sends a 2 byte request and we wait till we receive this request
        while np.frombuffer(self.uart_interface.read_from_uart_raw(2), dtype=np.uint16)[0] != 0:
            pass
        # sleep a short time to give CP time to start listening
        time.sleep(100e-6)

        b_array = []
        for m in messages:
            b_array += m.to_bytes()
        self.uart_interface.send_to_uart(b_array)

    def init_computation_agent(self):
        """ initializes the computation agent of this CU """

        delta_t = 1.0 / self.__ARGS.communication_freq_hz
        collision_constraint_sample_points = np.linspace(0, self.__ARGS.prediction_horizon * delta_t,
                                                         int((self.__ARGS.prediction_horizon * 2 + 1)))
        collision_constraint_sample_points = collision_constraint_sample_points[1:]

        trajectory_generator_options = tg.TrajectoryGeneratorOptions(
            # different amount of sample point to increase resolution
            objective_function_sample_points=np.linspace(delta_t, self.__ARGS.prediction_horizon * delta_t,
                                                         int((self.__ARGS.prediction_horizon * 1))),
            objective_function_sample_points_pos=np.array([self.__ARGS.prediction_horizon * delta_t]),
            state_constraint_sample_points=np.linspace(delta_t, self.__ARGS.prediction_horizon * delta_t,
                                                       int((self.__ARGS.prediction_horizon * 1))),
            collision_constraint_sample_points=collision_constraint_sample_points,
            weight_state_difference=np.eye(3) * 1,
            weight_state_derivative_1_difference=np.eye(3) * 0.0001,
            weight_state_derivative_2_difference=np.eye(3) * 0.05,
            weight_state_derivative_3_difference=np.eye(3) * 0.01,
            max_speed=np.array([2.0, 2.0, 2.0]),  # np.array([1.5, 1.5, 1.5])
            max_position=self.__ARGS.max_positions,  # =np.array([1.5, 1.5, 3])
            # np.array(self.__ARGS.testbed.edges()[1]),  # np.array([10.0, 10.0, 100.0]),
            max_acceleration=np.array([5.0, 5.0, 5.0]),  # np.array([5, 5, 5])
            max_jerk=np.array([5.0, 5.0, 5.0]),
            min_position=self.__ARGS.min_positions,  # =np.array([-1.5, -1.5, 0.1])
            # np.array(self.__ARGS.testbed.edges()[0]),  # np.array([-10.0, -10.0, 0.5]),
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
            cooperative_normal_vector_noise=0
        )

        self.computation_agent = da.ComputationAgent(ID=self.__cu_id,
                                                     slot_group_planned_trajectory_id=self.__message_type_trajectory_id,
                                                     slot_group_trajectory_initial_state=self.__message_type_trajectory_initital_state,
                                                     slot_group_drone_state=self.__message_type_drone_state,
                                                     init_positions=self.__ARGS.INIT_XYZS,
                                                     target_positions=self.__ARGS.INIT_TARGETS,
                                                     agents_ids=self.__ARGS.drone_ids, communication_delta_t=delta_t,
                                                     trajectory_generator_options=trajectory_generator_options,
                                                     pos_offset=self.__ARGS.pos_offset,
                                                     prediction_horizon=self.__ARGS.prediction_horizon,
                                                     num_computing_agents=self.__ARGS.num_computing_agents,
                                                     comp_agent_prio=sorted(self.__ARGS.computing_agent_ids).index(
                                                         self.__cu_id),
                                                     offset=(self.__cu_id - self.__ARGS.num_drones) * int(
                                                         self.__ARGS.num_drones / max(
                                                             (self.__ARGS.num_computing_agents), 1)),
                                                     use_event_trigger=self.__ARGS.event_trigger,
                                                     alpha_1=self.__ARGS.alpha_1, alpha_2=self.__ARGS.alpha_2,
                                                     alpha_3=self.__ARGS.alpha_3, alpha_4=self.__ARGS.alpha_4,
                                                     remove_redundant_constraints=self.__ARGS.remove_redundant_constraints,
                                                     computing_agents_ids=self.__ARGS.computing_agent_ids,
                                                     simulated=False,
                                                     ignore_message_loss=self.__ARGS.ignore_message_loss,
                                                     use_high_level_planner=self.__ARGS.use_high_level_planner,
                                                     use_own_targets=not self.__ARGS.dynamic_swarm)

    def send_socket(self, message: message.MixerMessage):
        if self.socket is None:
            return

        self.socket.sendall(message.serialize())

    @property
    def uart_interface(self):
        """ returns the current UART interface"""
        return self.__uart_interface

    @uart_interface.setter
    def uart_interface(self, new_uart_interface):
        """ sets a new UART interface """
        self.__uart_interface = new_uart_interface

    @property
    def computation_agent(self):
        """ returns the computation agent"""
        return self.__computation_agent

    @computation_agent.setter
    def computation_agent(self, new_computation_agent):
        """ sets a new UART interface """
        self.__computation_agent = new_computation_agent

    @property
    def cu_id(self):
        """ returns the uart_interface id """
        return self.__cu_id