import copy

import numpy as np


def format_message(message):
    """ formats the message by deleting trailing newline characters"""

    if len(message) == 0:
        pass
        # raise ValueError("Empty Message")


    # Ignore newline character
    """
    if chr(message[-1]) == '\n':
        message = message[:-1]
    """

    return message


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

def dequantizer_alexander(x, max_val):
    den = ((1 << 16) - 1)
    return max_val * 2.0 / den * x - max_val


class Message:

    def __init__(self):
        pass

    def parse_data(self, message):
        """ parses the transmitted message

        Parameters:
              message: byte sequence, containing the header (2 bytes) and data"""

        pass

    def serialize(self):
        """ serializes the message, creates byte sequence to transmit via UART"""


class MessageHeader:
    """ MessageHeader class. This class stores the received_data messages and parses the input. """

    def __init__(self, agent_ID=None, message_type=None, message_types=None):
        if message_types is None:
            # list of available message types, must be coherent with the messages of the CP
            message_types = {0: "TYPE_ERROR", 1: "TYPE_METADATA", 2: "TYPE_AP_ACK", 3: "TYPE_CP_ACK",
                             4: "TYPE_ALL_AGENTS_READY",
                             5: "TYPE_LAUNCH_DRONES", 6: "TYPE_TRAJECTORY", 7: "TYPE_DRONE_STATE",
                             8: "TYPE_REQ_TRAJECTORY", 9: "TYPE_SYS_SHUTDOWN"}

        self.__message_types = message_types
        # inverse the dictionary for faster lookups on encoding
        self.__rev_message_types = {v: k for k, v in self.__message_types.items()}
        self.__agent_ID = agent_ID
        self.__message_type = message_type

    def parse_header(self, message):
        """ parses the received_data message. Returns a bool, whether a valid message was received.

        Parameters:
              message: received_data byte sequence"""

        message = format_message(message)

        if len(message) == 0:
            return

        try:
            self.__message_type = self.__message_types[message[0]]
            self.agent_ID = message[1]
            return True
        except Exception:
            print("Message Type Unknown")
            print(message)
            return False

    def serialize(self, data_range=None):
        """ serializes the message, creates byte sequence to transmit via UART"""

        data_array = np.array([self.__rev_message_types[self.__message_type], self.__agent_ID], dtype=np.uint8)
        data_array = data_array.tobytes()
        return data_array

    def parse_data(self, message):
        """ parses the transmitted message

        Parameters:
              message: byte sequence, containing the header (2 bytes) and data"""
        pass

    @property
    def agent_ID(self):
        """ returns the messages agent ID"""
        return self.__agent_ID

    @agent_ID.setter
    def agent_ID(self, new_agent_id):
        """ sets a new agent ID

        Parameters:
            new_agent_id: int: ID of new agent"""
        self.__agent_ID = new_agent_id

    @property
    def message_type(self):
        """ returns the type of the current message"""
        return self.__message_type

    @message_type.setter
    def message_type(self, new_message_type):
        """ sets a new message type """
        if type(new_message_type) == str:
            self.__message_type = new_message_type
        elif type(new_message_type) == int or type(new_message_type) == int:
            self.__message_type = self.rev_message_types[new_message_type]

    @property
    def message_types(self):
        """ returns the dictionary of possible message types"""
        return self.__message_types

    @property
    def rev_message_types(self):
        """ returns the inverse dictionary of possible message types"""
        return self.__rev_message_types


class MessageType:

    def __init__(self, names, data_types):
        """
        Parameters
        ----------
        names: variable names of the messages content. (This correspond to variable names in a struct).
        data_types: datatypes of the variable. List of 2d tuples. First entry is the datatype, second the length.
        """
        self.__names = names
        self.__data_types = data_types
        self.__data_types_dict = {}
        self.__size = 0
        self.__elements_size = []
        for i in range(len(data_types)):
            data_type = data_types[i][0]
            nums = data_types[i][1]
            if self.__data_types[i][0] == "uint16_t":
                self.__data_types_dict[self.__names[i]] = np.uint16
            elif self.__data_types[i][0] == "uint8_t":
                self.__data_types_dict[self.__names[i]] = np.uint8
            else:
                print(self.__data_types[i][0])
                assert False

            if data_type == "uint16_t":
                self.__size += 2*nums
                self.__elements_size.append(2*nums)
            elif data_type == "uint8_t":
                self.__size += nums
                self.__elements_size.append(nums)

        self.__content = {}

    @property
    def size(self):
        return self.__size

    @property
    def names(self):
        return self.__names

    def get_data_type(self, name):
        return self.__data_types_dict[name]

    @property
    def data_types(self):
        return self.__data_types

    def get_name_idx(self, name):
        i = 0
        while i < len(self.__names):
            if name == self.__names[i]:
                return i
        assert False

    def get_content(self, name):
        return self.__content[name]

    def set_content(self, content):
        """
        Parameters
        ----------
            content: dictionary containing the names and the content of the message
        """
        for name in content.keys():
            assert name in self.__names
            self.__content[name] = content[name]

    def set_content_bytes(self, b):
        """
        sets the content from a bytes array
        """
        if len(b) != self.__size:
            print(len(b))
            print(self.__size)
            print(self)
            assert len(b) == self.__size
        current_idx = 0
        for i in range(len(self.__names)):
            data_type = None
            if self.__data_types[i][0] == "uint16_t":
                data_type = np.uint16
            if self.__data_types[i][0] == "uint8_t":
                data_type = np.uint8
            self.__content[self.__names[i]] = np.frombuffer(b[current_idx: current_idx+self.__elements_size[i]], dtype=data_type)
            current_idx += self.__elements_size[i]

    def to_bytes(self):
        b = bytearray()
        for i in range(len(self.__names)):
            new_content = self.__content[self.__names[i]].tobytes()
            assert len(new_content) == self.__elements_size[i]
            b += new_content
        assert len(b) == self.__size
        return b


class MixerMessage(Message):
    """ Mixer Message, contains a header and data"""

    def __init__(self, agent_ID=None, message_type=None, data=None, header=None, message_types=None):
        if header is None:
            self.__header = MessageHeader(agent_ID=agent_ID, message_type=message_type,
                                          message_types=message_types)
        else:
            self.__header = copy.deepcopy(header)
        self.data = copy.deepcopy(data)

        super().__init__()

    def parse_data(self, message, data_range=None):
        """ parses the transmitted message and stores it in the messages data. The message byte sequence is
        transformed into ints (uint_16 for the trajectories and drone states, uint_8 for all other data). If a data
        range is provided for the dequantization, the data is dequantized.

        Parameters:
              message: byte sequence containing the payload data
              data_range: range of the data that was transmitted"""

        message = format_message(message)
        if self.header.message_type in ["TYPE_TRAJECTORY", "TYPE_DRONE_STATE"]:
            data = np.frombuffer(message, dtype=np.uint16)  # interpret as 16 bit int if data is data or state
        else:
            data = np.frombuffer(message, dtype=np.uint8)

        if data_range is not None:  # dequantize data only if data_range is provided, otherwise treat as int
            data = np.array([dequantize_float(i, data_range) for i in data.tolist()])
        self.data = data

    def serialize(self, data_range=None):
        """ serializes the message, creates the byte sequence to transmit via UART

        Parameters:
            data_range: range of data for serialization"""

        if data_range is None:
            data_range = [0, 1]

        header_data = self.header.serialize()

        if self.data is not None:  # serialize payload only if it exists
            if type(self.data) is list:
                self.data = np.array(self.data)

            if len(self.data) > 1:
                data_array = np.reshape(self.data, (self.data.size,)).tolist()
                data_array = np.array([quantize_float(i, data_range) for i in data_array])
                data_array = data_array.tobytes()
            else:
                data_array = np.array([self.data], dtype=np.uint8)
                data_array = data_array.tobytes()

            return header_data + data_array
        else:
            return header_data

    @property
    def data(self):
        """ return previously received_data data"""
        return self.__data

    @data.setter
    def data(self, new_data):
        """ Update the messages data. If the

        Parameters:
            new_data: any"""

        if type(new_data) == int:
            new_data = np.array([new_data]).astype(np.uint8)
        self.__data = new_data

    @property
    def header(self):
        """ return the messages header"""
        return self.__header

    @header.setter
    def header(self, new_header):
        """ set a new header """
        self.__header = new_header
