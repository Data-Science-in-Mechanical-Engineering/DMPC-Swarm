import time
import numpy as np
import sys
import glob
import serial


class UartInterface():
    """ class that handles the communication via UART with the Communication Processor """

    def __init__(self, baudrate=921600, id=1):
        self.__baudrate = baudrate

        self.__start = time.time()
        done = False

        self.__send_to_cp = None
        self.__cp_id = id
        self.__timeout = 0.05
        self.__round = 0

        self.__serial_port = None

        self.__message_types = {0: "TYPE_ERROR", 1: "TYPE_METADATA", 2: "TYPE_AP_ACK", 3: "TYPE_CP_ACK",
                                 4: "TYPE_ALL_AGENTS_READY",
                                 5: "TYPE_LAUNCH_DRONES", 6: "TYPE_TRAJECTORY", 7: "TYPE_DRONE_STATE",
                                 8: "TYPE_REQ_TRAJECTORY", 9: "TYPE_SYS_SHUTDOWN"}

    def read_from_uart(self, serial_port=None):
        """ reads messages from uart and return it as an array."""

        if serial_port is None:
            serial_port = self.serial_port

        data_size = int.from_bytes(serial_port.read(size=2), 'little')  # receive size of data
        return serial_port.read(size=data_size)
        """
        idx = 0
        while idx < len(data):
            message_type = self.__message_types[int.from_bytes(data[idx:idx+2], 'big')]
            if message_type == "TYPE_TRAJECTORY":"""

    def read_from_uart_raw(self, size):
        """reads size bytes from UART"""
        return self.serial_port.read(size=size)

    def send_to_uart(self, uart_message):
        """ sends a message via UART"""
        start = time.time()
        self.serial_port.write(np.ushort(len(uart_message)).tobytes())
        #print(f"1: {time.time() - start}")
        #time.sleep(1e-6)  # in windows this waits way to long
        t_wait = 1e-3
        start = time.time()
        while time.time() - start < t_wait:
            pass
        #print(f"1: {time.time() - start}")
        #start = time.time()
        self.serial_port.write(uart_message)
        #print(time.time()-start)



    def initialize(self):
        """ initializes the UART communication to the CP"""

        port = None
        while port is None:
            port = self.scan_serial_ports(baudrate=self.__baudrate, cp_id=self.__cp_id)

        self.serial_port = serial.Serial(port=port, baudrate=self.__baudrate)

    def print(self, msg):

        msg = str(msg)  # make sure, the message is of type string for printing

        print("[" + str(self.__cp_id) + "]: " + msg)

    def scan_serial_ports(self, baudrate, cp_id):
        """ Returns the first port at which a CP with the correct ID answers at

                :parameters:
                    baudrate: int
                        baudrate of the serial interface

                    self.__cp_id: int
                        ID of Communication Processor, this instance of AP should work with

                    header: message.Header


                :raises EnvironmentError:
                    On unsupported or unknown platforms
                :returns:
                    A list of the serial ports available on the system
            """
        self.print("Scanning for CP")

        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')

        """ TRY EVERY AVAILABLE PORT """
        for port in ports:
            try:
                s = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
                init_message = self.read_from_uart(s)
                print(f"{port}: {len(init_message)}")
                if init_message is None:
                    continue
                elif len(init_message) == 0:
                    continue
                if int.from_bytes([init_message[0]], 'big') == 1 and int.from_bytes([init_message[1]], 'big') == cp_id:
                    self.print("Connected on Port " + port)
                    s.close()
                    return port
            except (OSError, serial.SerialException, ValueError):
                pass

    @property
    def serial_port(self):
        """ returns the currently used serial port """

        return self.__serial_port

    @serial_port.setter
    def serial_port(self, new_serial_port):
        """ sets and opens the serial port """
        if not new_serial_port.is_open:
            new_serial_port.open()

        self.__serial_port = new_serial_port