import uart_interface
import threading
import msvcrt
from user_input_thread import UserInputThread
from time import sleep

class CommunicationInterfaceThread(threading.Thread):

    def __init__(self, threadID, uart_interface):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.uart_interface = uart_interface

    def run(self):
        print("Starting Interface " + str(self.threadID))
        self.uart_interface.run()


if __name__ == "__main__":

    baudrate = 921600  # 460800 #921600
    inter1 = uart_interface.UartInterface(id=20, baudrate=baudrate, trajectory_length=15)
    inter2 = uart_interface.UartInterface(id=21, baudrate=baudrate, trajectory_length=15)

    thread_1 = CommunicationInterfaceThread(1, inter1)
    thread_2 = CommunicationInterfaceThread(2, inter2)

    thread_1.start()
    thread_2.start()
