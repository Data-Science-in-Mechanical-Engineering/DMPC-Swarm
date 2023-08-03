import threading
import time
import sys
import os

class UserInputThread(threading.Thread):
    def __init__(self, fileno):
        super().__init__()
        self.fileno = fileno
        sys.stdin = os.fdopen(fileno)
        self.__state = 0
        self.__user_input = -1
        self.__running_event = threading.Event()
        self.__running_event.clear()

    def start_user_input(self, state):
        print("Please input:")
        self.state = state
        self.user_input = 0
        self.__running_event.set()

    def user_input_running(self):
        return self.__running_event.is_set()

    def stop_user_input(self):
        self.__running_event.clear()

    def run(self):
        while True:
            if self.user_input_running():
                try:
                    value = input("")
                    if value == "":
                        self.user_input = 1

                    else:
                        self.user_input = 2

                    self.stop_user_input()

                except EOFError:
                    print("EOF ERROR")
            else:
                time.sleep(0.1)

    @property
    def state(self):
        """ returns the state of the input thread """
        return self.__state

    @state.setter
    def state(self, new_state):
        """ sets the state to a new value

        Parameters:
            new_state: int
                new value of state """

        self.__state = new_state

    @property
    def user_input(self):
        """ returns the last user input """
        return self.__user_input

    @user_input.setter
    def user_input(self, new_user_input):
        """ returns the last user input """
        self.__user_input = new_user_input

