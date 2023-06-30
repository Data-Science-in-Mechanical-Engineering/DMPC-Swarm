class EasyLogger:
    def __init__(self):
        self.__data = {}

    def add_data_point(self, key, data_point):
        if key in self.__data.keys():
            self.__data[key].append(data_point)
        else:
            self.__data[key] = [data_point]

    def get_data(self):
        return self.__data

class Logger:
    def __init__(self, name, num_agents):
        """
        Parameters
        ----------
            name: string
                name of file to save in
            num_agents: int
                number of agents
        """
        self.__name = name

        # generate files and delete every content
        for i in range(0, num_agents):
            with open("log/" + name + str(i) + ".txt", 'w') as f:
                f.write("")

    def log_data(self, ID, position, control_position, target):
        """log data

        Parameters
        ----------
            ID: int
                id of agent to log data for
            position: np.array, shape(3,)
                position to log
        """
        with open("log/" + self.__name + str(ID) + ".txt", 'a') as f:
            f.writelines(str(position[0]) + " " + str(position[1]) + " " + str(position[2]) + " " +
                         str(position[3]) + " " + str(position[4]) + " " + str(position[5]) + " "
                         + str(control_position[0]) + " "
                         + str(control_position[1]) + " "
                         + str(control_position[2]) + " "
                         + str(target[0]) + " "
                         + str(target[1]) + " "
                         + str(target[2]) + "\n")
