import pickle as p
from joblib import Parallel, delayed
import compute_unit.real_compute_unit as computing_unit
import define_ARGS
import sys
import threading


def parallel_simulation_wrapper(computation_unit, fileno):
    computation_unit.run(fileno)


class SystemStartupThread(threading.Thread):

    def __init__(self, threadID, uart_interface):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.uart_interface = uart_interface

    def run(self):
        print("Starting Interface " + str(self.threadID))
        self.uart_interface.start_system()

    @property
    def serial_port(self):
        return self.uart_interface.serial_port


if __name__ == "__main__":
    define_ARGS.define_ARGS()
    # LOAD SCENARIO
    args_path = "ARGS_for_testbed.pkl"
    ARGS = p.load(open(args_path, "rb"))
    if type(ARGS) is list:
        ARGS = ARGS[0]

    cu_ids = [20] #ARGS.computing_agent_ids  # IDs of computing units to be executed on this machine

    cus = [computing_unit.ComputingUnit(ARGS, num_static_drones=ARGS.num_static_drones,
                                        cu_id=i, is_initiator=True, sync_movement=False) for i in cu_ids]
    fn = sys.stdin.fileno()
    cus[0].run(fn)

    # RUN CUs
    #Parallel(n_jobs=len(cu_ids))(delayed(parallel_simulation_wrapper)(cus[i], fn) for i in range(len(cu_ids)))
