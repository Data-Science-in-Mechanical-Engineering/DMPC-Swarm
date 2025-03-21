"""
Manages the calculation of multiple simulations. The parameters for the simulation are initialilzed. These parameters
are used to initialize the simulations by using the Simulation class.
"""
import copy
import multiprocessing
import sys
import yaml
import cv2

from compute_unit import setpoint_creator

sys.path.append("../../../gym-pybullet-drones")
sys.path.append("../../../")
sys.path.append("gym-pybullet-drones/")

import argparse
import numpy as np
import pickle
import os
import multiprocessing as mp
import gc

from simulation.gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from simulation.gym_pybullet_drones.utils.utils import sync, str2bool

import simulation.simulator as simulation
import useful_scripts.initializer as initializer
from useful_scripts.plotter import Plotter
import useful_scripts.cuboid as cuboid

from pathlib import Path


def parse_params(ARGS, params):
    for key in params:
        setattr(ARGS, key, params[key])

    return ARGS


def save(simulation_logger, ARGS, path, simulation_number):
    with open(
            path + "simulation_result-" + str(ARGS.num_drones) + "_drones" + str(simulation_number) + ".pkl", 'wb') \
            as out_file:
        pickle.dump([simulation_logger, ARGS], out_file)


def create_dir(path_to_logger):
    try:
        os.makedirs(path_to_logger)
    finally:
        return


def parallel_simulation_wrapper(ARGS_for_simulation):
    sim = simulation.Simulation(ARGS_for_simulation)
    sim.run()
    del sim
    gc.collect()
    return None


def call_batch_simulation_hpc(ARGS_array):
    for ARGS in ARGS_array:
        ARGS.path = f"{ARGS.hpc_saving_path}/{ARGS.name}/" \
                    + f"dmpc_simulation_results_iml{ARGS.ignore_message_loss}_{int(100 * ARGS.message_loss_probability + 1e-7)}_{ARGS.num_computing_agents}cus_{'quant' if ARGS.simulate_quantization else ''}_{ARGS.trigger}"

        create_dir(ARGS.path)

    #with open(ARGS_array[0].path + f"/ARGS_{ARGS_array[0].num_drones}_drones.pkl", 'wb') as out_file:
        #pickle.dump(ARGS_array, out_file)
    print(f"Running {multiprocessing.cpu_count()} simulations in parallel.")
    max_threads = multiprocessing.cpu_count()
    p = mp.Pool(processes=np.min((max_threads, ARGS_array[0].num_simulations)), maxtasksperchild=1)  #
    simulation_logger = [x for x in p.imap(parallel_simulation_wrapper, ARGS_array)]
    p.close()
    p.terminate()
    p.join()


def call_single_simulation(ARGS):
    ARGS.path = os.path.dirname(os.path.abspath(__file__)) + f"/../../simulation_results/dmpc/{ARGS.name}"
    create_dir(ARGS.path)
    ARGS.sim_id = 1
    parallel_simulation_wrapper(ARGS)

    if ARGS.save_video:
        print("Saving video")
        video_name = 'Video.mp4'
        image_folder = ARGS.path + "_simnr_1"
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") and img.startswith("ImgPredict1")]
        frame = cv2.imread(os.path.join(image_folder, "ImgPredict1_0.jpg"))
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(os.path.join(image_folder, video_name), fourcc, fps=30, frameSize=(width, height))

        for i in range(1, len(images)):
            video.write(cv2.imread(os.path.join(image_folder, "ImgPredict1_" + str(i) + ".jpg")))

        video.release()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    # !!!!!!!!!!!!!!!! Downwash simulation is unrealistic atm.
    parser = argparse.ArgumentParser(
        description='CPS Testbed Simulator')

    #parser.add_argument('--param_path', default="parameters/hyperparameter_opt.yaml", type=str,
    #                    help='yaml file for parameters', metavar='')
    
    parser.add_argument('--param_path', default="parameters/batch_simulation_local_pc.yaml", type=str,
                        help='yaml file for parameters', metavar='')

    # the following are only needed for the hpc
    parser.add_argument('-i', "--iter_id", default=None, type=int, help='id of slurm job', metavar='')
    parser.add_argument('-n', "--name", default=None, type=str, help='name of params for slurm job', metavar='')

    # a lot of other parameters that need to be kept constant.
    parser.add_argument('--drone', default="cf2x", type=DroneModel, help='Drone model (default: CF2X)', metavar='',
                        choices=DroneModel)
    parser.add_argument('--physics', default="pyb_drag", type=Physics, help='Physics updates (default: PYB)',
                        metavar='', choices=Physics)
    parser.add_argument('--vision', default=False, type=str2bool, help='Whether to use VisionAviary (default: False)',
                        metavar='')
    parser.add_argument('--gui', default=False, type=str2bool, help='Whether to use PyBullet GUI (default: True)',
                        metavar='')
    parser.add_argument('--record_video', default=False, type=str2bool,
                        help='Whether to record a video (default: False) deprecated, dont use this', metavar='')

    parser.add_argument('--plot', default=False, type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=False, type=str2bool,
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')

    parser.add_argument('--aggregate', default=True, type=str2bool,
                        help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles', default=False, type=str2bool,
                        help='Whether to add obstacles to the environment (default: True)', metavar='')

    parser.add_argument('--drone_position_initialization_method', default='random_stratify_max_dist', type=str,
                        help='Method to initialize drone and target position')

    parser.add_argument('--interpolation_order', default=5, type=int,
                        help='Order of the Bernstein Interpolation (deprecated)')

    parser.add_argument('--rounds_lost', default=[0 + i for i in range(500000)] + [50000 + i for i in range(10)],
                        type=any,
                        help='Rounds in which message loss occur')

    parser.add_argument("--load_cus", default=False, type=float, help="")
    parser.add_argument("--load_cus_round_nmbr", default=150, type=int, help="")

    parser.add_argument("--save_snapshot_times", default=[], type=any, help="")

    parser.add_argument("--show_animation", default=False, type=bool,
                        help="This is only needed when we want to live plot what is happening (during operation of the testbed)")

    ARGS = parser.parse_args()

    # if this is set, we are on the cluster. So load the parameter sweeps
    if ARGS.iter_id is not None:
        ARGS.param_path = f"/work/mf724021/hpc_parameters/{ARGS.name}/params{ARGS.iter_id}.yaml"

    with open(ARGS.param_path, 'r') as file:
        params = yaml.safe_load(file)

    ARGS = parse_params(ARGS, params)

    ARGS.path = os.path.dirname(os.path.abspath(__file__)) + "/../../simulation_results/dmpc/demo/"

    # Trigger
    ARGS.alpha_1 = 0.0
    ARGS.alpha_2 = 0.0
    ARGS.alpha_3 = 0.0
    ARGS.alpha_4 = 0

    if ARGS.trigger == "HT":
        ARGS.alpha_1 = 1.0
    elif ARGS.trigger == "RR":
        ARGS.alpha_2 = 1.0
    elif ARGS.trigger == "DT":
        ARGS.alpha_3 = 1.0

    # we only simulate the vicon testbed
    ARGS.drones = {i: "Vicon" for i in range(1, ARGS.num_drones + 1)}
    ARGS.drone_ids = list(ARGS.drones.keys())

    # if the num_cus is set, use this instead of the ifs of the c
    if hasattr(ARGS, "num_computing_agents"):
        ARGS.computing_agent_ids = [40 + i for i in range(ARGS.num_computing_agents)]

    ARGS.num_computing_agents = len(ARGS.computing_agent_ids)

    ARGS.max_positions = {}
    ARGS.min_positions = {}
    ARGS.pos_offset = {}

    print("Initializing drones:")
    for key in ARGS.drones:
        testbed = ARGS.drones[key]
        offset = np.array(ARGS.testbeds[testbed][2])
        ARGS.pos_offset[key] = offset
        ARGS.min_positions[key] = np.array(ARGS.testbeds[testbed][0]) + offset
        ARGS.max_positions[key] = np.array(ARGS.testbeds[testbed][1]) + offset
        print(
            f"Drone {key} in {testbed} with offset {offset}, min_pos: {ARGS.min_positions[key]} and max_pos: {ARGS.max_positions[key]}")

    # only used if use_own_targets is set.
    ARGS.setpoint_creator = setpoint_creator.SetpointCreator(ARGS.drones, ARGS.testbeds,
                                                             demo_setpoints=setpoint_creator.DEMO_LLM)

    origin = np.array(ARGS.testbeds["Vicon"][0]) + np.array(ARGS.testbeds["Vicon"][2])
    testbed_size = np.array(ARGS.testbeds["Vicon"][1]) - np.array(ARGS.testbeds["Vicon"][0])
    testbed = cuboid.Cuboid(origin, np.array([testbed_size[0], 0, 0]),
                            np.array([0, testbed_size[1], 0]),
                            np.array([0, 0, testbed_size[2]]))

    ARGS.testbed = testbed

    # prepare the ARGS for simulations
    ARGS_array = []

    initializer = initializer.Initializer(testbed, rng_seed=10)

    for i in range(0, ARGS.num_simulations):
        ARGS_for_simulation = copy.deepcopy(ARGS)
        # num_sims_per_drone_config = int(ARGS.num_simulations / len(ARGS.num_drones))
        # ARGS_for_simulation.num_drones = ARGS.num_drones[i % len(ARGS.num_drones)]
        INIT_XYZS, INIT_TARGETS = initializer.initialize(
            ARGS_for_simulation.drone_position_initialization_method, dist_to_wall=0.25,
            num_points=ARGS_for_simulation.num_drones,
            min_dist=ARGS.r_min, scaling_factor=ARGS.downwash_scaling_factor)
        INIT_XYZS = np.array(INIT_XYZS)
        INIT_TARGETS = np.array(INIT_TARGETS)

        # for a normal simulation with randomly selected target positions this is not necessary as num_targets_per drone
        # is always one. However, for demo setpoints this is necessary.
        # TODO think about removing this, as we use the setpoint creator for demo setpoints.
        ARGS_for_simulation.num_targets_per_drone = len(INIT_TARGETS) // ARGS_for_simulation.num_drones

        ARGS_for_simulation.INIT_TARGETS = {ARGS.drone_ids[j]: INIT_TARGETS[j] for j in
                                            range(ARGS_for_simulation.num_drones)}

        for j in range(ARGS_for_simulation.num_drones,
                       ARGS_for_simulation.num_drones * ARGS_for_simulation.num_targets_per_drone):
            id = ARGS_for_simulation.drone_ids[j % ARGS_for_simulation.num_drones]
            ARGS_for_simulation.INIT_TARGETS[id] = np.vstack((ARGS_for_simulation.INIT_TARGETS[id], INIT_TARGETS[j]))

        ARGS_for_simulation.INIT_XYZS_id = {}
        for j in range(ARGS_for_simulation.num_drones):
            id = ARGS_for_simulation.drone_ids[j]
            ARGS_for_simulation.INIT_XYZS_id[id] = INIT_XYZS[j]
            ARGS_for_simulation.INIT_TARGETS[id] = np.vstack(
                (ARGS_for_simulation.INIT_TARGETS[id],))

        ARGS_for_simulation.INIT_XYZS = INIT_XYZS
        print("Prepared simulation number " + str(i) + " with " + str(ARGS_for_simulation.num_drones) + " Drones.")
        ARGS_for_simulation.sim_id = i + 1
        ARGS_array.append(ARGS_for_simulation)

    if ARGS.run_on_cluster:
        call_batch_simulation_hpc(ARGS_array)
    else:
        call_single_simulation(ARGS_array[0])
