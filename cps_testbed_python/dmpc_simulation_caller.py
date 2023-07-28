"""
Manages the calculation of multiple simulations. The parameters for the simulation are initialilzed. These parameters
are used to initialize the simulations by using the Simulation class.
"""
import copy
import multiprocessing
import sys
from joblib import Parallel, delayed

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


def call_batch_simulation(ARGS_array, name_files="test",
                          message_loss_probability=0.0, ignore_message_loss=False, num_cus=None,
						  simulate_quantization=False):
	for ARGS in ARGS_array:
		ARGS.path = os.path.dirname(os.path.abspath(__file__)) + "/../../batch_simulation_results/dmpc/" \
															+ name_files
		create_dir(ARGS.path)
		ARGS.message_loss_probability = message_loss_probability
		ARGS.ignore_message_loss = ignore_message_loss
		if num_cus is not None:
			ARGS.num_computing_agents = num_cus
			ARGS.computing_agent_ids = [i for i in range(40, 40+num_cus)]
		ARGS.simulate_quantization = simulate_quantization

	with open(ARGS_array[0].path + "/ARGS.pkl", 'wb') as out_file:
		pickle.dump(ARGS_array, out_file)

	max_threads = multiprocessing.cpu_count() - 2
	p = mp.Pool(processes=np.min((max_threads, ARGS_array[0].total_simulations)), maxtasksperchild=1)  #
	simulation_logger = [x for x in p.imap(parallel_simulation_wrapper, ARGS_array)]
	p.close()
	p.terminate()
	p.join()


if __name__ == "__main__":
	# os.environ["OMP_NUM_THREADS"] = "1"
	# os.environ["MKL_NUM_THREADS"] = "1"
	#### Define and parse (optional) arguments for the script ##
	# !!!!!!!!!!!!!!!! Downwash simulation is unrealistic atm. I will contact the authors of the paper and discuss it
	parser = argparse.ArgumentParser(
		description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
	parser.add_argument('--drone', default="cf2x", type=DroneModel, help='Drone model (default: CF2X)', metavar='',
						choices=DroneModel)
	parser.add_argument('--drones', default={i: "Vicon" for i in range(1, 11)}, type=dict,
						help='drone IDs with name of the testbed', metavar='')
	parser.add_argument('--computing_agent_ids', default=[i for i in range(40, 42)], type=list,
						help='List of Computing Agent IDs')
	parser.add_argument('--testbeds', default={"Vicon": ([-2.0, -2.0, 0.3], [2.0, 2.0, 3.0], [0, 0, 0])},
						type=dict, help='Testbeds of the system. Format: name: (min, max, offset)')
	parser.add_argument('--physics', default="pyb_drag", type=Physics, help='Physics updates (default: PYB)',
						metavar='', choices=Physics)
	parser.add_argument('--vision', default=False, type=str2bool, help='Whether to use VisionAviary (default: False)',
						metavar='')
	parser.add_argument('--gui', default=False, type=str2bool, help='Whether to use PyBullet GUI (default: True)',
						metavar='')
	parser.add_argument('--record_video', default=False, type=str2bool,
						help='Whether to record a video (default: False)', metavar='')

	parser.add_argument('--plot', default=False, type=str2bool,
						help='Whether to plot the simulation results (default: True)', metavar='')
	parser.add_argument('--plot_batch_results', default=True, type=str2bool,
						help='Whether to plot the batch simulation results (default: True)', metavar='')
	parser.add_argument('--user_debug_gui', default=False, type=str2bool,
						help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')

	parser.add_argument('--multiprocessing', default=True, type=str2bool,
						help='Whether simulations run in parallel', metavar='')
	parser.add_argument('--log', default=True, type=str2bool,
						help='Whether to log the simulations', metavar='')
	parser.add_argument('--log_state', default=False, type=str2bool,
						help='Whether to log the state', metavar='')
	parser.add_argument('--log_path', default='', type=str,
						help='Path for simulation logs', metavar='')
	parser.add_argument('--aggregate', default=True, type=str2bool,
						help='Whether to aggregate physics steps (default: False)', metavar='')
	parser.add_argument('--obstacles', default=False, type=str2bool,
						help='Whether to add obstacles to the environment (default: True)', metavar='')
	#parser.add_argument('--simulation_freq_hz', default=240, type=int, help='Simulation frequency in Hz (default: 240)',
	#					metavar='')
	parser.add_argument('--sim_steps_per_control', default=4, type=int, help='')
	parser.add_argument('--control_steps_per_round', default=12, type=int, help='')
	parser.add_argument('--use_constant_freq', default=False, type=bool, help='')
	#parser.add_argument('--control_freq_hz', default=60, type=int, help='Control frequency in Hz (default: 48)',
	#					metavar='')
	parser.add_argument('--duration_sec', default=200, type=int,
						help='Duration of the simulation in seconds (default: 5)', metavar='')
	#parser.add_argument('--communication_freq_hz', default=5, type=int,
	#					help='Communication frequency in Hz (default: 10)')
	parser.add_argument('--potential_function', default='chen', type=str,
						help='Potential Field Function for Anti Collision')
	parser.add_argument('--drone_position_initialization_method', default='random_stratify_max_dist', type=str,
						help='Method to initialize drone and target position')

	parser.add_argument('--testbed_size', default=[3.7, 3.7, 3.55], type=list,
						help='Size of the area of movement for the drones')

	parser.add_argument('--abort_simulation', default=True, type=bool, help='Total number of simulations')

	parser.add_argument('--total_simulations', default=100, type=int, help='Total number of simulations')
	parser.add_argument('--network_message_loss', default=[0], type=list,
						help='List of message loss values of the communication network')
	parser.add_argument('--prediction_horizon', default=15, type=int, help='Prediction Horizon for DMPC')
	parser.add_argument('--interpolation_order', default=5, type=int, help='Order of the Bernstein Interpolation')

	parser.add_argument('--r_min', default=0.5, type=float, help='minimum distance to each Drone')
	parser.add_argument('--r_min_crit', default=0.2, type=float, help='minimum distance to each Drone')

	parser.add_argument('--use_soft_constraints', default=False, type=bool, help='')
	parser.add_argument('--guarantee_anti_collision', default=True, type=bool, help='')
	parser.add_argument('--soft_constraint_max', default=0.2, type=float, help='')
	parser.add_argument('--weight_soft_constraint', default=0.01, type=float, help='')

	parser.add_argument('--sim_id', default=0, type=int, help='ID of simulation, used for random generator seed')

	parser.add_argument('--INIT_XYZS', default=[], type=list, help='Initial drone positions')
	parser.add_argument('--INIT_XYZS_id', default={}, type=list, help='Initial drone positions')
	parser.add_argument('--INIT_TARGETS', default={}, type=list, help='Initial target positions')

	parser.add_argument('--skewed_plane_BVC', default=False, type=bool,
						help='Select, whether the BVC planes should be skewed')
	parser.add_argument('--event_trigger', default=True, type=bool,
						help='Select, whether the event trigger should be used for scheduling')
	parser.add_argument('--downwash_scaling_factor', default=4, type=int,
						help='Scaling factor to account for the downwash')
	parser.add_argument('--downwash_scaling_factor_crit', default=4, type=int,
						help='Scaling factor to account for the downwash')
	parser.add_argument('--use_qpsolvers', default=True, type=bool,
						help='Select, whether qpsolver is used for trajectory planning')
	parser.add_argument('--alpha_1', default=10.0, type=bool,
						help='Weight in event-trigger')
	parser.add_argument('--alpha_2', default=10.0*0, type=bool,
						help='Weight in event-trigger')
	parser.add_argument('--alpha_3', default=0.0, type=bool,
						help='Weight in event-trigger')
	parser.add_argument('--alpha_4', default=10.0*0, type=bool,

						help='Weight in event-trigger')
	parser.add_argument('--save_video', default=False, type=bool,
						help='Select, whether a video should be saved')
	parser.add_argument('--remove_redundant_constraints', default=False, type=bool,
						help='Select, whether a video should be saved')
	parser.add_argument('--min_distance_cooperative', default=0.1, type=float,
						help='Select, whether a video should be saved')
	parser.add_argument('--weight_cooperative', default=0.0, type=float,
						help='Select, whether a video should be saved')
	parser.add_argument('--weight_state', default=1.0, type=float,
						help='Select, whether a video should be saved')
	parser.add_argument('--weight_speed', default=0.0001, type=float,
						help='Select, whether a video should be saved')
	parser.add_argument('--weight_acc', default=0.005, type=float,
						help='Select, whether a video should be saved')
	parser.add_argument('--weight_jerk', default=0.001, type=float,
						help='Select, whether a video should be saved')
	parser.add_argument('--rounds_lost', default=[0+i for i in range(500000)] + [50000+i for i in range(10)], type=any,
						help='Rounds in which message loss occur')
	parser.add_argument('--message_loss_probability', default=0.001,
						type=any,
						help='Message loss probability')
	parser.add_argument('--ignore_message_loss', default=False,
						type=any,
						help='if algorithm should ignore message loss')
	parser.add_argument('--use_real_testbed_dim', default=False,
						type=bool,
						help='Message loss probability')
	#parser.add_argument('--target_noise', default=0.009538805943392495 * 0, type=float,
    #						help='Select, whether a video should be saved')
	parser.add_argument('--cooperative_normal_vector_noise', default=0.17137653933980446 * 0, type=float)

	parser.add_argument('--use_high_level_planner', default=True, type=bool)

	parser.add_argument('--hyperparameter_optimization', default=False, type=bool,
						help='optmimize hyperparameters')

	parser.add_argument('--simulated', default=True, type=bool,
						help='optmimize hyperparameters')

	parser.add_argument('--use_optimized_constraints', default=True, type=bool,
						help='optmimize hyperparameters')

	parser.add_argument('--use_demo_setpoints', default=False, type=bool,
						help='if the drones should fly in a demo formation')

	parser.add_argument('--use_own_targets', default=False, type=bool,
						help='if the cus should use their own targets')

	parser.add_argument('--agent_dodge_distance', default=0.5, type=float)

	parser.add_argument("--weight_band", default=0.5, type=float, help="")
	parser.add_argument("--width_band", default=0.3, type=float, help="")

	parser.add_argument("--load_cus", default=False, type=float, help="")
	parser.add_argument("--load_cus_round_nmbr", default=150, type=int, help="")

	parser.add_argument("--save_snapshot_times", default=[], type=any, help="")

	parser.add_argument("--simulate_quantization", default=True, type=bool, help="")

	ARGS = parser.parse_args()

	ARGS.drone_ids = list(ARGS.drones.keys())
	ARGS.num_drones = [len(ARGS.drones)]
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
	ARGS.setpoint_creator = None

	if ARGS.hyperparameter_optimization:
		ARGS.save_video = False

	origin = np.array(ARGS.testbeds["Vicon"][0]) + np.array(ARGS.testbeds["Vicon"][2])
	testbed_size = np.array(ARGS.testbeds["Vicon"][1]) - np.array(ARGS.testbeds["Vicon"][0])
	testbed = cuboid.Cuboid(origin, np.array([testbed_size[0], 0, 0]),
							np.array([0, testbed_size[1], 0]),
							np.array([0, 0, testbed_size[2]]))

	ARGS.testbed = testbed

	### Initialize the plotter #################################
	plotter = Plotter()

	# Initialize the simulations
	simulations = []

	# prepare the ARGS for simulations
	ARGS_array = []

	initializer = initializer.Initializer(testbed, rng_seed=10)

	for i in range(0, ARGS.total_simulations):
		ARGS_for_simulation = copy.deepcopy(ARGS)
		num_sims_per_drone_config = int(ARGS.total_simulations / len(ARGS.num_drones))
		ARGS_for_simulation.num_drones = ARGS.num_drones[i % len(ARGS.num_drones)]
		INIT_XYZS, INIT_TARGETS = initializer.initialize(
			ARGS_for_simulation.drone_position_initialization_method, dist_to_wall=0.25,
			num_points=ARGS_for_simulation.num_drones,
			min_dist=ARGS.r_min, scaling_factor=ARGS.downwash_scaling_factor)
		INIT_XYZS = np.array(INIT_XYZS)
		INIT_TARGETS = np.array(INIT_TARGETS)

		ARGS_for_simulation.num_targets_per_drone = len(INIT_TARGETS) // ARGS_for_simulation.num_drones

		ARGS_for_simulation.INIT_TARGETS = {ARGS.drone_ids[j]: INIT_TARGETS[j] for j in range(ARGS_for_simulation.num_drones)}

		for j in range(ARGS_for_simulation.num_drones, ARGS_for_simulation.num_drones * ARGS_for_simulation.num_targets_per_drone):
			id = ARGS_for_simulation.drone_ids[j % ARGS_for_simulation.num_drones]
			ARGS_for_simulation.INIT_TARGETS[id] = np.vstack((ARGS_for_simulation.INIT_TARGETS[id], INIT_TARGETS[j]))

		for j in range(ARGS_for_simulation.num_drones):
			id = ARGS_for_simulation.drone_ids[j]
			ARGS_for_simulation.INIT_XYZS_id[id] = INIT_XYZS[j]
			ARGS_for_simulation.INIT_TARGETS[id] = np.vstack(
				(ARGS_for_simulation.INIT_TARGETS[id],))

		ARGS_for_simulation.INIT_XYZS = INIT_XYZS
		print("Prepared simulation number " + str(i) + " with " + str(ARGS_for_simulation.num_drones) + " Drones.")
		ARGS_for_simulation.sim_id = i + 1
		ARGS_array.append(ARGS_for_simulation)

	#call_batch_simulation(ARGS_array, name_files="dmpc_simulation_results_not_ignore_message_loss_001",
	#					  message_loss_probability=0.01, ignore_message_loss=False)
	#call_batch_simulation(ARGS_array, name_files="dmpc_simulation_results_ignore_message_loss_001",
	#					  message_loss_probability=0.01, ignore_message_loss=True)

	#call_batch_simulation(ARGS_array, name_files="dmpc_simulation_results_not_ignore_message_loss_005",
	#					  message_loss_probability=0.05, ignore_message_loss=False)
	#call_batch_simulation(ARGS_array, name_files="dmpc_simulation_results_ignore_message_loss_005",
	#					  message_loss_probability=0.05, ignore_message_loss=True)

	for num_cus in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
		for message_loss_prob in [0, 0.01, 0.1]:
			for simulate_quantization in [True]:
				call_batch_simulation(ARGS_array, name_files=f"dmpc_simulation_results_ignore_message_loss_{int(100*message_loss_prob+1e-7)}_{num_cus}cus_{'quant' if simulate_quantization else ''}",
									  message_loss_probability=message_loss_prob, ignore_message_loss=False,
									  num_cus=num_cus, simulate_quantization=simulate_quantization)

	#for num_cus in [1, 3, 5, 7, 9, 11, 13, 15]:
#		call_batch_simulation(ARGS_array, name_files=f"dmpc_simulation_results_not_ignore_message_loss_001_{num_cus}_cus5",#
							#  message_loss_probability=0.05, ignore_message_loss=True, num_cus=num_cus)