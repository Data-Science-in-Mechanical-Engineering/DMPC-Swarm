"""
Manages the calculation of multiple simulations. The parameters for the simulation are initialilzed. These parameters
are used to initialize the simulations by using the Simulation class.
"""
import copy
import multiprocessing

import cv2
import time
import argparse
import math
import numpy as np
import pickle
import os
from datetime import datetime
import multiprocessing as mp
import pickle as p
import gc

import shutil

from simulation.gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from simulation.gym_pybullet_drones.utils.utils import sync, str2bool

import simulation.simulator as simulation
import useful_scripts.initializer as initializer
from useful_scripts.plotter import Plotter
import useful_scripts.cuboid as cuboid

import compute_unit.setpoint_creator as setpoint_creator

import contextlib


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
	sim = simulation.Simulation(ARGS_for_simulation, ARGS_for_simulation.cus)
	sim.run()
	del sim
	gc.collect()
	return None


def bo_simulation_wrapper(ARGS_for_simulation):
	sim = simulation.Simulation(ARGS_for_simulation)
	logger = sim.run()
	del sim

	return logger.transition_times.tolist()


def evaluate_hyperparameters(hyperparameters, ARGS_array):
	data = []

	# supress console outputs
	with contextlib.redirect_stdout(None):
		for i in range(len(ARGS_array)):
			for key in hyperparameters.keys():
				ARGS_array[i].__dict__[key] = hyperparameters[key]
		max_threads = multiprocessing.cpu_count() - 2
		p = mp.Pool(processes=np.min((max_threads, ARGS.total_simulations)), maxtasksperchild=1)  #
		data = [x for x in p.imap(bo_simulation_wrapper, ARGS_array)]
		p.close()
		p.terminate()
		p.join()

	# remove nans (TODO think about better way)
	remove_inds = []
	for d in data:
		remove_ind = []
		for i in range(len(d)):
			if math.isnan(d[i]):
				d[i] = ARGS_array[0].duration_sec
	data = [max(d) for d in data]
	data = np.array(data)
	data = data.flatten()
	return data.mean(), data.std()*0


if __name__ == "__main__":
	# os.environ["OMP_NUM_THREADS"] = "1"
	# os.environ["MKL_NUM_THREADS"] = "1"
	#### Define and parse (optional) arguments for the script ##
	# !!!!!!!!!!!!!!!! Downwash simulation is unrealistic atm. I will contact the authors of the paper and discuss it
	parser = argparse.ArgumentParser(
		description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
	parser.add_argument('--drone', default="cf2x", type=DroneModel, help='Drone model (default: CF2X)', metavar='',
						choices=DroneModel)
	parser.add_argument('--drones', default={i: "Vicon" for i in range(1, 7)}, type=dict,
						help='drone IDs with name of the testbed', metavar='')
	parser.add_argument('--computing_agent_ids', default=[i for i in range(20, 22)], type=list, help='List of Computing Agent IDs')
	parser.add_argument('--testbeds', default={"Vicon": ([-1.8, -1.8, 0.3], [1.8, 1.8, 3.0], [0, 0, 0])},
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
	parser.add_argument('--log_state', default=True, type=str2bool,
						help='Whether to log the state', metavar='')
	parser.add_argument('--log_path', default='', type=str,
						help='Path for simulation logs', metavar='')
	parser.add_argument('--aggregate', default=True, type=str2bool,
						help='Whether to aggregate physics steps (default: False)', metavar='')
	parser.add_argument('--obstacles', default=False, type=str2bool,
						help='Whether to add obstacles to the environment (default: True)', metavar='')
	parser.add_argument('--sim_steps_per_control', default=4, type=int, help='')
	parser.add_argument('--control_steps_per_round', default=12, type=int, help='')
	parser.add_argument('--use_constant_freq', default=True, type=bool, help='')
	parser.add_argument('--duration_sec', default=60, type=int,
						help='Duration of the simulation in seconds (default: 5)', metavar='')
	parser.add_argument('--communication_freq_hz', default=5, type=int,
						help='Communication frequency in Hz (default: 10)')
	parser.add_argument('--potential_function', default='chen', type=str,
						help='Potential Field Function for Anti Collision')
	parser.add_argument('--drone_position_initialization_method', default='random_stratify_max_dist', type=str,
						help='Method to initialize drone and target position')

	parser.add_argument('--testbed_size', default=[3.7, 3.7, 3.55], type=list,
						help='Size of the area of movement for the drones')

	parser.add_argument('--abort_simulation', default=True, type=bool, help='Total number of simulations')

	parser.add_argument('--total_simulations', default=1, type=int, help='Total number of simulations')
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
	parser.add_argument('--downwash_scaling_factor', default=3, type=int,
						help='Scaling factor to account for the downwash')
	parser.add_argument('--downwash_scaling_factor_crit', default=4, type=int,
						help='Scaling factor to account for the downwash')
	parser.add_argument('--use_qpsolvers', default=True, type=bool,
						help='Select, whether qpsolver is used for trajectory planning')
	parser.add_argument('--alpha_1', default=1*1, type=bool,
						help='Weight in event-trigger')
	parser.add_argument('--alpha_2', default=1*0, type=bool,
						help='Weight in event-trigger')
	parser.add_argument('--alpha_3', default=1*0, type=bool,
						help='Weight in event-trigger')
	parser.add_argument('--alpha_4', default=1*1, type=bool,

						help='Weight in event-trigger')
	parser.add_argument('--save_video', default=True, type=bool,
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
	parser.add_argument('--weight_acc', default=0.05, type=float,
						help='Select, whether a video should be saved')
	parser.add_argument('--weight_jerk', default=0.01, type=float,
						help='Select, whether a video should be saved')
	parser.add_argument('--rounds_lost', default=[0+i for i in range(100000)] + [50000+i for i in range(10)], type=any,
						help='Rounds in which message loss occur')
	parser.add_argument('--message_loss_probability', default=0.000,
						type=any,
						help='Message loss probability')
	parser.add_argument('--use_real_testbed_dim', default=False,
						type=bool,
						help='Message loss probability')
	parser.add_argument('--ignore_message_loss', default=False,
						type=any,
						help='if algorithm should ignore message loss')
	#parser.add_argument('--target_noise', default=0.009538805943392495 * 0, type=float,
    #						help='Select, whether a video should be saved')
	parser.add_argument('--cooperative_normal_vector_noise', default=0.17137653933980446 * 0, type=float)

	parser.add_argument('--use_high_level_planner', default=True, type=bool)
	parser.add_argument('--agent_dodge_distance', default=0.5, type=float)

	parser.add_argument('--hyperparameter_optimization', default=False, type=bool,
						help='optmimize hyperparameters')

	parser.add_argument('--simulated', default=False, type=bool,
						help='optmimize hyperparameters')

	parser.add_argument('--use_optimized_constraints', default=True, type=bool,
						help='optmimize hyperparameters')

	parser.add_argument('--use_demo_setpoints', default=False, type=bool,
						help='if the drones should fly in a demo formation')

	parser.add_argument('--use_own_targets', default=True, type=bool,
						help='if the cus should use their own targets')

	parser.add_argument("--weight_band", default=0.5, type=float, help="")
	parser.add_argument("--width_band", default=0.3, type=float, help="")

	parser.add_argument("--load_cus", default=False, type=float, help="")
	parser.add_argument("--load_cus_round_nmbr", default=150, type=int, help="")

	parser.add_argument("--save_snapshot_times", default=[], type=any, help="")

	parser.add_argument("--simulate_quantization", default=True, type=bool, help="")

	parser.add_argument("--show_animation", default=True, type=bool, help="")

	ARGS = parser.parse_args()

	ARGS.drone_ids = list(ARGS.drones.keys())
	ARGS.num_drones = [len(ARGS.drones)]
	ARGS.num_computing_agents = len(ARGS.computing_agent_ids)

	ARGS.max_positions = {}
	ARGS.min_positions = {}
	ARGS.pos_offset = {}

	ARGS.cus = []

	print("Initializing drones:")
	for key in ARGS.drones:
		testbed = ARGS.drones[key]
		offset = np.array(ARGS.testbeds[testbed][2])
		ARGS.pos_offset[key] = offset
		ARGS.min_positions[key] = np.array(ARGS.testbeds[testbed][0]) + offset
		ARGS.max_positions[key] = np.array(ARGS.testbeds[testbed][1]) + offset
		print(
			f"Drone {key} in {testbed} with offset {offset}, min_pos: {ARGS.min_positions[key]} and max_pos: {ARGS.max_positions[key]}")

	ARGS.setpoint_creator = setpoint_creator.SetpointCreator(ARGS.drones, ARGS.testbeds, demo_setpoints=setpoint_creator.DEMO)

	if ARGS.hyperparameter_optimization:
		ARGS.save_video = False

	path = os.path.dirname(os.path.abspath(__file__)) + "/../../batch_simulation_results/dmpc/" \
														"dmpc_simulation_results_test_" + datetime.now().strftime(
		"%m_%d_%Y_%H_%M_%S")

	path = os.path.dirname(os.path.abspath(__file__)) + "\\..\\..\\batch_simulation_results\\dmpc\\" \
														"dmpc_simulation_results_not_ignore_message_loss_demo1"

	testbed = cuboid.Cuboid(np.array([0.4, 0.4, 0.3]), np.array([ARGS.testbed_size[0], 0, 0]),
							np.array([0, ARGS.testbed_size[1], 0]),
							np.array([0, 0, ARGS.testbed_size[2]]))

	ARGS.testbed = testbed

	ARGS.path = path
	create_dir(path)

	### Initialize the plotter #################################
	plotter = Plotter()

	# Initialize the simulations
	simulations = []

	# prepare the ARGS for simulations
	ARGS_array = []

	initializer = initializer.Initializer(testbed, rng_seed=100)

	load_data_path = os.path.dirname(os.path.abspath(__file__)) + "/../../batch_simulation_results/dmpc/" \
																  "PaperSimulations/dmpc_simulation_results_test_ET_Soft_constraints/" \
																  "ARGS.pkl"

	# If the loaded ARGS_array is sorted by num_drones, some information needs to be given in order to load them correctly
	num_drones_in_ARGS = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
	loaded_sims_per_drone_config = [0] * len(num_drones_in_ARGS)
	num_sims_per_drone_config_in_ARGS = 1000

	use_inits_const = True

	TESTBED = 0
	CIRCLE = 1
	init_method = CIRCLE
	a = 2

	cus = []
	# if we do not load cus, then use some other init positions.
	if not ARGS.load_cus:
		if init_method == CIRCLE:
			r = 1.5
			INIT_XYZS = [None for i in range(ARGS.num_drones[0])]
			INIT_TARGETS = [None for i in range(ARGS.num_drones[0])]
			for i in range(0, ARGS.num_drones[0]):
				INIT_XYZS[i] = [r * math.cos(2 * 3.141 / ARGS.num_drones[0] * i),
								  r * math.sin(2 * 3.141 / ARGS.num_drones[0] * i), 1.0]
				INIT_TARGETS[i] = [r * math.cos(2 * 3.141/ ARGS.num_drones[0] * i + math.pi * (1.0)),
									r * math.sin(2 * 3.141 / ARGS.num_drones[0] * i + math.pi * (1.0)), 1.0]

			INIT_XYZS = np.array(INIT_XYZS)
			INIT_TARGETS = np.array(INIT_TARGETS)
		elif init_method == TESTBED:
			INIT_XYZS = np.array([[-1, 1, 1], [0, 1, 1], [1, 1, 1],
								  [-1.5, 0, 1], [-0.5, 0, 1], [0.5, 0, 1],
								  #[1.5, 0, 1],
								  #[-1, -1, 1] #, [0, -1, 1], [1, -1, 1]
								  ])
			INIT_TARGETS = INIT_XYZS + 1000
	else:
		print("Loading CUs")
		# use init pos from loaded CUs
		for cu_id in ARGS.computing_agent_ids:
			with open("../../experiment_measurements" +
				f"/CU{cu_id}snapshot{ARGS.load_cus_round_nmbr}.p", 'rb') \
					as in_file:
				cus.append(pickle.load(in_file))

		INIT_XYZS = []
		for drone_id in cus[0].agent_ids:
			INIT_XYZS.append(list(cus[0].get_agent_position(drone_id)))
			# print(cus[0].get_agent_input(drone_id))
		INIT_XYZS = np.array(INIT_XYZS)
		print(INIT_XYZS)

		INIT_TARGETS = INIT_XYZS + 1000
		ARGS.cus = cus


	# load_data_path = None
	load_data_path = os.path.dirname(os.path.abspath(__file__)) + "/../../batch_simulation_results/dmpc/" \
																  "ARGSZ.pkl"

	load_data_path = None
	ARGS_loaded = None
	ARGS_sorted_by_num_drones = False  # select, whether the ARGS_array is sorted
	fly_back = False
	if load_data_path is not None:
		ARGS_loaded = p.load(open(load_data_path, "rb"))

	for i in range(0, ARGS.total_simulations):
		ARGS_for_simulation = copy.deepcopy(ARGS)
		num_sims_per_drone_config = int(ARGS.total_simulations / len(ARGS.num_drones))
		if num_sims_per_drone_config * len(ARGS.num_drones) != ARGS.total_simulations:
			raise ValueError('Total Simulations is not an integer multiple of the total number of drone configurations')

		if ARGS_loaded is not None and not use_inits_const and ARGS_sorted_by_num_drones:
			ARGS_for_simulation.num_drones = ARGS.num_drones[int(i / num_sims_per_drone_config)]
		else:
			ARGS_for_simulation.num_drones = ARGS.num_drones[i % len(ARGS.num_drones)]
		ARGS_for_simulation.network_message_loss = ARGS.network_message_loss[int(i / len(ARGS.num_drones)) %
																			 len(ARGS.network_message_loss)]

		if ARGS_loaded is None and not use_inits_const:
			INIT_XYZS, INIT_TARGETS = initializer.initialize(
				"random", dist_to_wall=0.25,
				num_points=ARGS_for_simulation.num_drones,
				min_dist=ARGS.r_min, scaling_factor=ARGS.downwash_scaling_factor)
			INIT_XYZS = np.array(INIT_XYZS)
			INIT_TARGETS = np.array(INIT_TARGETS)
		elif not use_inits_const:
			if ARGS_sorted_by_num_drones:
				current_drone_config = ARGS_for_simulation.num_drones
				idx = np.where(num_drones_in_ARGS == current_drone_config)[0][0]
				INIT_XYZS = ARGS_loaded[loaded_sims_per_drone_config[idx] + idx *
										num_sims_per_drone_config_in_ARGS].INIT_XYZS
				INIT_TARGETS = ARGS_loaded[loaded_sims_per_drone_config[idx] + idx *
										    num_sims_per_drone_config_in_ARGS].INIT_TARGETS
				loaded_sims_per_drone_config[idx] += 1
			else:
				print("Hello")
				ARGS_for_simulation.INIT_XYZS = ARGS_loaded[i].INIT_XYZS
				ARGS_for_simulation.INIT_TARGETS = ARGS_loaded[i].INIT_TARGETS

		ARGS_for_simulation.num_targets_per_drone = len(INIT_TARGETS) // ARGS_for_simulation.num_drones

		ARGS_for_simulation.INIT_TARGETS = {ARGS.drone_ids[j]: INIT_TARGETS[j] for j in range(ARGS_for_simulation.num_drones)}

		for j in range(ARGS_for_simulation.num_drones, ARGS_for_simulation.num_drones * ARGS_for_simulation.num_targets_per_drone):
			id = ARGS_for_simulation.drone_ids[j % ARGS_for_simulation.num_drones]
			ARGS_for_simulation.INIT_TARGETS[id] = np.vstack((ARGS_for_simulation.INIT_TARGETS[id], INIT_TARGETS[j]))

		for j in range(ARGS_for_simulation.num_drones):
			id = ARGS_for_simulation.drone_ids[j]
			ARGS_for_simulation.INIT_XYZS_id[id] = INIT_XYZS[j]
			if fly_back:
				ARGS_for_simulation.INIT_TARGETS[id] = np.vstack(
					(ARGS_for_simulation.INIT_TARGETS[id], INIT_XYZS[j]))  # append initial positions to target
			else:
				ARGS_for_simulation.INIT_TARGETS[id] = np.vstack(
					(ARGS_for_simulation.INIT_TARGETS[id],))

		ARGS_for_simulation.INIT_XYZS = INIT_XYZS
		"""
        # C setup to test worst-case deadlock
        ARGS_for_simulation.INIT_XYZS = np.array([[0.5, 2, 2], [1.5, 1.5, 2], [1.75, 1.5, 2], [2, 1.5, 2],
                                                  [2.25, 1.5, 2], [2.5, 1.5, 2], [1.5, 2.5, 2], [1.75, 2.5, 2],
                                                  [2, 2.5, 2], [2.25, 2.5, 2], [2.5, 2.5, 2], [2.5, 2.25, 2],
                                                  [2.5, 2, 2], [2.5, 1.75, 2]])

        ARGS_for_simulation.INIT_TARGETS = np.array([[4, 2, 2], [1.5, 1.5, 2], [1.75, 1.5, 2], [2, 1.5, 2],
                                                     [2.25, 1.5, 2], [2.5, 1.5, 2], [1.5, 2.5, 2], [1.75, 2.5, 2],
                                                     [2, 2.5, 2], [2.25, 2.5, 2], [2.5, 2.5, 2], [2.5, 2.25, 2],
                                                     [2.5, 2, 2], [2.5, 1.75, 2]])
        """

		print("Prepared simulation number " + str(i) + " with " + str(ARGS_for_simulation.num_drones) + " Drones.")
		ARGS_for_simulation.sim_id = i + 1
		ARGS_array.append(ARGS_for_simulation)

	with open(path + "/ARGS.pkl", 'wb') as out_file:
		pickle.dump(ARGS_array, out_file)

	# exit()
	# Run the simulations in parallel mode
	start = time.time()
	if ARGS.hyperparameter_optimization:
		"""hyperparams = {"alpha_2": 1, "alpha_3": 1, "alpha_4": 1, "weight_cooperative": 0.5, "weight_speed": 0.001,
					   "weight_acc": 0.001, "weight_jerk": 0.0001, "target_noise": 0.01,
					   "cooperative_normal_vector_noise": 0.1}
		hyperparams_to_optimize = {"alpha_2": {"bounds": [0, 500]},
								   "alpha_3": {"bounds": [0, 100]},
								   "alpha_4": {"bounds": [0, 100]},
								   "weight_cooperative": {"bounds": [0.0, 1.0]},
								   "weight_speed": {"bounds": [0.0, 1.0]},
								   "weight_acc": {"bounds": [0.0, 1.0]},
								   "weight_jerk": {"bounds": [0.0, 1.0], },
								   "target_noise": {"bounds": [0.0, 0.01], },
								   "cooperative_normal_vector_noise": {"bounds": [0.0, 1.0]}
								   }
		optimizer = hop.HyperparameterOptimizer(hyperparams, hyperparams_to_optimize,
												lambda x: evaluate_hyperparameters(x, ARGS_array))

		for i in range(100):
			optimizer.step()
			# optimizer.show_plots()

		# with open(os.path.join(ARGS.path, "optimization_result.pkl"), 'wb') as out_file:
		# optimizer.prepare_for_pickle()
		# pickle.dump(optimizer, out_file)

		optimizer.show_plots()"""
	else:
		if ARGS.multiprocessing and ARGS.total_simulations > 0:
			max_threads = multiprocessing.cpu_count() - 2
			p = mp.Pool(processes=np.min((max_threads, ARGS.total_simulations)), maxtasksperchild=1)  #
			simulation_logger = [x for x in p.imap(parallel_simulation_wrapper, ARGS_array)]
			p.close()
			p.terminate()
			p.join()

		# run the simulation in batch mode
		else:
			simulations = []
			# initialize simulations
			for current_ARGS in ARGS_array:
				simulations.append(simulation.Simulation(current_ARGS))

			# run simulations
			simulation_logger = []
			for sim in simulations:
				simulation_logger.append(sim.run())

		if ARGS.save_video:
			print("Saving video")
			video_name = 'Video.avi'
			image_folder = ARGS.path + "_simnr_1"
			images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") and img.startswith("ImgPredict1")]
			frame = cv2.imread(os.path.join(image_folder, "ImgPredict1_0.jpg"))
			height, width, layers = frame.shape
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			video = cv2.VideoWriter(os.path.join(image_folder, video_name), fourcc, fps=30, frameSize=(width, height))

			for i in range(1, len(images)):
				video.write(cv2.imread(os.path.join(image_folder, "ImgPredict1_" + str(i) + ".jpg")))

			video.release()

		# Plot the simulation sequentially
		if ARGS.plot:
			# plotter.plot('3D', path=path)
			plotter.plot('2D', path=path)

		print('Total simulation time: ' + str(round(time.time() - start, 2)) + 's.')

		# save the simulation result and ARGS
		# if ARGS.total_simulations != 0 and ARGS.log:
		#    save(simulation_logger, ARGS)

		if ARGS.event_trigger:
			title_addon = 'Event Trigger'
		else:
			title_addon = 'Round Robin'

		if ARGS.plot_batch_results:
			plotter.plot_batch_sim_results('av_transition_time', path=path, title_addon=title_addon)
			# plotter.plot_batch_sim_results('success_rate', path=path, title_addon='Event Trigger')
			# plotter.plot_batch_sim_results('dist_to_target_over_time', path=path)
			# plotter.plot_batch_sim_results('crashed', path=path)

		if not ARGS.log:
			shutil.rmtree(path)

		exit()
