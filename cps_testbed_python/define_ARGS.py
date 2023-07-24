import argparse
import sys

import useful_scripts.cuboid as cuboid
import useful_scripts.initializer as initializer
import numpy as np
import pickle as p
import compute_unit.setpoint_creator as sc


def define_ARGS():
    parser = argparse.ArgumentParser(
        description='ARGS for the ET-DMPC')
    parser.add_argument('--drones', default={1: "Vicon", 2: "Vicon", 3: "Vicon", 4: "Vicon", 5: "Vicon", 6: "Vicon", 7: "Vicon", 8: "Vicon", 9: "Vicon", 10: "Vicon"}, type=dict,
                        help='drone IDs with name of the testbed', metavar='')
    parser.add_argument('--num_targets_per_drone', default=3, type=int,
                        help='Number of targets', metavar='')
    parser.add_argument('--computing_agent_ids', default=[20, 21], type=list, help='List of Computing Agent IDs')

    parser.add_argument('--log_planned_trajectory', default=True, type=bool,
                        help='Select, whether the planned trajectories should be logged')
    parser.add_argument('--plot_planned_trajectory', default=True, type=bool,
                        help='Select, whether the planned trajectories should be plotted')

    parser.add_argument('--control_freq_hz', default=60, type=int, help='Control frequency in Hz (default: 48)',
                        metavar='')
    parser.add_argument('--communication_freq_hz', default=5, type=int,
                        help='Communication frequency in Hz (default: 10)')
    parser.add_argument('--drone_position_initialization_method', default='random', type=str,
                        help='Method to initialize drone and target position')

    parser.add_argument('--prediction_horizon', default=15, type=int, help='Prediction Horizon for DMPC')

    parser.add_argument('--r_min', default=0.5, type=float, help='minimum distance to each Drone')
    parser.add_argument('--r_min_crit', default=0.2, type=float, help='minimum distance to each Drone')

    parser.add_argument('--use_soft_constraints', default=False, type=bool, help='')
    parser.add_argument('--guarantee_anti_collision', default=True, type=bool, help='')
    parser.add_argument('--soft_constraint_max', default=0.2, type=float, help='')
    parser.add_argument('--weight_soft_constraint', default=0.01, type=float, help='')

    parser.add_argument('--sim_id', default=0, type=int, help='ID of simulation, used for random generator seed')
    parser.add_argument('--INIT_XYZS', default={}, type=dict, help='Initial drone positions')
    parser.add_argument('--INIT_TARGETS', default={}, type=dict, help='Initial target positions')
    parser.add_argument('--testbeds', default={"Vicon": ([-1.8, -1.8, 0.3], [1.8, 1.8, 3.0], [0, 0, 0]),
                                               "Mobile": ([-0.7, -0.7, 0.7], [0.7, 0.7, 1.1], [100, 100, 0])},
                        type=dict, help='Testbeds of the system. Format: name: (min, max, offset)')
    parser.add_argument('--pos_offset', default={}, type=dict, help='Corresponding spatial offsets for drones')
    parser.add_argument('--testbed_size', default=[3.7, 3.7, 3.7], type=list, help='Size of the testbed')

    parser.add_argument('--skewed_plane_BVC', default=False, type=bool,
                        help='Select, whether the BVC planes should be skewed')
    parser.add_argument('--event_trigger', default=True, type=bool,
                        help='Select, whether the event trigger should be used for scheduling')
    parser.add_argument('--downwash_scaling_factor', default=4, type=int,
                        help='Scaling factor to account for the downwash')
    parser.add_argument('--downwash_scaling_factor_crit', default=3, type=int,
                        help='Scaling factor to account for the downwash')
    parser.add_argument('--use_qpsolvers', default=True, type=bool,
                        help='Select, whether qpsolver is used for data planning')

    parser.add_argument('--alpha_1', default=100.0*1, type=bool,
                        help='Weight in event-trigger')
    parser.add_argument('--alpha_2', default=10.0*0, type=bool,
                        help='Weight in event-trigger')
    parser.add_argument('--alpha_3', default=100*0, type=bool,
                        help='Weight in event-trigger')
    parser.add_argument('--alpha_4', default=10.0*0, type=bool,
                        help='Weight in event-trigger')

    parser.add_argument('--remove_redundant_constraints', default=False, type=bool,
                        help='Select, whether a video should be saved')
    parser.add_argument('--min_distance_cooperative', default=0.1, type=float,
                        help='Select, whether a video should be saved')
    parser.add_argument('--weight_cooperative', default=0.0, type=float,
                        help='Select, whether a video should be saved')
    parser.add_argument('--ignore_message_loss', default=False, type=bool,
                        help='')
    parser.add_argument('--use_high_level_planner', default=True, type=bool)

    parser.add_argument('--dynamic_swarm', default=True, type=bool)   # if drones should be added dynamically or not.
    ARGS = parser.parse_args()

    ARGS.drone_ids = list(ARGS.drones.keys())
    ARGS.num_drones = len(ARGS.drones)
    ARGS.num_computing_agents = len(ARGS.computing_agent_ids)

    assert ARGS.num_drones == len(ARGS.drone_ids), "Wrong number of Drone Agent IDs"
    assert ARGS.num_computing_agents == len(ARGS.computing_agent_ids), "Wrong number of Computation Agent IDs"

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
        print(f"Drone {key} in {testbed} with offset {offset}, min_pos: {ARGS.min_positions[key]} and max_pos: {ARGS.max_positions[key]}")
    ARGS.setpoint_creator = sc.SetpointCreator(ARGS.drones, ARGS.testbeds)
    """
    testbed = cuboid.Cuboid(np.array([0.4, 0.4, 0.3]), np.array([ARGS.testbed_size[0], 0, 0]),
                            np.array([0, ARGS.testbed_size[1], 0]),
                            np.array([0, 0, ARGS.testbed_size[2]]))

    ARGS.testbed = testbed


    initializer = initializer.Initializer(testbed, rng_seed=2)


    INIT_XYZS, INIT_TARGETS = initializer.initialize(
        ARGS.drone_position_initialization_method, dist_to_wall=0.25,
        num_points=ARGS.num_drones,
        min_dist=ARGS.r_min, scaling_factor=ARGS.downwash_scaling_factor,
        num_targets=ARGS.num_targets_per_drone * ARGS.num_drones)
    """

    INIT_XYZS = np.array([[-1.0, 1.0, 1.2], [0.0, 0.0, 1.2], [1.0, 0.0, 1.2], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5]])
    INIT_TARGETS = np.array([[1.0, -1.0, 2.5], [-1.1, -1.1, 2.5], [1.1, 1.1, 0.8], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5],
                             [1.0, -1.0, 2.5], [1.1, 1.1, 0.8], [-1.1, -1.1, 2.5], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5],
                             [0.0, -1.0, 1.2], [1.0, -0.05, 1.2], [0.0, 0.05, 1.2], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5]])

    INIT_TARGETS = np.array([[0.0, -1.2, 1.0], [-0.4, 0.0, 1.0], [0.4, 0.0, 1.0], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5],
                             [0.0, 1.2, 1.0], [-0.4, 0, 1.0], [0.4, 0, 1.0], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5],
                             [0.0, -1.2, 1.0], [0.4, 0.1, 1.0], [-0.4, -0.1, 1.0], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5],
                             # [0.0, 1.2, 1.0], [-0.3, 0.1, 1.0], [0.3, -0.1, 1.0],
                             [0.0, -1.2, 1.0], [-0.4, 0.0, 1.0], [0.4, 0.0, 1.0], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5], [1.0, -1.0, 2.5]])


    COOP = 0
    FORWARD = 1
    CIRCLE3 = 2
    COMP_SIM = 3
    TEST = 4
    DEMO = 5
    LIGHTHOUSE = 6

    formation = 10120102012

    if formation == COOP:
        # cooperative movement
        INIT_XYZS = np.array([
            [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0]])
        INIT_TARGETS = np.array(
            [  # [-0.6, 1.0, 1.0], [0.0, 1.0, 1.0], [0.6, 1.0, 1.0], [-0.6, -1.0, 1.0], [0.0, -1.0, 1.0], [0.6, -1.0, 1.0],
                # [-0.6, -1.0, 1.0], [0.0, -1.0, 1.0], [0.6, -1.0, 1.0], [-0.6, 1.0, 1.0], [0.0, 1.0, 1.0], [0.6, 1.0, 1.0],
                [-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.4, 0.0, 1.0], [1.0, 0.0, 1.0], [-0.4, 0.0, 1.0], [0.0, -1.0, 1.0],
                [-1.0, 0.0, 1.0], [0.0, -1.0, 1.0], [0.4, 0.0, 1.0], [1.0, 0.0, 1.0], [-0.4, 0.0, 1.0], [0.0, 1.0, 1.0],
                [-0.6, 1.0, 1.0], [0.0, 1.0, 1.0], [0.6, 1.0, 1.0], [-0.6, -1.0, 1.0], [0.0, -1.0, 1.0], [0.6, -1.0, 1.0],
                [-0.6, -1.0, 1.0], [0.0, -1.0, 1.0], [0.6, -1.0, 1.0], [-0.6, 1.0, 1.0], [0.0, 1.0, 1.0], [0.6, 1.0, 1.0]
            ])
    elif formation == DEMO:
        # cooperative movement
        INIT_XYZS = np.array([
            [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0]])
        INIT_TARGETS = np.array(
            [  # [-0.6, 1.0, 1.0], [0.0, 1.0, 1.0], [0.6, 1.0, 1.0], [-0.6, -1.0, 1.0], [0.0, -1.0, 1.0], [0.6, -1.0, 1.0],
                # [-0.6, -1.0, 1.0], [0.0, -1.0, 1.0], [0.6, -1.0, 1.0], [-0.6, 1.0, 1.0], [0.0, 1.0, 1.0], [0.6, 1.0, 1.0],
                #[-1.0, 0.0, 1.0], [0.0, -1.0, 1.0], [0.4, 0.0, 1.0], [1.0, 0.0, 1.0], [-0.4, 0.0, 1.0], [0.0, 1.0, 1.0],
                [-0.65, 1.0, 1.0], [0.05, 1.0, 1.0], [0.65, 1.0, 1.0], [-0.6, -1.0, 1.0], [0.0, -1.0, 1.0], [0.6, -1.0, 1.0],
                [-0.6, -1.0, 1.0], [0.0, -1.0, 1.0], [0.6, -1.0, 1.0], [-0.6, 1.0, 1.0], [0.0, 1.0, 1.0], [0.6, 1.0, 1.0],
                #[-1.0, 0.0, 1.0], [0.0, -1.0, 1.0], [0.4, 0.0, 1.0], [1.0, 0.0, 1.0], [-0.4, 0.0, 1.0], [0.0, 1.0, 1.0]
            ])

    elif formation == FORWARD:
        # forward flight
        INIT_XYZS = np.array([
            [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0]])
        INIT_TARGETS = np.array(
            [[0.0, 1.8, 1.0], [-0.3, 1.2, 1.0], [0.3, 1.2, 1.0], [0.0, 0.6, 1.0], [-0.6, 0.6, 1.0],
            [0.6, 0.6, 1.0],
            [0.0, -0.3, 1.0], [-0.3, -0.9, 1.0], [0.3, -0.9, 1.0], [0.0, -1.5, 1.0], [-0.6, -1.5, 1.0],
             [0.6, -1.5, 1.0],
             [0.0, 1.8, 1.0], [-0.3, 1.2, 1.0], [0.3, 1.2, 1.0], [0.0, 0.6, 1.0], [-0.6, 0.6, 1.0],
             [0.6, 0.6, 1.0]])

    elif formation == CIRCLE3:
        # 3 drones fly in a circle
        INIT_XYZS = np.array([
            [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0]])

        b_temp = 0.7
        INIT_TARGETS = np.array(
            [[0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [0.0, -0.6, 1.0], [-1.0, -1.0, 1.0],
             [1.0, -1.0, 1.0],
             [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0]
             ])

    elif formation == COMP_SIM:
        # forward flight
        INIT_XYZS = np.array([
            [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0]])
        INIT_TARGETS = np.array([[1.0, 1.0, 1.0], [-1.0, 1.05, 1.0], [1.0, 0.0, 1.0], [-1.0, 0.05, 1.0], [1.0, -1.0, 1.0],
         [-1.0, -1.05, 1.0]
         ])
    elif formation == TEST:
        # forward flight
        INIT_XYZS = np.array([
            [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0]])
        INIT_TARGETS = np.array(
            [
             [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0]
             ])
    elif formation == LIGHTHOUSE:
        INIT_XYZS = np.array([
            [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 1.0, 1.0],  [-1.0, 1.0, 1.0],  [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])

        b_temp = 0.7
        INIT_TARGETS = np.array(
            [[0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0],  [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
             [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [-b_temp, 0.0, 1.0],  [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
             [-b_temp, 0.0, 1.0], [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0],  [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
             [0.0, b_temp, 1.0], [b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0], [-b_temp, 0.0, 1.0],  [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
             [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [-b_temp, 0.0, 1.0],  [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]
            ])
    else:
        INIT_XYZS = np.array([[0, 0, 0] for _ in range(ARGS.num_drones)])
        INIT_TARGETS = np.array([[0, 0, 0] for _ in range(ARGS.num_drones)])


    ARGS.num_targets_per_drone = len(INIT_TARGETS) // ARGS.num_drones

    ARGS.INIT_TARGETS = {ARGS.drone_ids[i]: INIT_TARGETS[i] for i in range(ARGS.num_drones)}

    """ARGS.INIT_TARGETS = {
        ARGS.drone_ids[i]: zone_transform(old_limits, [max_poss[ARGS.drone_ids[i]], min_poss[ARGS.drone_ids[i]]],
                                          INIT_TARGETS[i]) for i in range(ARGS.num_drones)}"""

    for i in range(ARGS.num_drones, ARGS.num_drones * ARGS.num_targets_per_drone):
        id = ARGS.drone_ids[i % ARGS.num_drones]

        ARGS.INIT_TARGETS[id] = np.vstack((ARGS.INIT_TARGETS[id], INIT_TARGETS[i]))

        #ARGS.INIT_TARGETS[id] = np.vstack((ARGS.INIT_TARGETS[id], zone_transform(old_limits,
        #                                                                         [max_poss[id],
        #                                                                          min_poss[id]],
        #                                                                         INIT_TARGETS[i])))


    for i in range(ARGS.num_drones):
        id = ARGS.drone_ids[i]

        ARGS.INIT_XYZS[id] = INIT_XYZS[i]
        ARGS.INIT_TARGETS[id] = np.vstack((ARGS.INIT_TARGETS[id], INIT_XYZS[i]))  # append initial positions to targets

        #ARGS.INIT_XYZS[id] = zone_transform(old_limits, [max_poss[id], min_poss[id]], INIT_XYZS[i]) # INIT_XYZS[i]
        ARGS.INIT_TARGETS[id] = np.vstack((ARGS.INIT_TARGETS[id], ARGS.INIT_XYZS[id]))  # append initial pos to targets

    if ARGS.dynamic_swarm:
        ARGS.num_drones = 0
        ARGS.drone_ids = []

    path = ""
    with open(path + "ARGS_for_testbed.pkl", 'wb') as out_file:
        p.dump(ARGS, out_file)

    np.random.seed(1)


def zone_transform(old_zone, new_zone, point):
    """
    Transforms a point from old zone coordinates to new zone coordinates
    old_zone = [max_pos_old, min_pos_old]
    new_zone = [max_pos_new, min_pos_new]
    point = [x1, y1, z1]
    return [x2, y2, z2]
    """
    d_1 = old_zone[0] - old_zone[1]
    d_2 = new_zone[0] - new_zone[1]
    c_1 = old_zone[1] + d_1 / 2
    c_2 = new_zone[1] + d_2 / 2
    ref_1 = point - c_1
    ref_2 = ref_1 * d_2/d_1
    return c_2 + ref_2

