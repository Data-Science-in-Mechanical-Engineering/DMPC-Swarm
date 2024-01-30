import argparse
import sys
from pathlib import Path

import yaml

import useful_scripts.cuboid as cuboid
import useful_scripts.initializer as initializer
import numpy as np
import pickle as p
import compute_unit.setpoint_creator as sc
from dmpc_simulation_caller import parse_params


def define_ARGS():
    parser = argparse.ArgumentParser(
        description='ARGS for the ET-DMPC')
    parser.add_argument('--drones', default={1: "Vicon", 2: "Vicon", 3: "Vicon", 4: "Vicon", 5: "Vicon", 6: "Vicon", 7: "Vicon", 8: "Vicon", 9: "Vicon", 10: "Vicon",
                                             11: "Vicon", 12: "Vicon", 13: "Vicon", 14: "Vicon", 15: "Vicon", 16: "Vicon"}, type=dict,
                        help='drone IDs with name of the testbed', metavar='')
    parser.add_argument('--param_path', default="parameters/testbed_experiment.yaml", type=str,
                        help='yaml file for parameters', metavar='')

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

    parser.add_argument("--num_static_drones", default=8, type=int, help="")
    parser.add_argument('--dynamic_swarm', default=True, type=bool)  # if drones should be added dynamically or not.

    ARGS = parser.parse_args()

    with open(ARGS.param_path, 'r') as file:
        params = yaml.safe_load(file)

    ARGS = parse_params(ARGS, params)

    ARGS.num_drones = 0
    ARGS.drone_ids = []
    ARGS.computing_agent_ids = []

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

    ARGS.setpoint_creator = sc.SetpointCreator(ARGS.drones, ARGS.testbeds, demo_setpoints=sc.CIRCLE_PERIODIC)

    path = ""
    with open(path + "ARGS_for_testbed.pkl", 'wb') as out_file:
        p.dump(ARGS, out_file)

    np.random.seed(1)
