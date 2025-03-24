#!/usr/bin/env python

import numpy as np

# from pycrazyswarm import *

import pickle

if __name__ == "__main__":
    # swarm = Crazyswarm()
    # timeHelper = swarm.timeHelper
    # allcfs = swarm.allcfs

    num_drones = 8

    # load demo
    with open('../../simulation_results/dmpc/local_test/simulation_result-8_drones_simnr_1.pkl', 'rb') as f:
        demo = pickle.load(f)

    # Configure the CFs so that the LED ring displays the solid color.
    # Overrides the launch file and the firmware default.
    # for cf in allcfs.crazyflies:
    #     cf.setParam("ring/effect", 7)

    # Take off the CFs
    # allcfs.takeoff(targetHeight=1.0, duration=2.0)
    # timeHelper.sleep(2.5)

    starting_positions = [[-1.0, 1.5, 1.0], [0, 1.0, 1.0], [1.0, 1.5, 1.0],
                                [-1.0, 0, 1.0],   [0.0, 0, 1.0] ,[1.0, 0, 1.0],
                                [-1.0, -1.5, 1.0], [0, -1.0, 1.0]]

    # for i in range(num_drones):
    #     cf = allcfs.crazyflies[i]
    #     cf.goTo(starting_positions[i], 0, 5.0)
    #     cf.setLEDColor(1.0, 1.0, 1.0)
    
    # timeHelper.sleep(5.0)

    # Execute the trajectory
    for t in range(len(demo[f"state_set{0}"])):
        for i in range(num_drones):
            # cf = allcfs.crazyflies[i]
            setpoint = demo[f"state_set{i}"][t]
            # cf.cmdFullState(setpoint[0:3], setpoint[3:6], setpoint[6:9], 0, [0.0, 0.0, 0.0])
        
        if t % 12 == 0:
            for i in range(num_drones):
                # cf = allcfs.crazyflies[i]
                setpoint = demo[f"rgb{i}"][t]
                print(setpoint)
        #         cf.setLEDColor(setpoint[0] / 255, setpoint[1]/ 255, setpoint[2]/ 255)
        # timeHelper.sleepForRate(12 * 5)
