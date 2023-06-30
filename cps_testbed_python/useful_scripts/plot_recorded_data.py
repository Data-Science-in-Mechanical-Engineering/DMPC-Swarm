import glob
import os
import pickle
from plotter import Plotter
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # initialize the plotter
    plotter = Plotter()

    # load list of all recorded measurements
    path_to_simulations = "../batch_simulation_results/dmpc/dmpc_simulation_results-test_01_14_2022_16_09_19"
    files = glob.glob(path_to_simulations)
    plotter.plot_batch_sim_results('av_transition_time', path=path_to_simulations)

    # plotter.plot('2D', path=path_to_simulations)
    # plotter.plot('3D', path=path_to_simulations)
    exit()