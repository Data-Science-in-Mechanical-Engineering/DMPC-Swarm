import os

import numpy as np


def merge(files, num_elem):
    data = np.zeros((int(10e6), num_elem))
    offset = 0
    for i, f in enumerate(files):
        d = np.fromfile(f)
        num_trajs = d.shape[0] // num_elem
        d = d.reshape((num_trajs, num_elem))
        data[offset:offset+num_trajs, :] = d
        offset += num_trajs
        print(f"{i} / {len(files)}")
    return data[0:offset, :]


if __name__ == "__main__":
    path = "../../dampc_dataset"

    prediction_horizon = 15
    num_drones = 10
    num_inputs = (num_drones * (prediction_horizon*3+9+1) + 3)
    num_outputs = prediction_horizon*3

    files = [f[7:] for f in os.listdir(path) if f.startswith(f"output")]

    output_merged = merge([os.path.join(path, f"output_{f}") for f in files], num_outputs)
    input_merged = merge([os.path.join(path, f"input_{f}") for f in files], num_inputs)

    print(output_merged.shape)
    print(input_merged.shape)

    output_merged.tofile("/home/alex/torch_datasets/DAMPC/output.npy")
    input_merged.tofile("/home/alex/torch_datasets/DAMPC/input.npy")