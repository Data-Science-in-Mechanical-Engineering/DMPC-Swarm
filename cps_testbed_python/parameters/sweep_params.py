import copy

import yaml
import os
from pathlib import Path
import copy


def sweep_parameters(params, params_sweep):
    params_results = []

    keys = list(params_sweep.keys())
    key = keys[0]

    if len(keys) != 1:
        new_params_sweep = copy.deepcopy(params_sweep)
        new_params_sweep.pop(key)

        new_params = sweep_parameters(params, new_params_sweep)
    else:
        new_params = [copy.deepcopy(params)]

    for v in params_sweep[key]:
        for p in new_params:
            params_results.append(copy.deepcopy(p))
            params_results[-1][key] = v

    return params_results


if __name__ == "__main__":
    param_path = "batch_simulation.yaml"
    param_sweep_path = "comparison_dmpc_mlr_dmpc.yaml"

    # param_path = "../src/parameters/electric_devices/electric_devices.yaml"
    # param_sweep_path = "../src/parameters/electric_devices/electric_devices_sweep.yaml"

    with open(param_path, "r") as file:
        params = yaml.safe_load(file)
    with open(param_sweep_path, "r") as file:
        params_sweep = yaml.safe_load(file)

    name_sweep = params_sweep["name_sweep"][0]
    print("Generating params for: " + name_sweep)
    param_target_path = f"/work/mf724021/hpc_parameters/{name_sweep}/"
    if not os.path.exists(param_target_path):
        os.makedirs(param_target_path)

    swept_params = sweep_parameters(params, params_sweep)
    for comb_idx in range(len(swept_params)):
        with open(param_target_path + f"params{comb_idx}.yaml", "w") as file:
            params = yaml.safe_dump(swept_params[comb_idx], file)

    print(f"{len(swept_params)} parameter settings")


