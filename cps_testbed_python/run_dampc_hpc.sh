#!/usr/local_rwth/bin/zsh
#
#SBATCH --job-name=dampc
#SBATCH --output=/home/mf724021/hpc_data/slurm_output/%A_%a.out
#SBATCH --partition=c23test
#SBATCH --account=supp0006
# #SBATCH --account=rwth1483
#SBATCH --cpus-per-task=48
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alexander.graefe@dsme.rwth-aachen.de
#SBATCH --mem-per-cpu=2700MB # memory per node
#SBATCH --time=2-00:00:00
#SBATCH --array=0-0

module load GCCcore/.9.3.0
module load Python/3.8.2

source venv/bin/activate
python dmpc_simulation_caller.py -n DAMPC --param_path parameters/dampc.yaml
deactivate
