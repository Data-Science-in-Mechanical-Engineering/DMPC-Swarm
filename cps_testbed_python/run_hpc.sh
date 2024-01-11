#!/usr/local_rwth/bin/zsh
#
#SBATCH --job-name=eval_dmpc
#SBATCH --output=/home/mf724021/hpc_data/slurm_output/%A_%a.out
#SBATCH --account=rwth1483
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alexander.graefe@dsme.rwth-aachen.de
#SBATCH --mem-per-cpu=1GB # memory per node
#SBATCH --time=0-00:10:00
#SBATCH --array=0-1

module load GCCcore/.9.3.0
module load Python/3.8.2

source venv/bin/activate
python dmpc_simulation_caller.py -i $SLURM_ARRAY_TASK_ID -n DMPC_test
deactivate