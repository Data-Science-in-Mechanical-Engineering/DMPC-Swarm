#!/usr/local_rwth/bin/zsh
#
#SBATCH --job-name=eval_dmpc
#SBATCH --output=/work/mf724021/slurm_output/%A_%a.out
#SBATCH --account=p0022034
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alexander.graefe@dsme.rwth-aachen.de
#SBATCH --time=4-00:00:00
#SBATCH --array=0-528

module load GCCcore/.9.3.0
module load Python/3.8.2

source venv/bin/activate
python dmpc_simulation_caller.py -i $SLURM_ARRAY_TASK_ID -n COMPARISON_TRIGGERS
deactivate
