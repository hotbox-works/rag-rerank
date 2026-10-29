#!/bin/bash
# Run any project step on a Scholar GPU node, without keeping a notebook open.
#
#   sbatch jobs/run_gpu.sbatch python evaluate.py run --group baselines sweep
#   sbatch jobs/run_gpu.sbatch python train_scorer.py sweep --grad-checkpointing
#   squeue -u $USER                        # is it running?
#   tail -f logs/<jobid>.out               # follow the output
#
# The GPU queue allows at most 4 hours per job. Every step resumes where it stopped (evaluate runs skip finished
# questions, the scorer sweep skips finished configurations), so if a job hits the limit, submit it again.
# Scholar needs an account (see `slist`), a partition, and an explicit GPU request.
# Create the log folder once before the first submit: mkdir -p logs (Slurm will not create it).
#SBATCH -A scholar
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=project2
#SBATCH --output=logs/%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1  # show progress in the log as it happens
cd "${SLURM_SUBMIT_DIR:-$HOME/project2}"
mkdir -p logs
ENV_PATH="${ENV_PATH:-/scratch/scholar/$USER/envs/retr}"
export PATH="$ENV_PATH/bin:$PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "from utils import env; env.setup()"
if [ "$#" -eq 0 ]; then
  echo "Nothing to run. Usage: sbatch jobs/run_gpu.sbatch python evaluate.py run --group baselines"
  exit 1
fi
"$@"