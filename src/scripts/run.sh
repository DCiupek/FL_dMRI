#!/bin/bash
#SBATCH -A # fill the grant
#SBATCH -p # fill the partition
#SBATCH --mem=150G
#SBATCH -c 10
#SBATCH -t 30:00

VENV_DIRECTORY=""
SCRIPT_PATH=""

PARAMETER=""
MAX_VAL=""
B_VAL=""
NO_EPOCHS=""
BATCH_SIZE=""
LR_RATE=""
DATASET=""
MODEL_PATH=""
DATA_TYPE=""
IN_SIZE=""
WANDB=""


if [[ -z "$PARAMETER" || -z "$MAX_VAL" || -z "$B_VAL" || -z "$NO_EPOCHS" || -z "$BATCH_SIZE" || -z "$LR_RATE" || -z "$DATASET" || -z "$MODEL_PATH" || -z "$DATA_TYPE" || -z "$IN_SIZE" || -z "$WANDB" ]]; then
    echo "Error: One or more parameters are not set. Please set all parameters before running the script."
    exit 1
fi

ml python/3.11.5-gcccore-13.2.0

startup () {
    python3 -m venv venv
    source "$VENV_DIRECTORY/bin/activate"
    python3 -m pip install --no-cache-dir -r requirements.txt
}

cd "$WORKDIR" || exit 1
[ ! -e "venv" ] && startup
source "$VENV_DIRECTORY/bin/activate"

if [[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_SRUN_COMM_HOST:-}" ]]; then
    python3 "$SCRIPT_PATH" \
        "${PARAMETER}" \
        "$MAX_VAL" \
        "$B_VAL" \
        "$NO_EPOCHS" \
        "$BATCH_SIZE" \
        "$LR_RATE" \
        "$DATASET" \
        "$MODEL_PATH" \
        "$DATA_TYPE" \
        "$IN_SIZE" \
        "$WANDB"
fi
