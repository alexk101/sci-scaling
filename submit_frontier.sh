#!/bin/bash
#SBATCH -A GEO163 
#SBATCH -J weather-transformer
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8  # 8 tasks per node for 8 MI250X GPUs
#SBATCH --cpus-per-task=7
#SBATCH --signal=B:USR1@60  # Send signal 10 minutes before time limit
#SBATCH -o %x-%j.out

# =============================================================================
# Frontier Submission Script for Weather Transformer Training
# =============================================================================
# This script sets up the environment and launches the training job on Frontier.
# 
# Usage:
#   sbatch -N <num_nodes> submit_frontier.sh --config_name <config> [options]
#
# Options:
#   --config_name <name>   : Specify configuration name from configs.yaml
#   --run_name <name>      : Specify a custom run name (default: auto-generated)
#   --precision <precision>: Specify precision (fp32, fp16, bf16, etc.)
#   --strategy <strategy>  : Specify training strategy (deepspeed, ddp, etc.)
#   --devices <num>        : Number of devices per node (default: 8)
#   --num_nodes <num>      : Number of nodes (default: from SLURM)
#   --enable_flops         : Enable FLOPS profiler
#   --flops_budget <budget>: Set FLOPs budget for training (e.g., 1e20)
#   --time_budget <hours>  : Set time budget in hours (e.g., 5.5)
# =============================================================================

# Parse command line arguments
CONFIG_NAME=""
RUN_NAME=""
PRECISION="bf16"
STRATEGY="deepspeed"
DEVICES=8
NUM_NODES=1  # Default to 1 node
ENABLE_FLOPS=false
FLOPS_BUDGET=""
TIME_BUDGET=""

# Display help message
show_help() {
    echo "Usage: sbatch -N <num_nodes> submit_frontier.sh --config_name <config> [options]"
    echo ""
    echo "Options:"
    echo "  --config_name <name>   : Specify configuration name from configs.yaml"
    echo "  --run_name <name>      : Specify a custom run name (default: auto-generated)"
    echo "  --precision <precision>: Specify precision (fp32, fp16, bf16, etc.)"
    echo "  --strategy <strategy>  : Specify training strategy (deepspeed, ddp, etc.)"
    echo "  --devices <num>        : Number of devices per node (default: 8)"
    echo "  --num_nodes <num>      : Number of nodes (default: from SLURM)"
    echo "  --enable_flops         : Enable FLOPS profiler"
    echo "  --flops_budget <budget>: Set FLOPs budget for training (e.g., 1e20)"
    echo "  --time_budget <hours>  : Set time budget in hours (e.g., 5.5)"
    echo "  --help                 : Show this help message"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config_name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        --num_nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --enable_flops)
            ENABLE_FLOPS=true
            shift
            ;;
        --flops_budget)
            FLOPS_BUDGET="$2"
            shift 2
            ;;
        --time_budget)
            TIME_BUDGET="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CONFIG_NAME" ]]; then
    echo "Error: Configuration name is required (--config_name)"
    show_help
fi

# Check the number of nodes from SLURM environment
if [[ -n "$SLURM_NNODES" ]]; then
    # If running inside SLURM, use the actual allocated nodes
    ACTUAL_NODES="$SLURM_NNODES"
    echo "Running with $ACTUAL_NODES nodes as allocated by SLURM"
    
    # Verify if the requested nodes match what SLURM allocated
    if [[ "$NUM_NODES" != "$ACTUAL_NODES" ]]; then
        echo "Warning: You requested $NUM_NODES nodes but SLURM allocated $ACTUAL_NODES nodes"
        echo "Using SLURM's allocation: $ACTUAL_NODES nodes"
        NUM_NODES="$ACTUAL_NODES"
    fi
else
    # If this script is being submitted, verify we're using sbatch -N
    echo "Important: Make sure to submit this job with: sbatch -N $NUM_NODES submit_frontier.sh ..."
    echo "Otherwise, your node count request may not be properly allocated"
fi

# Handle SLURM signals for graceful termination
cleanup_handler() {
    echo "Received cleanup signal - forwarding to Python process for graceful termination"
    
    # Find the Python processes that are children of this script
    for pid in $(pgrep -P $$ python); do
        echo "Forwarding SIGUSR1 to Python process $pid"
        kill -USR1 $pid
    done
}
trap 'cleanup_handler' USR1

source export_frontier_vars.sh

# Location of the conda environment
export CONDA_ENV_PATH=/lustre/orion/geo163/world-shared/python-envs/deepspeed
source activate ${CONDA_ENV_PATH}

# Define the path to the DeepSpeed environment file
export DS_ENV_FILE="${HOME}/.deepspeed_env"
echo "Using DeepSpeed environment file: ${DS_ENV_FILE}"

# Force DeepSpeed to use CPU operations only
export DS_SKIP_CUDA_CHECK=1

# Set up logging directories
LOG_DIR="logs/$SLURM_JOB_ID"
mkdir -p "$LOG_DIR"

# Create a hostfile for DeepSpeed multi-node training
HOSTFILE="$LOG_DIR/hostfile_${SLURM_JOB_ID}.txt"
# Generate hostfile in the format DeepSpeed expects: hostname slots=N
scontrol show hostnames $SLURM_JOB_NODELIST | while read -r host; do
    echo "$host slots=$DEVICES" >> $HOSTFILE
done
echo "Created hostfile for DeepSpeed in format 'hostname slots=N':"
cat $HOSTFILE

# Use default DeepSpeed config if one isn't explicitly configured
DS_CONFIG="config/ds_config.json"

echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Run Name: $RUN_NAME"
echo "Config Name: $CONFIG_NAME"
echo "Log Directory: $LOG_DIR"
echo "Number of Nodes: $NUM_NODES"
echo "Devices per Node: $DEVICES"
echo "Precision: $PRECISION"
echo "Strategy: $STRATEGY"
echo "FLOPS Profiler: $ENABLE_FLOPS"
echo "DeepSpeed Config: $DS_CONFIG"
echo "Timestamp: $TIMESTAMP"
echo "======================"

export DEVICES

# Generate DeepSpeed environment file if it doesn't exist
if [ ! -f "${HOME}/.deepspeed_env" ]; then
    echo "Generating DeepSpeed environment file..."
    ./setup_deepspeed_env.sh
else
    echo "DeepSpeed environment file already exists at ${HOME}/.deepspeed_env"
fi

echo "Setting up environment variables for distributed training:"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"

# Build the command with srun to avoid launcher conflicts
CMD="srun --export=ALL \
    bash -c 'source export_DDP_vars.sh && \
    ${CONDA_ENV_PATH}/bin/python -u train.py \
    --config config/configs.yaml \
    --config_name $CONFIG_NAME \
    --ds_config $DS_CONFIG \
    --precision $PRECISION \
    --strategy $STRATEGY \
    --num_nodes $NUM_NODES'"

# Add FLOPS profiler if enabled
if [[ "$ENABLE_FLOPS" == true ]]; then
    CMD="${CMD%\'} --enable_flops_profiler'"
fi

# Add FLOPS budget if provided
if [[ -n "$FLOPS_BUDGET" ]]; then
    CMD="${CMD%\'} --flops_budget $FLOPS_BUDGET'"
fi

# Add time budget if provided
if [[ -n "$TIME_BUDGET" ]]; then
    CMD="${CMD%\'} --time_budget $TIME_BUDGET'"
fi

# Print the command for debugging
echo "Running command: $CMD"

# Execute the command
eval $CMD

# Check the exit status
if [[ $? -ne 0 ]]; then
    echo "Error: Training failed with exit code $?"
    exit 1
fi

echo "Training completed successfully"