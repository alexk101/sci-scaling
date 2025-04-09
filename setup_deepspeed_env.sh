#!/bin/bash
# setup_deepspeed_env.sh
#
# This script generates the .deepspeed_env file with necessary
# environment variables for running DeepSpeed with ROCm on Frontier

set -e  # Exit immediately if a command exits with non-zero status

# Define the path for the DeepSpeed environment file
DS_ENV_FILE="${DS_ENV_FILE:-${HOME}/.deepspeed_env}"

echo "Setting up DeepSpeed environment file at: ${DS_ENV_FILE}"

# Create the .deepspeed_env file using existing environment variables
# Format: one ENV_VAR=VALUE per line, no comments
cat > "${DS_ENV_FILE}" << EOL
DS_SKIP_CUDA_CHECK=1
CUDA_HOME=${CUDA_HOME:-/opt/rocm-${ROCM_VERSION}}
ROCM_HOME=${ROCM_HOME:-/opt/rocm-${ROCM_VERSION}}
PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx90a}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0}
NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}
NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}
NCCL_IB_HCA=${NCCL_IB_HCA:-hsn0}
NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-ipv4}
FI_CXI_ATS=${FI_CXI_ATS:-0}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
MACHINE=${MACHINE:-frontier}
DEVICES=${DEVICES:-8}
MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)}
MASTER_PORT=${MASTER_PORT:-3442}
EOL

echo "DeepSpeed environment file has been created at: ${DS_ENV_FILE}"
echo "Environment variables will be propagated to all nodes during DeepSpeed launch."
echo ""
echo "Content of ${DS_ENV_FILE}:"
echo "----------------------------"
cat "${DS_ENV_FILE}"
echo "----------------------------"

exit 0
