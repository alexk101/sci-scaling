# Set up ROCm environment
ROCM_VERSION=6.2.4
LIBFABRIC_VERSION=1.22.0

# Load required modules
module purge
module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/${ROCM_VERSION}
module load craype-accel-amd-gfx90a
module load cray-hdf5-parallel/1.12.2.11
module load libfabric/${LIBFABRIC_VERSION}

# Set up RCCL and ROCM for distributed training
export LD_LIBRARY_PATH=/lustre/orion/geo163/scratch/kiefera/libs/networking/rccl/aws-ofi-rccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/rocm-${ROCM_VERSION}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cray/libfabric/${LIBFABRIC_VERSION}/lib64:$LD_LIBRARY_PATH

export HDF5_USE_FILE_LOCKING=TRUE # Required for HDF5 on Frontier
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Set thread counts for OpenMP and MKL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Set ROCm environment for AMD GPUs
export PYTORCH_ROCM_ARCH=gfx90a  # MI250X on Frontier uses gfx90a architecture
export ROCM_PATH=/opt/rocm-${ROCM_VERSION}

# Set CUDA_HOME for DeepSpeed to find CUDA
export CUDA_HOME=${ROCM_PATH}
export ROCM_HOME=${ROCM_PATH}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/miopen=${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# Add these environment variables before source export_DDP_vars.sh
# export NCCL_DEBUG=INFO # For debugging network issues
# export NCCL_DEBUG=WARN # For debugging network issues, less verbose
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=ipv4
export NCCL_CROSS_NIC=1       # On large systems, this NCCL setting has been found to improve perf
export NCCL_IB_HCA=hsn0

# Use job-specific temporary directory for NCCL
export NCCL_TEMP_DIR="/tmp/nccl-${SLURM_JOB_ID}"
mkdir -p $NCCL_TEMP_DIR

# Ensure we use the correct network interface
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_ATS=0
export MACHINE=frontier
export ROCM_SMI_PATH="/opt/rocm-${ROCM_VERSION}/libexec/rocm_smi/"

# Get the hostname of the master node for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3442 # Frontier default