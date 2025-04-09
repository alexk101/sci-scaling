#!/bin/bash

CONDA_ENV="/lustre/orion/geo163/world-shared/deepspeed"
SCRATCH="/lustre/orion/geo163/scratch/kiefera"

if [ ! -d "$CONDA_ENV" ]; then
    echo "Loading required modules..."
    module load PrgEnv-gnu/8.6.0
    module load miniforge3/23.11.0-0
    module load rocm/6.2.4
    module load craype-accel-amd-gfx90a
    module load cray-hdf5-parallel/1.12.2.11

    echo "Creating conda environment at $CONDA_ENV..."
    conda create -p $CONDA_ENV python=3.11
    source activate $CONDA_ENV

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing MPI and HDF5 dependencies..."
    MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py
    HDF5_MPI="ON" CC=cc HDF5_DIR=${HDF5_ROOT} pip install --no-cache-dir --no-binary=h5py h5py

    echo "Installing PyTorch with ROCm support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
    pip install ninja

    # Install flash-attn
    if [ ! -d "$SCRATCH/flash-attention" ]; then
        cd $SCRATCH
        echo "Installing flash-attention..."
        git clone https://github.com/ROCm/flash-attention
    else
        rm -rf $SCRATCH/flash-attention/build
    fi

    cd $SCRATCH/flash-attention
    MAX_JOBS=$((`nproc` - 1)) pip install -v .
    cd $SCRATCH/sci-scaling

    echo "Installing other requirements..."
    pip install -r requirements.txt

    echo "Environment setup complete!"
    source deactivate
else
    echo "Environment already exists at $CONDA_ENV"
fi