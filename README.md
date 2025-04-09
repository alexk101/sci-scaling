# Weather Transformer: Deep Learning for Weather Prediction

This repository contains an experimental implementation of a Vision Transformer (ViT) for weather prediction, built with PyTorch, PyTorch Fabric, and DeepSpeed. The model is designed to predict future weather states from current weather data, with a focus on scalability and efficiency.

## Features

- **Vision Transformer Architecture**: Custom implementation optimized for weather data
- **Flexible Training**: PyTorch Fabric integration for easy switching between training strategies (DeepSpeed, FSDP, DDP)
- **Distributed Training**: Support for multi-GPU and multi-node training
- **Scaling Experiments**: Support for various scaling experiments (model size, embedding, sequence length, resolution)
- **Flexible Configuration**: YAML-based configuration system
- **Efficient Data Loading**: Optimized HDF5 data loading with lazy loading and normalization
- **Memory Optimizations**: Support for mixed precision training and flash attention
- **AMD GPU Support**: Optimized for AMD GPUs with ROCm support
- **Comprehensive Metrics**: Multiple evaluation metrics including MSE, MAE, and RMSE
- **FLOPs Profiling**: DeepSpeed integration for detailed computational cost analysis
- **Synthetic Data Generation**: Built-in utilities for generating realistic test data
- **Apple Silicon Support**: Optimized for Mac with MPS (Metal Performance Shaders) acceleration

## Model Architecture

The Weather Transformer is based on the Vision Transformer architecture with several optimizations for weather data:

- **Input Processing**:
  - Takes weather data as input (shape: [batch_size, channels, height, width])
  - Patches the input into non-overlapping patches
  - Linear projection of patches to embedding space

- **Transformer Encoder**:
  - Multi-head self-attention with AMD-optimized flash attention
  - MLP blocks with GELU activation
  - Layer normalization
  - Dropout for regularization
  - Configurable number of layers and heads

- **Output Processing**:
  - Linear projection back to weather data space
  - Reshapes to match target dimensions
  - Bilinear upsampling to restore original spatial resolution
  - Outputs predictions with same dimensions as input

## Configuration System

The project has transitioned to a YAML-based configuration system while maintaining backward compatibility with legacy dataclass configurations.

### YAML-Based Configuration (Recommended)

All configurations are defined in `config/configs.yaml` using YAML with anchors and references for inheritance and reuse. This approach provides several benefits:

- **Simplicity**: Configuration is more readable and easier to modify
- **Inheritance**: Base configurations can be extended for specific use cases
- **Flexibility**: Configuration can be modified without code changes
- **Organization**: Related configurations are grouped in a single file

Example of YAML configuration with inheritance:

```yaml
# Base configuration that others can inherit from
base: &base
  # Model architecture parameters
  embed_dim: 384
  num_layers: 12
  num_heads: 8
  
  # Training parameters
  batch_size: 32
  learning_rate: 1e-4
  
# Inherit base config and override specific values
small_model: 
  <<: *base
  embed_dim: 128
  num_layers: 4
  num_heads: 4
  batch_size: 16
  
# Another configuration inheriting from base
large_model:
  <<: *base
  embed_dim: 768
  num_layers: 24
  num_heads: 16
```

Usage example:

```bash
# Using a specific configuration from configs.yaml
python train.py --config config/configs.yaml --config_name large_model
```

### Available Configurations

The `configs.yaml` file includes several predefined configurations:

- **base**: Standard model with 12 layers, 384 embedding dim
- **local**: Smaller model for local testing (4 layers, 128 embedding dim)
- **small/medium/large**: Different model sizes for scaling experiments
- **bs16/bs32/bs64/bs128**: Batch size scaling configurations
- **seq4/seq8/seq16**: Sequence length scaling configurations
- **cuda_optimized**: Configuration optimized for NVIDIA GPUs
- **mps_optimized**: Configuration optimized for Apple Silicon

### Creating Custom Configurations

You can easily add your own configurations to `configs.yaml`:
```yaml
my_custom_config:
  <<: *base  # Inherit from base configuration
  # Override specific parameters
  num_layers: 8
  embed_dim: 256
  batch_size: 24
  name: "my_experiment"
  description: "My custom experiment configuration"
```

Then use it:
```bash
python train.py --config_name my_custom_config
```

### ConfigWrapper for Compatibility

The code uses a `ConfigWrapper` class that allows seamless usage of both dataclass and dictionary configurations:

- Attribute access with dot notation works regardless of the underlying format
- Same code can work with either dataclass or dictionary configurations
- Provides helpful utility methods for safely accessing configuration values

### Legacy Configuration Support

The legacy dataclass-based configuration system has been fully removed from the codebase. All configurations now use the YAML format exclusively.

The `ConfigWrapper` class still provides backward compatibility for code that expects a dataclass-like interface with attribute access, but internally it works with dictionary configurations only.

If you have custom scripts that relied on the old dataclass system, you'll need to update them to use the YAML configuration system.

## GPU Support

This implementation supports multiple GPU platforms:

### AMD GPU Support

Optimized for AMD GPUs with ROCm support. Key features include:

- **Flash Attention**: AMD-optimized implementation for faster attention computation
- **Mixed Precision**: Support for bfloat16 (bf16) precision
- **ROCm Integration**: Native support for AMD's ROCm platform
- **Performance Optimizations**:
  - Hardware-aware attention algorithm
  - Reduced memory bandwidth usage
  - Efficient on-chip computation
  - Optimized for AMD GPU architecture

### NVIDIA GPU Support

Full support for NVIDIA GPUs with CUDA:

- **Mixed Precision**: Support for fp16 and bf16 precision
- **CUDA Optimizations**: Efficient CUDA kernels
- **Multi-GPU Training**: Efficient scaling across multiple NVIDIA GPUs

### Apple Silicon (MPS) Support

Optimized for Macs with Apple Silicon:

- **MPS Backend**: Uses Metal Performance Shaders for GPU acceleration
- **Precision Support**: Uses 32-bit precision for maximum compatibility
- **Memory Optimization**: Reduced memory usage for Mac hardware

## Data Format

The model expects weather data in HDF5 format with the following structure:

### Real Data Format

- Data stored in directories: train/, valid/, test/
- Each directory contains multiple HDF5 files (typically one per year)
- Each HDF5 file contains a dataset named 'fields'
- Shape: [time_steps, channels, height, width]
- Data type: float32
- Normalized using pre-computed means and standard deviations

Required files:

- `train/*.h5`, `valid/*.h5`, `test/*.h5`: Weather data files
- `stats/global_means.npy`: Pre-computed means for normalization
- `stats/global_stds.npy`: Pre-computed standard deviations for normalization

### Synthetic Data Generation

The codebase includes utilities to generate synthetic weather data for testing:

```python
from utils.synthetic_data import create_test_data

# Generate synthetic data with default settings
data_dir = create_test_data()

# Generate data with custom dimensions
data_dir = create_test_data(height=64, width=128)
```

This creates:

- Multiple HDF5 files in train/ and valid/ directories
- Each file represents a "year" of synthetic weather data
- Realistic weather-like patterns with spatial and temporal correlations
- Proper normalization statistics in stats/ directory

## Usage

### Running a Single Experiment

  1. Create a configuration file:

   ```bash
   cp config/base_config.py config/my_experiment.py
   # Edit my_experiment.py with your desired settings
   ```

  2. Run the experiment:

   ```bash
   python train.py \
      --config config/my_experiment.py \
      --strategy deepspeed  # or "fsdp", "ddp"
      --precision bf16     # or "fp16", "fp32", "32-true" (for Mac)
      --devices 8         # number of GPUs
      --num_nodes 1       # number of nodes
   ```

### Running Scaling Experiments

The codebase includes a script to automate scaling experiments across various dimensions of the model and training process. The script has been updated to use the new YAML-based configuration system.

### Basic Usage

```bash
python run_scaling_experiments.py --base_config config/configs.yaml --base_config_name base
```

This will run a series of experiments using the default scaling types (model, embedding, sequence, batch) and scaling factors (0.5, 1.0, 2.0, 4.0).

### Advanced Usage

You can customize the scaling experiments by specifying the scaling types and factors:

```bash
python run_scaling_experiments.py \
  --base_config config/configs.yaml \
  --base_config_name base \
  --scale_types model embedding sequence batch \
  --scale_factors 0.5 1.0 2.0 4.0 \
  --strategy deepspeed \
  --num_nodes 1 \
  --devices_per_node 8
```

### Available Scaling Types

- `model`: Scales the model architecture (number of layers and heads)
- `embedding`: Scales the embedding dimension
- `sequence`: Scales the sequence length
- `resolution`: Scales the input image resolution
- `batch`: Scales the batch size (and adjusts learning rate)
- `data`: Scales the data subset size
- `precision`: Changes precision mode (0=fp32, 1=fp16, 2=bf16)
- `node`: Sets the number of nodes to use

### Understanding the Script

The script creates a scaled configuration for each combination of scaling type and factor, then runs an experiment with that configuration. Each experiment:

1. Creates a new YAML configuration file in the `experiments/` directory
2. Runs the training script with the new configuration
3. Outputs results to the specified log directory

The scaling process directly modifies appropriate parameters in the YAML configuration based on the scaling type.

### SLURM Integration

For HPC environments, use the provided SLURM submission script:

```bash
sbatch submit_frontier.sh \
    --config config/my_experiment.py \
    --strategy deepspeed \
    --precision bf16 \
    --devices 8 \
    --num_nodes 1
```

### Local Testing

For local testing on a single device (CPU, CUDA GPU, or Apple Silicon), follow these steps:

1. **Environment Setup**:

   ```bash
   # Create and activate a new conda environment
   conda create -n weather-transformer python=3.8
   conda activate weather-transformer
   
   # Install PyTorch with appropriate backend
   # For CUDA:
   pip install torch torchvision torchaudio
   # For MPS (Apple Silicon):
   pip install torch torchvision torchaudio
   # For CPU only:
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
   
   # Install other dependencies
   pip install h5py numpy wandb pytorch-lightning
   ```

2. **Generate Test Data**:

   The synthetic data generation is built into the training process:

   ```bash
   # The script will automatically generate synthetic data
   # when you run train_local.py
   ```

3. **Run Local Training**:

   ```bash
   python train_local.py
   ```

The local testing configuration includes:

- Smaller model size (4 layers, 4 heads, 128 embedding dim)
- Reduced batch size (4)
- Fewer epochs (10)
- Smaller image dimensions (64×128) for faster computation
- Hardware-specific optimizations detected automatically
- Simplified training setup for local testing

### Device-Specific Features

- **CUDA Support**: Automatic detection and use of NVIDIA GPUs
  - Native PyTorch Flash Attention for PyTorch 2.0+
  - External flash-attn package support
- **MPS Support**: Automatic detection and use of Apple Silicon GPU
- **CPU Fallback**: Runs on CPU if no GPU is available
- **Synthetic Data**: Built-in data generation with proper directory structure
- **Optimized Configuration**: Settings tuned for local development
- **Automatic Precision**: Selects appropriate precision based on hardware

### Troubleshooting

If you encounter issues:

1. **Memory Issues**:
   - Reduce batch size in `config/local_config.py`
   - Decrease model size (layers, heads, embed_dim)
   - Use CPU if GPU memory is insufficient

2. **Performance Issues**:
   - Adjust number of workers in training config
   - Modify data loading parameters
   - Consider using smaller synthetic dataset

3. **Data Loading Issues**:
   - Ensure directory structure matches expected format (train/, valid/ directories)
   - Check HDF5 file dimensions and format
   - Verify normalization statistics are available

## Directory Structure

```
experimental/
├── config/
│   ├── base_config.py      # Configuration classes
│   ├── local_config.py     # Local testing configuration
│   ├── base_config.yaml    # Base configuration
│   └── ds_config.yaml      # DeepSpeed configuration
├── models/
│   ├── weather_transformer.py  # Main model implementation
│   └── architecture.py     # Architecture components (attention, blocks)
├── utils/
│   ├── data_loader.py      # Data loading utilities
│   ├── loss.py             # Loss functions
│   ├── metrics.py          # Evaluation metrics
│   └── synthetic_data.py   # Synthetic data generation
├── train.py                # Main training script
├── train_local.py          # Local training script for single device
├── run_scaling_experiments.py  # Experiment runner
├── submit_frontier.sh      # SLURM submission script
├── env_setup.sh            # Environment setup script
└── export_frontier_vars.sh # Export environment variables for Frontier
```

## Dependencies

- Python 3.8+
- PyTorch 2.0+
  - With ROCm support (AMD GPUs)
  - With CUDA support (NVIDIA GPUs)
  - With MPS support (Apple Silicon)
- PyTorch Fabric/Lightning
- DeepSpeed (optional, for DeepSpeed strategy)
- Flash Attention 2.0+ (optional, for optimized attention)
- h5py
- numpy
- wandb (optional, for logging)

## Performance Optimizations

1. **Memory Efficiency**:
   - Mixed precision training (bf16, fp16, 32-true for Mac)
   - Optimized attention mechanisms (flash attention where available)
   - Gradient checkpointing option
   - Efficient data loading with lazy file handling
   - Automatic memory optimization through Fabric
   - Reduced worker count on Mac to prevent multiprocessing issues

2. **Training Efficiency**:
   - Multiple training strategies (DeepSpeed, FSDP, DDP)
   - Automatic gradient scaling
   - Configurable batch sizes and workers
   - Optimized data loading pipeline
   - Built-in performance monitoring
   - Platform-specific optimizations (ROCm, CUDA, MPS)

3. **Scaling Features**:
   - Model size scaling
   - Embedding dimension scaling
   - Sequence length scaling
   - Spatial resolution scaling
   - Easy switching between scaling strategies

## Performance Monitoring

### FLOPs Profiling

The codebase includes DeepSpeed's FLOPs Profiler for detailed computational cost analysis. This is particularly useful for the scaling experiments outlined in `EXPS.md`. The profiler can be enabled in the DeepSpeed configuration:

```yaml
flops_profiler:
  enabled: true
  profile_step: 1  # Profile every N steps
  module_depth: -1  # -1 for all modules
  top_modules: 3  # Number of top modules to show
  detailed: true  # Show detailed profile
  output_file: "flops_profile.txt"  # Output file for the profile
```

The profiler will:

1. Track total FLOPs, MACs, and parameters
2. Log metrics to Weights & Biases if enabled
3. Generate detailed module-wise profiles
4. Help identify computational bottlenecks

This is particularly useful for:

- Understanding model scaling behavior
- Identifying performance bottlenecks
- Optimizing model architecture
- Validating scaling laws

## Budget-Based Training

The codebase supports budget-based training, allowing you to set either a FLOPs budget or a time budget for your training runs instead of a fixed number of epochs. This is particularly useful for:

- Ensuring consistent computational cost across experiments
- Fair comparisons between different model architectures
- Managing training time on shared computing resources
- Scheduling jobs with known duration requirements

### Budget Types

Two types of budgets are supported:

1. **FLOPs Budget**: Train until a specific number of floating-point operations is reached
2. **Time Budget**: Train for a specific duration in hours

### Using Budget-Based Training

Budget-based training can be enabled through YAML configuration or command-line arguments:

#### Via configuration file:

```yaml
training:
  # Other parameters...
  epochs: 100  # Maximum epochs (will be overridden by budget)
  flops_budget: 1e20  # Stop after 1e20 FLOPs
  # OR
  time_budget_hours: 4.5  # Stop after 4.5 hours
```

#### Via command-line:

```bash
# Train with a FLOPs budget
python train.py --config_name base --flops_budget 1e20

# Train with a time budget
python train.py --config_name base --time_budget 4.5
```

### How Budget Tracking Works

1. **FLOPs Budget**: 
   - Uses DeepSpeed's FLOPs Profiler to measure actual FLOPs usage
   - Profiles the first batch to estimate FLOPs per epoch
   - Tracks accumulated FLOPs and stops when budget is reached
   - Provides progress updates in logs

2. **Time Budget**:
   - Tracks elapsed time using Python's `time` module
   - Calculates minutes per epoch based on actual measurements
   - Provides estimates of remaining epochs within budget
   - Stops when time budget is reached

### Budget Calculator Tool

A `budget_calculator.py` script is provided to help plan experiments with budgets:

```bash
# Estimate epochs for a time budget
python budget_calculator.py time_to_epochs --hours 4 --minutes_per_epoch 10

# Estimate time for a number of epochs
python budget_calculator.py epochs_to_time --epochs 50 --minutes_per_epoch 10

# Generate a budget table for planning SLURM jobs with different node counts
python budget_calculator.py budget_table --min_nodes 1 --max_nodes 32 --minutes_per_epoch 30

# Calculate SLURM job settings
python budget_calculator.py slurm_settings --hours 8 --nodes 4 --minutes_per_epoch 30
```

### Budget-Based SLURM Jobs

For SLURM-based HPC environments, you can use the budget system to optimize resource usage:

```bash
# Submit a job with a time budget
sbatch submit_frontier.sh --config_name base --time_budget 8

# Submit a job with a FLOPs budget
sbatch submit_frontier.sh --config_name base --flops_budget 1e20
```

The training will automatically stop when the budget is reached, ensuring efficient use of your allocation.

## Checkpoint Management

The model training includes an intelligent checkpoint management system:

- **Configurable retention**: Set `max_checkpoints` to control how many checkpoints to keep
- **Automatic rotation**: Automatically deletes oldest checkpoints when limit is reached
- **Zero means unlimited**: Set `max_checkpoints=0` to keep all checkpoints
- **Default value**: By default, only the 3 most recent checkpoints are kept
- **Run-specific folders**: Checkpoints are organized in separate folders by experiment name

### Checkpoint Organization

Checkpoints are automatically organized in the following directory structure:

```
checkpoints/
├── experiment_name_1/
│   ├── checkpoint_epoch_5.pt
│   ├── checkpoint_epoch_10.pt
│   └── checkpoint_epoch_15.pt
├── experiment_name_2/
│   ├── checkpoint_epoch_5.pt
│   ├── checkpoint_epoch_10.pt
│   └── checkpoint_epoch_15.pt
└── ...
```

This organization makes it easy to:
- Keep track of checkpoints from different experiments
- Apply different retention policies per experiment
- Easily find the latest checkpoint for a specific run
- Clean up checkpoints from completed experiments

### Configuration Options

You can configure checkpoint management in several ways:

### Via configuration file:

```yaml
training:
  # Other parameters...
  save_frequency: 5  # Save every 5 epochs
  checkpoint_dir: "checkpoints"
  max_checkpoints: 3  # Keep only 3 most recent checkpoints
```

### Via constructor parameter:

```python
trainer = WeatherTrainer(
    config=config,
    strategy="deepspeed",
    precision="bf16",
    devices=8,
    num_nodes=1,
    max_checkpoints=5  # Keep 5 most recent checkpoints
)
```

### Via environment variable:

```bash
# Set an environment variable to override default
export MAX_CHECKPOINTS=10
python train.py
```

This helps manage disk space when training large models over many epochs.

## Single-Task Multi-GPU Setup for Frontier

This repository has been updated to support running on Frontier with a single-task multi-GPU configuration, where each node runs a single task that has access to all 8 GPUs on the node. This approach has several advantages:

1. **Simplified Process Management**: One task per node with access to all GPUs means cleaner process and memory management.
2. **Reduced Communication Overhead**: Minimizes unnecessary intra-node communication overhead.
3. **Better DeepSpeed Integration**: DeepSpeed works well with this model as it can handle GPU assignments internally.
4. **Improved Resource Utilization**: All CPUs are allocated to the single task, providing ample resources for data loading and preprocessing.

### Key Changes:

1. **Updated DeepSpeed Configuration**:
   - Modified `config/ds_config_safe.json` to better handle the single-task multi-GPU setup
   - Disabled mixed precision by default to avoid dtype issues
   - Added proper communication settings for multi-node operation

2. **Optimizer Validation Utilities**:
   - Added `utils/validate_optimizer.py` to handle DeepSpeed's internal optimizer state
   - Provides robust error handling for common DeepSpeed issues with mixed data structures
   - Automatically fixes the "KeyError: 0" issue that can occur in DeepSpeed ZeRO optimizer

3. **Enhanced Training Process**:
   - Updated backward pass handling to properly manage dtype conversion 
   - Added gradient flow checks to verify model parameter updates
   - Improved error recovery, especially for mixed precision and multi-GPU setups

4. **Testing and Diagnostics**:
   - Created `test_model.py` to diagnose issues in the single-task multi-GPU environment
   - Provides detailed logging of distributed environment variables
   - Tests forward and backward passes with different precision settings

### Usage:

To run training with the single-task multi-GPU setup, use the provided `submit_frontier.sh` script:

```bash
sbatch -N <num_nodes> submit_frontier.sh --config_name <config> --strategy deepspeed --precision fp32
```

For initial testing, it's recommended to:
1. Use `fp32` precision to avoid dtype issues
2. Use the `ds_config_safe.json` configuration
3. Run the diagnostics script first: `srun python test_model.py --precision 32-true`

### Troubleshooting:

If you encounter issues:

1. Check the logs for "KeyError: 0" or "dtype mismatch" errors
2. Verify that the DeepSpeed configuration matches your environment
3. Run the test script to isolate the problem
4. Ensure the batch size settings are consistent between the script and DeepSpeed config

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
