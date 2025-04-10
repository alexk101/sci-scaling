# Utils Module

This directory contains utility modules for the Weather Transformer project.

## Module Structure

### `data_loader.py`
Handles loading and preprocessing of weather data from HDF5 files:
- `WeatherDataset`: PyTorch Dataset class for efficient loading of weather data
- Includes normalization and lazy loading features

### `loss.py`
Custom loss functions for weather prediction:
- `l2_loss_opt`: Optimized L2 loss for spatial data

### `metrics.py`
Comprehensive metrics collection and tracking utilities:
- `MetricsTracker`: Class for tracking system and model metrics during training
- Weather-specific metrics like weighted RMSE and anomaly correlation
- Platform-specific metrics for different GPU types (NVIDIA, AMD, Apple)

### `visualization.py`
Visualization utilities for spatial weather data:
- `create_comparison_figure`: Creates side-by-side comparison of predictions, ground truth, and error
- `create_spatial_error_map`: Generates heatmaps of spatial error distribution
- `create_lat_lon_error_plots`: Analyzes error patterns across latitude and longitude
- `create_difference_map`: Generates difference maps between predictions and ground truth
- `create_animation`: Creates animations of temporal weather patterns
- All functions support customization for different weather variables

### `logging.py`
Unified logging utilities with support for both TensorBoard and Weights & Biases:
- `Logger`: Main class for unified logging to multiple backends
- `initialize_wandb`: Legacy function for wandb-specific logging
- `log_training_batch`, `log_validation_results`: Legacy functions for backward compatibility
- Supports selecting between logging backends or using both simultaneously
- Handles both scalar metrics and rich visualizations
- Implements safe logging with type checking to prevent errors

### `synthetic_data.py`
Utilities for generating synthetic weather data for testing:
- `create_test_data`: Creates realistic synthetic weather data with spatial correlations
- Built-in normalization statistics generation

## Performance Optimizations

### Tensor Core Utilization
The codebase automatically enables Tensor Core optimizations for NVIDIA GPUs with compute capability 7.0 or higher (Volta, Turing, Ampere, Ada Lovelace, and Hopper architectures):

- **Automatic Detection**: Detects RTX, V100, A100, H100, and T4 GPUs
- **Precision Optimization**: Sets `torch.set_float32_matmul_precision('high')` for optimal performance
- **Performance Impact**: Up to 2-3x speedup for matrix multiplication operations
- **Trade-off**: Slight precision loss is allowed for substantial performance gain

You can manually adjust the precision level in `train.py`:
```python
# Choose one of: 'highest', 'high', 'medium'
torch.set_float32_matmul_precision('medium')  # More performance, less precision
torch.set_float32_matmul_precision('high')    # Balance of performance and precision
torch.set_float32_matmul_precision('highest') # Highest precision, less performance boost
```

Learn more at the [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html).

## Usage Examples

### Unified Logging Example
```python
from utils.logging import Logger

# Initialize the unified logger
logger = Logger(
    backend="both",  # Use both TensorBoard and wandb
    log_dir="logs",  # Directory for TensorBoard logs
    project_name="weather-transformer",
    run_name="experiment-1",
    config=config_dict
)

# Log scalar metrics (safely handles non-numeric values)
logger.log_scalar("train/loss", loss_value, step=global_step)

# Log multiple metrics at once - non-numeric values are filtered out
logger.log_scalars({
    "train/loss": loss_value, 
    "train/lr": learning_rate,
    "train/accuracy": accuracy,
    "train/platform": "cuda"  # String values are automatically filtered out for safety
}, step=global_step)

# Log images
logger.log_image("prediction_vs_truth", figure, caption="Model prediction comparison")

# Log rich visualizations
logger.log_validation_visualizations(
    vis_y_pred=prediction,
    vis_y=ground_truth,
    epoch=current_epoch,
    config=model_config
)

# Close the logger when done
logger.close()
```

### Selecting Logging Backend

You can choose which logging backend to use in several ways:

#### 1. When creating the Logger:

```python
# TensorBoard only
logger = Logger(backend="tensorboard", log_dir="logs")

# Weights & Biases only
logger = Logger(backend="wandb", project_name="my-project")

# Both backends
logger = Logger(backend="both", log_dir="logs", project_name="my-project")

# No logging
logger = Logger(backend="none")
```

#### 2. Via environment variable:

```bash
# Set the backend for the current run
export LOGGING_BACKEND="tensorboard"
python train.py

# Or for wandb only
export LOGGING_BACKEND="wandb"
python train.py

# Or for both
export LOGGING_BACKEND="both"
python train.py
```

#### 3. In the WeatherTrainer constructor:

```python
trainer = WeatherTrainer(
    config=config,
    strategy="deepspeed",
    precision="bf16",
    devices=8,
    num_nodes=1,
    logging_backend="tensorboard"  # Choose backend here
)
```

## Offline Viewing of Metrics

### TensorBoard

TensorBoard logs are stored locally in the `logs` directory and can be viewed with:

```bash
tensorboard --logdir=logs
```

### Weights & Biases

The `logging.py` module supports offline viewing of metrics through wandb local:

1. **Set the mode to offline** in your configuration or environment:
   ```python
   # In code
   os.environ["WANDB_MODE"] = "offline"
   
   # Or via the command line
   export WANDB_MODE="offline"
   ```

2. **Run your training** - metrics will be stored locally

3. **Start a local wandb server**:
   ```bash
   wandb local
   ```

4. **Sync your offline runs**:
   ```bash
   wandb sync wandb/run-TIMESTAMP-ID/
   ```

5. **View in browser** at http://localhost:8080

### Metrics Safety
The logging system has built-in safety features:
- All metrics are type-checked before logging
- Non-numeric values (strings, None, etc.) are automatically filtered out
- Error handling prevents logging failures from crashing training
- Warning messages are printed for skipped metrics
