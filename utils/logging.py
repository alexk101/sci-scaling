import os
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Any
from torch.utils.tensorboard import SummaryWriter
import logging
import sys
from dataclasses import asdict
from utils.config_utils import Config, LoggingBackend
import datetime
import time

from utils.visualization import (
    create_comparison_figure, 
    fig_to_image, 
    create_animation,
    create_spatial_error_map,
    create_lat_lon_error_plots,
    create_difference_map,
    create_synthetic_sequence
)

# ------------------------------------------------------------------------
# Directory Structure
# ------------------------------------------------------------------------
# logs/
#   └── run_name_timestamp/         # Main run directory
#       ├── rank_logs/              # Rank-specific log files
#       │   ├── rank_0.log
#       │   ├── rank_1.log
#       │   └── ...
#       ├── tensorboard/           # TensorBoard logs
#       ├── checkpoints/           # Model checkpoints
#       └── images/                # Generated images
#
# ------------------------------------------------------------------------

# Note: We're using LoggingBackend from utils.config_utils

def init_run_directory(log_dir: str = "logs") -> Path:
    """Create and initialize the run directory structure and configure logging.
    
    Args:
        log_dir: Base directory for logs
        
    Returns:
        Path to the run directory
    """
    # Get rank information
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create timestamp and job ID for directory naming
    job_id = os.environ.get('SLURM_JOB_ID', f'local_{timestamp}')
    
    # Create main run directory
    run_dir = Path(log_dir) / f"{job_id}"
    
    # First, EVERY rank creates the top level directory
    # This is important because the run_dir could be on network filesystem
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Then only rank 0 creates subdirectories
    if rank == 0:
        # Create subdirectories
        (run_dir / "rank_logs").mkdir(exist_ok=True)
        (run_dir / "tensorboard").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "images").mkdir(exist_ok=True)
        
        # Create a flag file to indicate directories are ready
        flag_file = run_dir / ".directories_ready"
        with open(flag_file, 'w') as f:
            f.write(f"Directories created at {datetime.datetime.now().isoformat()}")
    
    # Synchronize to ensure all ranks see the directories
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # First barrier to ensure rank 0 has created directories
        torch.distributed.barrier()
        
        # All non-zero ranks check for the flag file with retry
        if rank != 0:
            # Try multiple times with exponential backoff
            flag_file = run_dir / ".directories_ready"
            max_retries = 5
            for retry in range(max_retries):
                if flag_file.exists():
                    break
                wait_time = 0.1 * (2 ** retry)  # Exponential backoff
                print(f"Rank {rank}: Waiting {wait_time:.2f}s for directories to be created (attempt {retry+1}/{max_retries})")
                time.sleep(wait_time)
            
            if not flag_file.exists():
                print(f"WARNING: Rank {rank} could not find flag file after {max_retries} retries")
        
        # Second barrier to ensure all ranks proceed together
        torch.distributed.barrier()
    else:
        # If not using distributed training, just wait a bit
        time.sleep(1)
    
    # IMPORTANT: Every rank creates its own specific log directory to avoid race conditions
    # Create rank_logs directory if it doesn't exist yet (can happen if rank 0 failed to create it)
    rank_logs_dir = run_dir / "rank_logs"
    rank_logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up Python logging
    
    # Create a custom filter that only shows logs from rank 0
    class RankFilter(logging.Filter):
        def __init__(self, rank):
            super().__init__()
            self.rank = rank
        
        def filter(self, record):
            # Always show logs from rank 0
            if self.rank == 0:
                return True
            
            # For other ranks, only show critical errors or logs explicitly marked for all ranks
            if hasattr(record, 'show_all_ranks') and record.show_all_ranks:
                return True
            
            if record.levelno >= logging.ERROR:
                return True
                
            # Filter out non-critical logs from non-zero ranks
            return False
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure format to show rank
    formatter = logging.Formatter(
        '[%(asctime)s][Rank %(rank)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Only show logs from rank 0 by default
    if rank != 0:
        console_handler.addFilter(RankFilter(rank))
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    
    # Create rank-specific log file
    # Make sure parent directory exists
    node_name = os.environ.get('SLURMD_NODENAME', 'unknown')
    rank_log_file = rank_logs_dir / f"rank_{rank:04d}_{node_name}.log"
    
    try:
        # Try to create the file handler
        file_handler = logging.FileHandler(rank_log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # If file handler creation fails, log to console only with an error message
        print(f"WARNING: Failed to create log file {rank_log_file}: {str(e)}")
        print(f"WARNING: Continuing with console logging only for rank {rank}")
    
    # Create an adapter to add rank to log records
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    # Log initial setup
    logging.info(f"Logging configured for rank {rank} (local rank {local_rank}) out of {world_size}")
    logging.info(f"Using run directory: {run_dir}")
    logging.info(f"Rank-specific log file: {rank_log_file}")
    
    return run_dir

# Custom logger to handle metrics and visualization
class Logger:
    """Logger class for tracking metrics, visualizations, and checkpoints."""
    
    def __init__(
        self, 
        run_dir: Path,
        config: Optional[Config] = None,
        backend: LoggingBackend = LoggingBackend.TENSORBOARD,
        project_name: str = "weather-transformer",
    ):
        """Initialize the metrics logger.
        
        Args:
            run_dir: Path to the run directory (created init_run_directory)
            config: Configuration object
            backend: Logging backend to use
            project_name: Name of the project (for wandb)
        """
        self.run_dir = run_dir
        self.config = config
        self.backend = backend
        self.project_name = project_name
        
        # Get rank information using variables from export_DDP_vars.sh
        self.world_rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Get the regular Python logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize backend loggers based on rank
        self.tensorboard = None
        self.wandb = None
        
        # Only initialize metrics loggers on rank 0
        if self.world_rank == 0:
            if self.backend in [LoggingBackend.TENSORBOARD, LoggingBackend.BOTH]:
                self._initialize_tensorboard()
            if self.backend in [LoggingBackend.WANDB, LoggingBackend.BOTH]:
                self._initialize_wandb()
            # Log the run configuration
            self._log_run_config()
    
    def _initialize_tensorboard(self):
        """Initialize TensorBoard logger."""
        tensorboard_dir = self.run_dir / "tensorboard"   
        # Now initialize TensorBoard
        self.tensorboard = SummaryWriter(log_dir=str(tensorboard_dir))
        self.logger.info(f"TensorBoard logging initialized at {tensorboard_dir}")
    
    def _initialize_wandb(self):
        """Initialize Weights & Biases logger."""
        try:
            # Extract config details for wandb
            wandb_config = None
            if self.config:
                wandb_config = asdict(self.config)
            
            # Initialize wandb
            wandb.init(
                project=self.project_name,
                name=self.run_dir.name,
                config=wandb_config,
                dir=str(self.run_dir),
            )
            self.wandb = wandb
            self.logger.info(f"Weights & Biases logging initialized for project {self.project_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Weights & Biases: {e}")
            self.wandb = None
    
    def _log_run_config(self):
        """Log the run configuration to all active backends."""
        if self.config is None:
            return
        
        # Log configuration as text and as hyperparameters
        config_dict = asdict(self.config)
        
        if self.tensorboard:
            # Instead of using add_hparams which creates a subdirectory,
            # just log the config as text
            config_str = str(config_dict)
            self.tensorboard.add_text("config", config_str)
            
            # Log individual hyperparameters as scalars for better visualization
            # This won't create subdirectories
            flattened_config = self._flatten_dict(config_dict)
            for param_name, param_value in flattened_config.items():
                if isinstance(param_value, (int, float)):
                    self.tensorboard.add_scalar(f"hparams/{param_name}", param_value, 0)
            
            # Flush to ensure data is written
            self.tensorboard.flush()
        
        if self.wandb:
            # W&B automatically logs the config provided during init
            # We can update it if needed
            self.wandb.config.update(config_dict)
    
    def _flatten_dict(self, d, parent_key='', sep='.'):
        """Flatten a nested dictionary for TensorBoard hparams."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (int, float, str, bool)):
                items.append((new_key, v))
        return dict(items)
    
    # ---- Logging Methods ----
    
    def log_scalar(self, name: str, value: float, step: int = None):
        """Log a scalar value to all active backends.
        
        Args:
            name: Name of the scalar
            value: Value to log
            step: Training step
        """
        if self.world_rank != 0:
            return
            
        if self.tensorboard:
            self.tensorboard.add_scalar(name, value, step)
            self.tensorboard.flush()  # Explicitly flush after each write
        
        if self.wandb:
            self.wandb.log({name: value}, step=step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int = None):
        """Log multiple scalars to all active backends.
        
        Args:
            main_tag: Main tag name
            tag_scalar_dict: Dictionary of tag names and values
            step: Training step
        """
        if self.world_rank != 0:
            return
            
        if self.tensorboard:
            # Instead of using add_scalars which creates subdirectories,
            # use individual add_scalar calls with prefixed tags
            for tag, value in tag_scalar_dict.items():
                tag_name = f"{main_tag}/{tag}"
                self.tensorboard.add_scalar(tag_name, value, step)
            self.tensorboard.flush()  # Explicitly flush after each write
        
        if self.wandb:
            # Prefix all keys with main_tag for W&B
            wandb_dict = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
            self.wandb.log(wandb_dict, step=step)
    
    def log_image(self, name: str, image: np.ndarray, step: int = None, save_to_disk: bool = False):
        """Log an image to all active backends.
        
        Args:
            name: Name of the image
            image: Image as numpy array
            step: Training step
            save_to_disk: Whether to also save the image to the images directory
        """
        if self.world_rank != 0:
            return
            
        if self.tensorboard:
            self.tensorboard.add_image(name, image, step, dataformats='HWC')
            self.tensorboard.flush()  # Explicitly flush after each write
        
        if self.wandb:
            self.wandb.log({name: wandb.Image(image)}, step=step)
            
        if save_to_disk:
            self.save_image_to_disk(name, image, step)
    
    def log_figure(self, name: str, figure: plt.Figure, step: int = None, save_to_disk: bool = False):
        """Log a matplotlib figure to all active backends.
        
        Args:
            name: Name of the figure
            figure: Matplotlib figure
            step: Training step
            save_to_disk: Whether to also save the figure to the images directory
        """
        if self.world_rank != 0:
            return
            
        if self.tensorboard:
            self.tensorboard.add_figure(name, figure, step)
            self.tensorboard.flush()  # Explicitly flush after each write
        
        if self.wandb:
            self.wandb.log({name: wandb.Image(figure)}, step=step)
            
        if save_to_disk:
            self.save_figure_to_disk(name, figure, step)
    
    def log_histogram(self, name: str, values: np.ndarray, step: int = None, bins: str = 'auto'):
        """Log a histogram to all active backends.
        
        Args:
            name: Name of the histogram
            values: Values to create histogram from
            step: Training step
            bins: Number of bins or method (passed to numpy.histogram)
        """
        if self.world_rank != 0:
            return
            
        if self.tensorboard:
            self.tensorboard.add_histogram(name, values, step, bins=bins)
            self.tensorboard.flush()  # Explicitly flush after each write
        
        if self.wandb:
            self.wandb.log({name: wandb.Histogram(values)}, step=step)
    
    def log_model_parameters(self, model: torch.nn.Module, step: int = None):
        """Log model parameter histograms.
        
        Args:
            model: PyTorch model
            step: Training step
        """
        if self.world_rank != 0:
            return
            
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.log_histogram(f"parameters/{name}", param.detach().cpu().numpy(), step)
                
                if param.grad is not None:
                    self.log_histogram(f"gradients/{name}", param.grad.detach().cpu().numpy(), step)
    
    def close(self):
        """Close all logging backends."""
        if self.world_rank != 0:
            return
            
        if self.tensorboard:
            # Make sure all data is written before closing
            self.tensorboard.flush()
            self.tensorboard.close()
            
        if self.wandb:
            wandb.finish()
            
        self.logger.info("Metrics logger closed")
    
    def save_image_to_disk(self, name: str, image: np.ndarray, step: int = None):
        """Save an image to the images directory.
        
        Args:
            name: Name of the image (will be used as filename)
            image: Image as numpy array
            step: Training step (will be included in filename if provided)
        """
        if self.world_rank != 0:
            return
            
        # Replace any characters that wouldn't work in a filename
        safe_name = name.replace('/', '_').replace(' ', '_')
        
        # Add step to filename if provided
        if step is not None:
            filename = f"{safe_name}_{step}.png"
        else:
            filename = f"{safe_name}.png"
            
        # Create the path to the images directory
        images_dir = self.run_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the image
        try:
            import cv2
            cv2.imwrite(str(images_dir / filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self.logger.info(f"Saved image to {images_dir / filename}")
        except ImportError:
            try:
                from PIL import Image
                Image.fromarray(image).save(images_dir / filename)
                self.logger.info(f"Saved image to {images_dir / filename}")
            except ImportError:
                self.logger.warning("Could not save image to disk: neither cv2 nor PIL is available")
                
    def save_figure_to_disk(self, name: str, figure: plt.Figure, step: int = None):
        """Save a matplotlib figure to the images directory.
        
        Args:
            name: Name of the figure (will be used as filename)
            figure: Matplotlib figure
            step: Training step (will be included in filename if provided)
        """
        if self.world_rank != 0:
            return
            
        # Replace any characters that wouldn't work in a filename
        safe_name = name.replace('/', '_').replace(' ', '_')
        
        # Add step to filename if provided
        if step is not None:
            filename = f"{safe_name}_{step}.png"
        else:
            filename = f"{safe_name}.png"
            
        # Create the path to the images directory
        images_dir = self.run_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the figure
        figure_path = images_dir / filename
        figure.savefig(figure_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Saved figure to {figure_path}")

# For backward compatibility - original functions
def initialize_wandb(
    config: Any, 
    project_name: str, 
    run_name: str, 
    strategy: str,
    precision: str,
    devices: int,
    num_nodes: int
) -> None:
    """Initialize wandb with detailed configuration."""
    if int(os.environ.get('RANK', 0)) != 0:
        # Only initialize wandb on rank 0
        return
        
    # Get wandb mode from environment or use online by default
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    
    from dataclasses import asdict
    
    wandb.init(
        project=project_name,
        name=run_name,
        mode=wandb_mode,
        config={
            "model": asdict(config.model),
            "training": asdict(config.training),
            "experiment": asdict(config),
            "platform": {
                "strategy": strategy,
                "precision": precision,
                "devices": devices,
                "num_nodes": num_nodes,
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            }
        },
        # Enable system tracking for more detailed metrics
        monitor_gym=True,
        config_exclude_keys=["data_path", "checkpoint_dir"]
    )
    
    # Print info about wandb mode
    logging.info(f"Weights & Biases initialized in {wandb_mode} mode")
    if wandb_mode in ["offline", "dryrun"]:
        logging.info("To view metrics locally after training, start a local server with 'wandb local'")
        logging.info("Then sync your runs with 'wandb sync wandb/run-*'")

def log_training_batch(
    logger: Logger, 
    batch_idx: int,
    epoch: int,
    loss: float,
    train_loader_len: int,
    learning_rate: Optional[float] = None,
    system_metrics: Optional[Dict[str, float]] = None
) -> None:
    """Log metrics for a training batch.
    
    Args:
        logger: Logger instance
        batch_idx: Current batch index
        epoch: Current epoch number
        loss: Current loss value
        train_loader_len: Length of training loader
        learning_rate: Current learning rate
        system_metrics: Dictionary of system metrics
    """
    # Only log from rank 0 (this check is already in logger.log_scalar)
    step = epoch * train_loader_len + batch_idx
    
    # Log metrics
    logger.log_scalar('train/loss', loss, step)
    logger.log_scalar('train/batch', batch_idx, step)
    logger.log_scalar('train/epoch', epoch, step)
    logger.log_scalar('train/progress', batch_idx / train_loader_len, step)
    
    # Log learning rate if provided
    if learning_rate is not None:
        logger.log_scalar('train/learning_rate', learning_rate, step)
    
    # Add system metrics if available
    if system_metrics:
        for k, v in system_metrics.items():
            logger.log_scalar(f"system/{k}", v, step)

def log_validation_results(
    logger: Logger,
    epoch: int,
    val_loss: float,
    val_metrics: Dict[str, float],
    step: int,
    learning_rate: Optional[float] = None,
    vis_y_pred: Optional[torch.Tensor] = None,
    vis_y: Optional[torch.Tensor] = None,
    config: Optional[Config] = None,
    save_to_disk: bool = True
) -> None:
    """Log validation results and visualizations.
    
    Args:
        logger: Logger instance
        epoch: Current epoch number
        val_loss: Validation loss value
        val_metrics: Dictionary of validation metrics
        step: Current global step
        learning_rate: Current learning rate
        vis_y_pred: Model predictions tensor for visualization
        vis_y: Ground truth tensor for visualization
        config: Configuration object
        save_to_disk: Whether to save visualizations to disk
    """
    # Basic metrics
    logger.log_scalar('val/loss', val_loss, step)
    logger.log_scalar('val/epoch', epoch, step)
    
    # Log all validation metrics
    val_metric_dict = {}
    for k, v in val_metrics.items():
        val_metric_dict[k] = v
    
    logger.log_scalars('val', val_metric_dict, step)
    
    # Log learning rate if provided
    if learning_rate is not None:
        logger.log_scalar('train/learning_rate', learning_rate, step)
    
    # Create visualizations if data is available
    if vis_y_pred is not None and vis_y is not None:
        create_and_log_visualizations(logger, epoch, step, vis_y_pred, vis_y, save_to_disk)

def create_and_log_visualizations(
    logger: Logger,
    epoch: int,
    step: int,
    vis_output: torch.Tensor,
    vis_target_data: torch.Tensor,
    save_to_disk: bool = True,
    vmin: float = None,
    vmax: float = None,
    error_max: float = None,
    diff_max: float = None
) -> None:
    """Create and log visualizations of model predictions.
    
    Args:
        logger: Logger instance
        epoch: Current epoch number
        step: Current step number
        vis_output: Tensor with model predictions
        vis_target_data: Tensor with ground truth data
        save_to_disk: Whether to save visualizations to disk
        vmin: Optional minimum value for the colorbar (for consistent scales)
        vmax: Optional maximum value for the colorbar (for consistent scales)
        error_max: Optional maximum error value for error plots (for consistent scales)
        diff_max: Optional maximum difference value for difference plots (for consistent scales)
    """
    # Create visualizations for a subset of samples
    for i in range(min(2, vis_output.size(0))):  # Only use first 2 samples
        # Create a lat-lon error plot (averaged across all channels)
        fig = create_lat_lon_error_plots(
            vis_output[i:i+1],
            vis_target_data[i:i+1],
            epoch=epoch,
            error_max=error_max
        )
        
        # Log the lat-lon error plot
        logger.log_figure(
            f"val_lat_lon_error/sample_{i}", 
            fig, 
            step=step,
            save_to_disk=save_to_disk
        )
        plt.close(fig)
        
        # Create a single comparison figure for the first channel only
        fig = create_comparison_figure(
            vis_output[i:i+1],
            vis_target_data[i:i+1],
            channel_idx=0,
            channel_name="first_channel",
            vmin=vmin,
            vmax=vmax,
            error_max=error_max
        )
        
        logger.log_figure(
            f"val_comparison/sample_{i}", 
            fig, 
            step=step,
            save_to_disk=save_to_disk
        )
        plt.close(fig)
        
        # Create mean error map across all channels
        fig = create_spatial_error_map(
            vis_output[i:i+1],
            vis_target_data[i:i+1],
            epoch=epoch,
            error_max=error_max
        )
        
        logger.log_figure(
            f"val_error/sample_{i}", 
            fig, 
            step=step,
            save_to_disk=save_to_disk
        )
        plt.close(fig)
        
        # Create a difference map for just the first channel
        fig = create_difference_map(
            vis_output[i:i+1],
            vis_target_data[i:i+1],
            channel_idx=0,
            channel_name="first_channel",
            epoch=epoch,
            diff_max=diff_max
        )
        
        logger.log_figure(
            f"val_difference/sample_{i}", 
            fig, 
            step=step,
            save_to_disk=save_to_disk
        )
        plt.close(fig) 