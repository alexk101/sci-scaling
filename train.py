import os
import torch
import torch.optim as optim
import argparse
from dataclasses import asdict
from lightning.fabric import Fabric
from lightning.fabric.strategies import FSDPStrategy
from typing import Optional, Union
from tqdm.contrib.logging import tqdm_logging_redirect
from pathlib import Path
import logging
import shutil
import signal
import numpy as np
import gc  # For garbage collection

# Global flag to indicate if we should terminate due to SLURM signal
received_term_signal = False

# Signal handler for graceful termination
def handle_termination_signal(signum, frame):
    global received_term_signal
    signal_name = signal.Signals(signum).name
    logging.warning(f"Received termination signal {signal_name} ({signum}). Will save checkpoint and exit.")
    received_term_signal = True

# Register signal handlers
signal.signal(signal.SIGUSR1, handle_termination_signal)  # SLURM time limit signal
signal.signal(signal.SIGTERM, handle_termination_signal)  # Standard termination signal

# Import DeepSpeed FLOPs profiler
try:
    from deepspeed.profiling.flops_profiler import FlopsProfiler
    FLOPS_PROFILER_AVAILABLE = True
    logging.info("DeepSpeed FLOPs profiler available")
except ImportError:
    FLOPS_PROFILER_AVAILABLE = False
    logging.warning("DeepSpeed FLOPs profiler not available. Install with: pip install deepspeed")

# Set matplotlib backend to non-interactive Agg to avoid display issues
import matplotlib
matplotlib.use('Agg')

from models.weather_transformer import WeatherTransformer, FLASH_ATTN_AVAILABLE
from utils.data_loader import WeatherDataset, get_data_loader
from utils.loss import l2_loss_opt
from utils.metrics import MetricsTracker, compute_metrics, weighted_rmse_channels
from utils.logging import (
    Logger, 
    init_run_directory, 
    log_training_batch, 
    log_validation_results, 
    create_and_log_visualizations
)   
from utils.config_utils import (
    Config,
    load_config,
    PrecisionType,
    LoggingBackend
)
from utils.average_meter import AverageMeter


class WeatherTrainer:
    """Trainer class for WeatherTransformer model."""
    
    def __init__(
        self,
        config: Config,
        strategy: str = "deepspeed",
        precision: Union[str, PrecisionType] = PrecisionType.BF16,
        devices: Optional[int] = None,
        num_nodes: int = 1,
        logging_backend: Union[str, LoggingBackend] = LoggingBackend.TENSORBOARD,
        max_checkpoints: int = 3,
        enable_flops_profiler: bool = False,
        ds_config_path: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object
            strategy: Training strategy (deepspeed, ddp, etc.)
            precision: Mixed precision type (bf16, fp16, fp32)
            devices: Number of devices to use (if None, will be determined automatically)
            num_nodes: Number of nodes to use
            logging_backend: Logging backend (tensorboard, wandb, both)
            max_checkpoints: Maximum number of checkpoints to keep
            enable_flops_profiler: Whether to enable FLOPs profiling
            ds_config_path: Path to DeepSpeed config file
        """
        self.config = config
        self.strategy = strategy
        self.precision = precision
        
        self.devices = devices
        
        # Determine number of nodes
        if 'SLURM_NNODES' in os.environ:
            self.num_nodes = int(os.environ['SLURM_NNODES'])
            logging.info(f"Using {self.num_nodes} nodes from SLURM allocation")
        else:
            self.num_nodes = num_nodes
            logging.info(f"Using {self.num_nodes} nodes from config")
        
        self.logging_backend = logging_backend
        self.max_checkpoints = max_checkpoints
        self.enable_flops_profiler = enable_flops_profiler
        self.ds_config_path = ds_config_path
        
        # Initialize model and optimizer
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Initialize data loaders
        self.train_loader = None
        self.val_loader = None
        
        # Initialize fabric and logger
        self.fabric = None
        self.metrics_logger = None
        
        # Initialize visualization scale values for consistency across epochs
        self.vis_vmin = None
        self.vis_vmax = None
        self.vis_error_max = None
        self.vis_diff_max = None
        
        # Initialize FLOPs profiler if enabled
        self.flops_profiler = None
        if enable_flops_profiler and FLOPS_PROFILER_AVAILABLE:
            self.flops_profiler = FlopsProfiler(self.model)
    
    def setup(self, resume_from_checkpoint=None):
        """Set up the trainer.
        
        Args:
            resume_from_checkpoint: Path to a checkpoint to resume from
        """
        # Log environment information
        if torch.cuda.is_available():
            logging.info(f"CUDA available: {torch.cuda.is_available()}")
            logging.info(f"CUDA device count: {torch.cuda.device_count()}")
            logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        
        # Log distributed environment variables
        distributed_vars = [
            "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", 
            "LOCAL_RANK", "NODE_RANK", "SLURM_PROCID", "SLURM_LOCALID",
            "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_NODEID",
            "SLURM_JOB_NUM_NODES", "SLURM_NTASKS", "SLURM_NTASKS_PER_NODE"
        ]
        
        for var in distributed_vars:
            if var in os.environ:
                logging.info(f"{var}: {os.environ[var]}")
        
        # Apply performance optimizations from config
        if torch.cuda.is_available() and hasattr(self.config.model, 'allow_tf32'):
            # Set TF32 option from config if device supports it
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = self.config.model.allow_tf32
                logging.info(f"Setting CUDA TF32 to: {self.config.model.allow_tf32}")
            
            # Set matmul precision based on TF32 setting
            if self.config.model.allow_tf32:
                torch.set_float32_matmul_precision('high')
                logging.info("Setting float32 matmul precision to 'high'")
        
        # Set up model
        model_params = {
            'img_size': self.config.model.img_size,
            'patch_size': self.config.model.patch_size,
            'in_channels': self.config.model.input_channels,
            'out_channels': self.config.model.output_channels,
            'embed_dim': self.config.model.embed_dim,
            'num_heads': self.config.model.num_heads,
            'num_layers': self.config.model.num_layers,
            'qkv_bias': self.config.model.qkv_bias,
            'drop_rate': self.config.model.dropout,
            'attn_drop_rate': self.config.model.attention_dropout,
            'drop_path_rate': self.config.model.stochastic_depth,
            'use_flash_attention': self.config.model.use_flash_attention and FLASH_ATTN_AVAILABLE,
            'use_checkpointing': self.config.model.use_checkpointing,
            'precision': self.config.model.precision
        }
        
        self.model = WeatherTransformer(**model_params)
        self.metrics = MetricsTracker(self.model)
        
        # Ensure devices is properly set - it can't be None for Fabric
        # Default to using the number of local GPUs if not explicitly set
        if self.devices is None:
            if torch.cuda.is_available():
                self.devices = torch.cuda.device_count()
            else:
                self.devices = 1  # Default to 1 CPU if no GPUs available
        
        logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        
        # Configure strategy with FSDP optimizations if using FSDP
        if 'fsdp' in str(self.strategy).lower() and not isinstance(self.strategy, FSDPStrategy):
            try:
                # Create FSDPStrategy with optimized parameters
                from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload, ShardingStrategy
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
                import functools
                
                logging.info("Configuring Lightning Fabric FSDP strategy with memory optimizations")
                
                # Define auto-wrap policy based on parameter count
                min_params = 1e5  # Wrap layers with more than 100K parameters
                auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy, min_num_params=min_params
                )
                
                # Configure activation checkpointing for transformer blocks
                activation_checkpointing_policy = None
                if hasattr(self.model, 'blocks'):
                    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
                    from torch.nn.modules.transformer import TransformerEncoderLayer
                    
                    # Use native transformer blocks for checkpointing if available
                    if hasattr(self.model.blocks[0], '__class__'):
                        activation_checkpointing_policy = {self.model.blocks[0].__class__}
                        logging.info(f"Using activation checkpointing for {self.model.blocks[0].__class__.__name__}")
                
                # Create FSDP strategy with memory optimizations
                self.strategy = FSDPStrategy(
                    auto_wrap_policy=auto_wrap_policy,
                    activation_checkpointing_policy=activation_checkpointing_policy,
                    cpu_offload=CPUOffload(offload_params=True) if self.config.model.use_checkpointing else None,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    state_dict_type="full"  # Save full state dict for easier checkpoint handling
                )
                
                logging.info("Successfully configured FSDP strategy with memory optimizations")
            
            except ImportError as e:
                logging.warning(f"Could not configure FSDP strategy: {e}")
                logging.warning("Will use default FSDP settings")
        
        # Set up fabric with the configured strategy
        self.fabric = Fabric(
            strategy=self.strategy,
            precision=self.precision,
            devices=self.devices,
            num_nodes=self.num_nodes,
            loggers=None  # We'll set up our own logger
        )
        
        # Launch fabric before setting up model
        self.fabric.launch()
        logging.info(f"Fabric launched successfully")
        
        # Now that Fabric is initialized and distributed environment is set up, create the logger
        self.metrics_logger = Logger(
            run_dir=Path(self.config.training.log_dir),  # Use the run directory directly from config
            config=self.config,
            backend=self.logging_backend,
            project_name=self.config.training.wandb_project if self.config.training.use_wandb else "weather-transformer"
        )
        
        logging.info(f"Logger created")
        # Determine if we're using distributed training
        using_distributed = (
            self.strategy == "deepspeed" or
            self.strategy == "ddp" or
            (isinstance(self.strategy, str) and "ddp" in self.strategy) or
            isinstance(self.strategy, FSDPStrategy)
        )
        
        logging.info(f"Using distributed: {using_distributed}")
        
        # Set up optimizer with gradient clipping
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Only create scheduler for non-DeepSpeed strategies
        if self.strategy != "deepspeed":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs
            )
        else:
            self.scheduler = None
        
        # Create datasets
        train_dataset = WeatherDataset(
            data_path=self.config.training.train_data_path,
            dt=self.config.training.dt,
            img_size=self.config.model.img_size,
            input_channels=self.config.model.input_channels,
            output_channels=self.config.model.output_channels,
            normalize=self.config.training.normalize,
            means_path=self.config.training.means_path,
            stds_path=self.config.training.stds_path,
            train=True
        )
        
        val_dataset = WeatherDataset(
            data_path=self.config.training.valid_data_path,
            dt=self.config.training.dt,
            img_size=self.config.model.img_size,
            input_channels=self.config.model.input_channels,
            output_channels=self.config.model.output_channels,
            normalize=self.config.training.normalize,
            means_path=self.config.training.means_path,
            stds_path=self.config.training.stds_path,
            train=False
        )
        
        logging.info(f'Initializing training data loader')
        # Set up data loaders with appropriate sampling for distributed training - AFTER Fabric is launched
        self.train_loader, self.train_sampler = get_data_loader(
            dataset=train_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            distributed=using_distributed,
            train=True,
            drop_last=True,
            world_size=self.fabric.world_size if using_distributed else None,
            rank=self.fabric.global_rank if using_distributed else None
        )
        
        logging.info(f'Initializing validation data loader')
        self.val_loader, self.val_sampler = get_data_loader(
            dataset=val_dataset,
            batch_size=self.config.training.batch_size // 4,  # Use smaller batch size for validation
            num_workers=self.config.training.num_workers,
            distributed=using_distributed,
            train=False,
            drop_last=False,
            world_size=self.fabric.world_size if using_distributed else None,
            rank=self.fabric.global_rank if using_distributed else None
        )
        logging.info(f'Data loaders initialized')
        
        # Set up model, optimizer, and data loaders with fabric
        logging.info(f'Setting up model, optimizer, and data loaders with fabric')
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        logging.info(f'Model, optimizer, and data loaders set up with fabric')

        logging.info(f'Setting up train and val data loaders with fabric')
        self.train_loader, self.val_loader = self.fabric.setup_dataloaders(
            self.train_loader, self.val_loader
        )
        logging.info(f'Train and val data loaders set up with fabric')
        
        # Load checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        # Initialize FLOPs profiler if enabled
        if self.enable_flops_profiler and FLOPS_PROFILER_AVAILABLE:
            self.flops_profiler = FlopsProfiler(self.model)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint to load
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Check if this is a DeepSpeed universal checkpoint
        is_universal = False
        
        # Look for the universal version of the checkpoint if available
        universal_path = checkpoint_path.parent / f"{checkpoint_path.name}_universal"
        if universal_path.exists() and universal_path.is_dir():
            checkpoint_path = universal_path
            is_universal = True
            logging.info(f"Loading from universal checkpoint: {checkpoint_path}")
        
        # Different loading logic for universal vs. regular checkpoints
        if is_universal and self.strategy == "deepspeed":
            # Load from universal checkpoint using DeepSpeed's mechanism
            # This will be handled automatically by DeepSpeed + Fabric
            try:
                # Add --universal-checkpoint to DeepSpeed config if not already present
                import deepspeed
                if hasattr(self.fabric.strategy, "config"):
                    ds_config = self.fabric.strategy.config
                    if 'checkpoint' not in ds_config:
                        ds_config['checkpoint'] = {}
                    if 'universal_checkpoint' not in ds_config['checkpoint']:
                        ds_config['checkpoint']['universal_checkpoint'] = True
                
                # Use Fabric's load with the universal checkpoint
                logging.info(f"Loading universal checkpoint from: {checkpoint_path}")
                checkpoint = self.fabric.load(str(checkpoint_path))
                
                # Extract and apply the non-model parts of the checkpoint
                if 'epoch' in checkpoint:
                    self.config.training.start_epoch = checkpoint['epoch']
                    logging.info(f"Resuming from epoch {checkpoint['epoch']}")
                
                # Note: DeepSpeed + Universal checkpoint will handle model parameters automatically
                
                # Load optimizer state if available
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load scheduler state if available and scheduler exists
                if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                logging.info(f"Successfully loaded universal checkpoint from {checkpoint_path}")
                return True
            except Exception as e:
                logging.error(f"Error loading universal checkpoint: {str(e)}", exc_info=True)
                return False
        else:
            # Regular checkpoint loading
            try:
                logging.info(f"Loading checkpoint from: {checkpoint_path}")
                checkpoint = self.fabric.load(str(checkpoint_path))
                
                # Load model state if available
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    # For DeepSpeed checkpoints, Fabric handles this differently
                    pass
                
                # Load optimizer state if available
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load scheduler state if available and scheduler exists
                if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # Update the starting epoch if available
                if 'epoch' in checkpoint:
                    logging.info(f"Resuming from epoch {checkpoint['epoch']}")
                
                logging.info(f"Successfully loaded checkpoint from {checkpoint_path}")
                return True
            except Exception as e:
                logging.error(f"Error loading checkpoint: {str(e)}", exc_info=True)
                return False
                
    def train(self, resume_from_checkpoint=None):
        """Train the model.
        
        Args:
            resume_from_checkpoint: Path to a checkpoint to resume from
        """
        self.setup(resume_from_checkpoint=resume_from_checkpoint)
        
        # Log training information
        self._log_training_start()
        
        # Determine starting epoch
        start_epoch = 0
        if hasattr(self.config.training, 'start_epoch'):
            start_epoch = self.config.training.start_epoch
        
        # Training loop
        for epoch in range(start_epoch, self.config.training.epochs):
            # Check for termination signal before each epoch
            if received_term_signal:
                logging.warning("Detected termination signal before starting epoch. Saving checkpoint and exiting.")
                self.save_checkpoint(epoch)
                return
                
            train_loss = self._train_epoch(epoch)
            logging.info(f"Finished epoch {epoch+1}, Avg Loss: {train_loss.avg:.6f}")
            self._validate_epoch(epoch)
            
            # Save checkpoint if needed
            # Note: all ranks should call save_checkpoint to ensure proper synchronization
            # but only rank 0 should log the message
            if (epoch + 1) % self.config.training.save_every == 0:
                if self.fabric.is_global_zero:
                    logging.info(f"Saving checkpoint at epoch {epoch+1} (save_every={self.config.training.save_every})")
                self.save_checkpoint(epoch + 1)
            
            # Check for termination signal after each epoch
            if received_term_signal:
                logging.warning("Detected termination signal after completing epoch. Exiting.")
                return
        
        # Save final checkpoint
        if self.fabric.is_global_zero:
            logging.info(f"Saving final checkpoint")
        self.save_checkpoint(self.config.training.epochs)
            
        logging.info(f"Training completed")
    
    def _log_training_start(self):
        """Log information at the start of training."""
        logging.info(f"Starting training with {self.config.training.epochs} epochs")
        logging.info(f"Training on {self.num_nodes} nodes, {self.devices} devices per node")
        logging.info(f"Using {self.strategy} strategy with {self.precision} precision")
        
        # Log rank-specific info using fabric's global rank (safer)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            logging.info(f"Training on GPU: {gpu_name}")
        else:
            logging.info(f"Training on CPU")
    
    def _train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            AverageMeter with training loss statistics
        """
        self.model.train()
        train_loss = AverageMeter()
        
        # Log epoch start
        logging.info(f"Starting epoch {epoch+1}/{self.config.training.epochs}")
        
        # Set epoch for DistributedSampler to ensure proper shuffling
        if hasattr(self, 'train_sampler') and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
            logging.info(f"Set epoch {epoch} in DistributedSampler for proper shuffling")
        
        # Only show progress bar on rank 0, but redirect logging for all ranks
        should_display_pbar = self.fabric.is_global_zero
        
        # Use tqdm_logging_redirect which combines tqdm with logging redirection
        with tqdm_logging_redirect(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.training.epochs}",
            disable=not should_display_pbar
        ) as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Check for termination signal periodically (every 10 batches)
                if batch_idx % 10 == 0 and received_term_signal:
                    logging.warning(f"Detected termination signal during epoch {epoch+1}, batch {batch_idx}. "
                                   f"Will finish current epoch and save checkpoint.")
                    break
                    
                # Unpack the batch tuple
                input_data, target_data = batch
                
                # Zero gradients before forward pass
                self.optimizer.zero_grad()
                
                # Forward pass with Fabric's autocast for automatic precision handling
                with self.fabric.autocast():
                    output = self.model(input_data)
                    loss = l2_loss_opt(output, target_data)
                
                # Use native fabric backward pass with memory optimizations for FSDP
                self.fabric.backward(loss)
                
                # Step the optimizer
                self.optimizer.step()
                
                # Update metrics
                train_loss.update(loss.item())
                
                # Update progress bar with current loss
                if should_display_pbar:
                    pbar.set_postfix(loss=f"{loss.item():.6f}")
                
                # Periodically log batch info to avoid flooding the log
                if batch_idx % 10 == 0:
                    logging.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
                
                # Log metrics
                if self.fabric.is_global_zero:
                    self._log_training_metrics(epoch, batch_idx, loss.item())
        
        # Log epoch completion (will be seen by all ranks)
        logging.info(f"Completed epoch {epoch+1}/{self.config.training.epochs} with avg loss: {train_loss.avg:.6f}")
        
        return train_loss
    
    def _log_training_metrics(self, epoch, batch_idx, loss):
        """
        Log training metrics at appropriate frequency.
        
        Args:
            epoch: Current epoch number
            batch_idx: Current batch index
            loss: Current loss value
        """
        # Calculate global step
        step = epoch * len(self.train_loader) + batch_idx
        
        # Get log frequency from config (default to 1 if not specified)
        log_frequency = getattr(self.config.training, 'log_frequency', 1)
        
        # Log at specified frequency or at the end of the epoch
        is_last_batch = batch_idx == len(self.train_loader) - 1
        if batch_idx % log_frequency == 0 or is_last_batch:
            # Use the function from utils.logging
            log_training_batch(
                logger=self.metrics_logger,
                batch_idx=batch_idx,
                epoch=epoch,
                loss=loss,
                train_loader_len=len(self.train_loader),
                learning_rate=self.optimizer.param_groups[0]['lr']
            )
    
    def _validate_epoch(self, epoch):
        """
        Validate the model on the validation set following the approach in SC24.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (validation loss, validation metrics)
        """
        # Set model to evaluation mode
        self.model.eval()
        
        if self.fabric.is_global_zero:
            logging.info("Starting validation...")
            log_memory_usage("Validation Start")
        
        # Initialize accumulators with zero tensors on the device (following SC24 approach)
        device = next(self.model.parameters()).device
        val_loss = torch.zeros(1, device=device)
        val_rmse = torch.zeros(self.config.model.output_channels, dtype=torch.float32, device=device)
        valid_steps = 0
        
        # Validation loop following SC24 implementation pattern
        with torch.inference_mode():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                # Log diagnostic info
                if batch_idx % 10 == 0 and self.fabric.is_global_zero:
                    logging.info(f"Validating batch {batch_idx}, steps: {valid_steps}")
                
                # Forward pass with autocast for precision handling
                with self.fabric.autocast():
                    outputs = self.model(inputs)
                    loss = l2_loss_opt(outputs, targets)
                    # Calculate weighted RMSE directly (matches SC24 pattern)
                    rmse = weighted_rmse_channels(outputs, targets)
                
                # Simple accumulation (not weighted by batch size) - following SC24
                val_loss += loss
                val_rmse += rmse.sum(dim=0)  # Sum across batch dimension
                valid_steps += 1
                
                # Free memory immediately
                del inputs, targets, outputs, loss, rmse
                
                # Occasional garbage collection
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # Final memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Average the accumulated metrics by step count (exactly as SC24 does)
        if valid_steps > 0:
            val_loss = val_loss / valid_steps
            val_rmse = val_rmse / valid_steps
        
        # Create metrics dictionary
        metrics = {
            "loss": val_loss.item(),
            "weighted_rmse": torch.mean(val_rmse).item()
        }
        
        # Add per-channel weighted RMSE
        for i in range(len(val_rmse)):
            metrics[f"weighted_rmse_ch{i}"] = val_rmse[i].item()
        
        # Handle distributed validation - average metrics across all processes
        if self.fabric.world_size > 1:
            # Create a tensor to hold all metrics for reduction
            metric_names = list(metrics.keys())
            metric_values = torch.tensor([metrics[name] for name in metric_names], device=device)
            
            # All-reduce to average across processes
            torch.distributed.all_reduce(metric_values, op=torch.distributed.ReduceOp.SUM)
            metric_values = metric_values / self.fabric.world_size
            
            # Update metrics dictionary with reduced values
            for i, name in enumerate(metric_names):
                metrics[name] = metric_values[i].item()
        
        # Log results
        self._log_validation_summary(epoch, metrics["loss"], metrics)
        self._log_validation_metrics(epoch, epoch * len(self.train_loader), metrics["loss"], metrics)
        
        # Log final memory usage
        if self.fabric.is_global_zero:
            log_memory_usage("Validation End")
        
        # Set model back to training mode
        self.model.train()
        
        return metrics["loss"], metrics
    
    def _update_visualization_scales(self, outputs, targets):
        """
        Update visualization scale parameters based on current batch data.
        This ensures consistent colorbar scales across epochs.
        
        Args:
            outputs: Model predictions tensor [batch, channels, height, width]
            targets: Ground truth tensor [batch, channels, height, width]
        """
        # Initialize global min/max values on first validation run if not already set
        if self.vis_vmin is None or self.vis_vmax is None:
            # Get data for the first channel
            output_values = outputs[:, 0].cpu().numpy()
            target_values = targets[:, 0].cpu().numpy()
            self.vis_vmin = float(min(np.min(output_values), np.min(target_values)))
            self.vis_vmax = float(max(np.max(output_values), np.max(target_values)))
            logging.info(f"Setting visualization vmin={self.vis_vmin}, vmax={self.vis_vmax}")
            
        # Calculate and potentially update error_max
        errors = torch.abs(outputs - targets).cpu().numpy()
        current_error_max = float(np.max(errors))
        if self.vis_error_max is None or current_error_max > self.vis_error_max:
            self.vis_error_max = current_error_max
            logging.info(f"Setting visualization error_max={self.vis_error_max}")
            
        # Calculate and potentially update diff_max
        diffs = (outputs - targets).cpu().numpy()
        current_diff_max = float(max(abs(np.min(diffs)), abs(np.max(diffs))))
        if self.vis_diff_max is None or current_diff_max > self.vis_diff_max:
            self.vis_diff_max = current_diff_max
            logging.info(f"Setting visualization diff_max={self.vis_diff_max}")
    
    def _log_validation_summary(self, epoch, val_loss, val_metrics):
        """
        Log validation summary to console.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss value
            val_metrics: Dictionary of metrics
        """
        logging.info(f"Validation: Epoch {epoch+1}, Loss: {val_loss:.6f}")
        # Log only the main metrics to avoid cluttering the console
        for k in ["mse", "mae", "rmse", "weighted_rmse"]:
            if k in val_metrics:
                logging.info(f"Validation: {k}={val_metrics[k]:.6f}")
    
    def _log_validation_metrics(self, epoch, step, val_loss, val_metrics):
        """
        Log validation metrics to logger.
        
        Args:
            epoch: Current epoch number
            step: Current step number
            val_loss: Validation loss
            val_metrics: Dictionary of metrics
        """
        # Use the function from utils.logging
        log_validation_results(
            logger=self.metrics_logger,
            epoch=epoch,
            val_loss=val_loss,
            val_metrics=val_metrics,
            step=step,
            learning_rate=self.optimizer.param_groups[0]['lr']
        )
    
    def _create_and_log_visualizations(self, epoch, step, vis_output, vis_target_data):
        """
        Create and log visualizations of model predictions.
        
        Args:
            epoch: Current epoch number
            step: Current step number
            vis_output: Tensor with model predictions
            vis_target_data: Tensor with ground truth data
        """
        # Generate and log visualizations if we have sample data
        if vis_output is None or vis_target_data is None:
            return
            
        # Only create visualizations at specified frequency
        log_frequency = getattr(self.config.training, 'log_frequency', 1)
        if (epoch + 1) % log_frequency != 0:
            return
            
        # Use the function from utils.logging with consistent scale parameters
        create_and_log_visualizations(
            logger=self.metrics_logger, 
            epoch=epoch, 
            step=step, 
            vis_output=vis_output, 
            vis_target_data=vis_target_data, 
            save_to_disk=True,
            vmin=self.vis_vmin,
            vmax=self.vis_vmax,
            error_max=self.vis_error_max,
            diff_max=self.vis_diff_max
        )
    
    def convert_to_universal_checkpoint(self, checkpoint_path):
        """
        Convert a DeepSpeed checkpoint to Universal Checkpoint format.
        
        Args:
            checkpoint_path: Path to the DeepSpeed checkpoint
        """
        # Skip if not using DeepSpeed
        if self.strategy != "deepspeed":
            return
            
        # Only run on rank 0 to avoid conflicts
        if not self.fabric.is_global_zero:
            return
            
        try:
            # Make sure the checkpoint path exists and is a directory (DeepSpeed format)
            if not checkpoint_path.is_dir():
                logging.warning(f"Cannot convert checkpoint to universal format: {checkpoint_path} is not a directory")
                return
                
            # Path to the ds_to_universal.py script in DeepSpeed
            # This assumes DeepSpeed is installed through pip
            import deepspeed
            ds_path = Path(deepspeed.__file__).parent
            ds_to_universal_script = ds_path / "checkpoint" / "ds_to_universal.py"
            
            if not ds_to_universal_script.exists():
                logging.warning(f"Cannot convert checkpoint to universal format: {ds_to_universal_script} not found")
                return
                
            # Output path for universal checkpoint
            universal_checkpoint_path = checkpoint_path.parent / f"{checkpoint_path.name}_universal"
            
            # Run the conversion script
            import subprocess
            cmd = [
                f"{os.environ.get('CONDA_ENV_PATH', '/bin/false')}/bin/python", 
                str(ds_to_universal_script),
                "--input_folder", str(checkpoint_path/'checkpoint'),
                "--output_folder", str(universal_checkpoint_path),
                "--inject_missing_state"
            ]
            
            logging.info(f"Converting checkpoint to universal format: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"Successfully converted checkpoint to universal format at {universal_checkpoint_path}")
                
                # Create a 'latest_universal' file pointing to the universal checkpoint
                with open(checkpoint_path.parent / "latest_universal", "w") as f:
                    f.write(universal_checkpoint_path.name)

                logging.info(f"Removing old DeepSpeed checkpoint at {checkpoint_path}")
                shutil.rmtree(checkpoint_path)
                    
                return universal_checkpoint_path
            else:
                logging.error(f"Failed to convert checkpoint to universal format: {result.stderr}")
                return None
        except Exception as e:
            logging.error(f"Error converting checkpoint to universal format: {str(e)}", exc_info=True)
            return None

    def save_checkpoint(self, epoch: int):
        """
        Save a checkpoint.
        
        Args:
            epoch: Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'model': self.model,  # Include the model itself for DeepSpeed strategy
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        # Only add scheduler state if scheduler exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        checkpoint['config'] = asdict(self.config)
        
        # Save to the checkpoints directory within the run directory
        checkpoint_dir = Path(self.config.training.log_dir) / "checkpoints"
        
        # Create the directory if it doesn't exist (should already exist from init_run_directory)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint with epoch number
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        self.fabric.save(str(checkpoint_path), checkpoint)
        
        # Log the checkpoint save
        logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Convert to universal checkpoint format if using DeepSpeed
        if self.strategy == "deepspeed":
            # Wait for all ranks to finish saving
            self.fabric.barrier()
            self.convert_to_universal_checkpoint(checkpoint_path)
            
        # Remove old checkpoints if needed
        if self.max_checkpoints > 0 and self.fabric.is_global_zero:
            # Only rank 0 should handle the checkpoint cleanup
            logging.info("Rank 0 cleaning up old checkpoints")
            
            # Get all checkpoint files and their associated universal checkpoints
            all_checkpoints = []
            
            # Get regular DeepSpeed checkpoints
            ds_checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            for cp in ds_checkpoints:
                epoch_num = int(cp.stem.split('_')[-1].split('.')[0])
                all_checkpoints.append((epoch_num, cp, False))  # (epoch, path, is_universal)
                
            # Get universal checkpoints
            uni_checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt_universal'))
            for ucp in uni_checkpoints:
                # Extract the epoch number from the base filename (before _universal)
                base_name = ucp.name.split('_universal')[0]
                epoch_num = int(base_name.split('_')[-1].split('.')[0])
                all_checkpoints.append((epoch_num, ucp, True))  # (epoch, path, is_universal)
                
            # Sort by epoch number
            all_checkpoints.sort(key=lambda x: x[0])
            
            # Group checkpoints by epoch number
            epoch_groups = {}
            for epoch_num, path, is_universal in all_checkpoints:
                if epoch_num not in epoch_groups:
                    epoch_groups[epoch_num] = {"regular": None, "universal": None}
                
                if is_universal:
                    epoch_groups[epoch_num]["universal"] = path
                else:
                    epoch_groups[epoch_num]["regular"] = path
            
            # Sort epochs and keep only the newest max_checkpoints
            sorted_epochs = sorted(epoch_groups.keys())
            epochs_to_keep = sorted_epochs[-self.max_checkpoints:] if len(sorted_epochs) > self.max_checkpoints else sorted_epochs
            
            # Remove epochs not in the keep list
            for epoch_num in sorted_epochs:
                if epoch_num not in epochs_to_keep:
                    checkpoint_info = epoch_groups[epoch_num]
                    
                    # Remove regular checkpoint if it exists
                    if checkpoint_info["regular"] is not None and checkpoint_info["regular"].exists():
                        if checkpoint_info["regular"].is_dir():
                            shutil.rmtree(checkpoint_info["regular"])
                        else:
                            checkpoint_info["regular"].unlink()
                        logging.info(f"Removed old checkpoint: {checkpoint_info['regular']}")
                    
                    # Remove universal checkpoint if it exists
                    if checkpoint_info["universal"] is not None and checkpoint_info["universal"].exists():
                        if checkpoint_info["universal"].is_dir():
                            shutil.rmtree(checkpoint_info["universal"])
                        else:
                            checkpoint_info["universal"].unlink()
                        logging.info(f"Removed old universal checkpoint: {checkpoint_info['universal']}")
        
        # Ensure all ranks wait for cleanup to complete before continuing
        self.fabric.barrier()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Weather Transformer model")
    
    # Config options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--config", type=str, default="config/configs.yaml",
                            help="Path to the YAML configuration file")
    config_group.add_argument("--config_name", type=str, default="base",
                            help="Name of the configuration to use from the YAML file")

    # Training options
    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--strategy", type=str, default=None,
                              help="Training strategy (deepspeed, ddp, etc.). Overrides config value if provided.")
    training_group.add_argument("--precision", type=str, default=None,
                              help="Precision for training (fp32, fp16, bf16, etc.). Overrides config value if provided.")
    training_group.add_argument("--devices", type=int, default=None,
                              help="Number of devices to use. Overrides config value if provided.")
    training_group.add_argument("--num_nodes", type=int, default=None,
                              help="Number of nodes to use. Overrides config value if provided.")
    training_group.add_argument("--logging_backend", type=str, default=None,
                              help="Logging backend to use (tensorboard, wandb, both, none).")
    training_group.add_argument("--max_checkpoints", type=str, default=None,
                              help="Maximum number of checkpoints to keep.")
    training_group.add_argument("--resume_from_checkpoint", type=str, default=None,
                              help="Path to checkpoint to resume training from.")
    
    # Budget options
    budget_group = parser.add_argument_group("Budget")
    budget_group.add_argument("--flops_budget", type=float, default=None,
                            help="FLOPs budget for training (e.g., 1e20). Overrides config value if provided.")
    budget_group.add_argument("--time_budget", type=float, default=None,
                            help="Time budget in hours (e.g., 5.5). Overrides config value if provided.")
    
    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--ds_config", type=str, default=None,
                            help="Path to DeepSpeed configuration file")
    output_group.add_argument("--enable_flops_profiler", action="store_true",
                            help="Enable FLOPs profiler to measure compute efficiency")
    
    return parser.parse_args()

def main():
    """Main training function."""
    # Use RANK environment variable directly from export_DDP_vars.sh
    rank = int(os.environ.get('RANK', 0))
    
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize the run directory and configure logging
    run_dir = init_run_directory(config.training.log_dir)
    
    # Only log startup information on rank 0
    if rank == 0:
        # Log startup information
        logging.info(f"Starting Weather Transformer training process")
        if 'SLURM_JOB_ID' in os.environ:
            logging.info(f"Slurm Job ID: {os.environ.get('SLURM_JOB_ID')}")
            logging.info(f"Slurm Node: {os.environ.get('SLURMD_NODENAME', 'unknown')}")
        
        logging.info(f"Loaded configuration: {args.config}")
        logging.info(f"Run name: {run_dir.name}")
    
    # Update config to use the new directory structure
    config.training.log_dir = str(run_dir)
    
    # Handle devices parameter properly
    devices = args.devices or (
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    )
    if rank == 0:
        logging.info(f"Using {devices} devices per node")
    
    # Configure strategy based on args
    strategy = args.strategy or config.training.strategy
    
    # Initialize trainer with proper settings
    trainer = WeatherTrainer(
        config=config,
        strategy=strategy,
        precision=args.precision or config.training.precision,
        devices=devices,
        num_nodes=args.num_nodes or config.training.num_nodes if hasattr(config.training, 'num_nodes') else 1,
        logging_backend=args.logging_backend or config.training.logging_backend,
        max_checkpoints=int(args.max_checkpoints) if args.max_checkpoints else 3,
        enable_flops_profiler=args.enable_flops_profiler,
        ds_config_path=args.ds_config
    )
    
    try:
        # Train the model
        if args.resume_from_checkpoint:
            if rank == 0:
                logging.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        if rank == 0:
            logging.info(f"Training completed successfully")
    except Exception as e:
        # Log any uncaught exceptions
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise

def log_memory_usage(prefix="Memory"):
    """Log memory usage information for debugging."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        
        logging.info(f"{prefix} Usage: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")
        
        # Reset peak stats
        torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    main() 
